"""Run the current full handoff pipeline on WiseNET and score it afterward."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from multicam_reid.association.topology import (
    add_overlap_arguments,
    parse_overlap_pairs,
)
from multicam_reid.detection.yolo import YoloConfig, YoloPersonDetector
from multicam_reid.evaluation.wisenet import evaluate_wisenet
from multicam_reid.inputs.wisenet import WiseNETSequence
from multicam_reid.pipeline.camera_demo import (
    DEFAULT_CONFIG_PATH,
    build_reid_pipelines,
    load_camera_demo_config,
    with_overlap_overrides,
)
from multicam_reid.pipeline.handoff_replay import run_handoff_replay
from multicam_reid.reid.encoder import ReIDEncoder, ReIDEncoderConfig
from multicam_reid.reid.tracklets import TrackletEmbeddingConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/wisenet_set2"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cameras", type=int, nargs="+", default=[3, 4])
    add_overlap_arguments(parser)
    parser.add_argument("--stride", type=int, default=6, help="6 means 5 FPS at source 30 FPS")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/handoff/wisenet_set2"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate without loading models.")
    args = parser.parse_args(argv)
    if args.overlap is not None:
        try:
            parse_overlap_pairs(args.overlap, tuple(f"C{c}" for c in args.cameras), numeric=True)
        except ValueError as error:
            parser.error(str(error))
    sequence = WiseNETSequence(
        args.data_root, tuple(args.cameras), stride=args.stride, start=args.start, end=args.end
    )
    output = args.output_dir.resolve()
    if args.dry_run:
        models = with_overlap_overrides(
            load_camera_demo_config(args.config).models,
            args.overlap,
            sequence.camera_ids,
            numeric=True,
            reconciliation_enabled=False if args.no_reconcile else None,
        )
        print(
            json.dumps(
                {
                    "camera_ids": sequence.camera_ids,
                    "association_mode": models.association_mode,
                    "overlap_pairs": models.overlap_pairs,
                    "overlap_max_cosine_distance": models.overlap_max_cosine_distance,
                    "overlap_confirmations": models.overlap_confirmations,
                    "reconciliation_enabled": models.reconciliation_enabled,
                    "reconciliation_confirmations": models.reconciliation_confirmations,
                    "reconciliation_min_seconds": models.reconciliation_min_seconds,
                    "output_dir": str(output),
                },
                indent=2,
            )
        )
        return 0
    if output.exists() and not args.evaluate_only:
        parser.error("output directory exists; select a new --output-dir to preserve results")
    if not args.evaluate_only:
        config = load_camera_demo_config(args.config)
        models = with_overlap_overrides(
            config.models,
            args.overlap,
            sequence.camera_ids,
            numeric=True,
            reconciliation_enabled=False if args.no_reconcile else None,
        )
        detector = YoloPersonDetector(
            YoloConfig(
                model=models.detector_model,
                confidence=models.confidence,
                image_size=models.image_size,
                device=models.device,
            )
        )
        encoder = ReIDEncoder.from_checkpoint(
            models.checkpoint, ReIDEncoderConfig(device=models.device)
        )
        pipelines = build_reid_pipelines(
            sequence.camera_ids,
            detector,
            encoder,
            tracklet_config=TrackletEmbeddingConfig(
                models.history_size,
                models.embedding_interval,
                models.max_idle_updates,
            ),
        )
        summary = run_handoff_replay(
            sequence.root,
            pipelines=pipelines,
            models=models,
            output_dir=output,
            display=replace(config.display, enabled=args.show),
            freshness_seconds=config.runtime.association_window_seconds,
            camera_ids=sequence.camera_ids,
            frame_packets=sequence.frames(),
            video_fps=sequence.fps / sequence.stride,
        )
        summary["source"] = "WiseNET Set 2 full-scene videos"
        summary["sampling"] = {
            "stride": sequence.stride,
            "fps": sequence.fps,
            "start": sequence.start,
            "end": sequence.end,
        }
        summary["accuracy"] = "See evaluation.json; development diagnostic, not a benchmark."
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = evaluate_wisenet(sequence, output / "assignments.jsonl")
    events = [
        json.loads(line)
        for line in (output / "handoff_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report["logged_cross_camera_recoveries"] = sum(
        event["from_camera"] != event["to_camera"] for event in events
    )
    report["logged_same_camera_recoveries"] = sum(
        event["from_camera"] == event["to_camera"] for event in events
    )
    overlap_log = output / "overlap_events.jsonl"
    report["logged_overlap_matches"] = (
        len(overlap_log.read_text(encoding="utf-8").splitlines()) if overlap_log.exists() else 0
    )
    merge_log = output / "merge_events.jsonl"
    report["logged_reconciliation_merges"] = (
        len(merge_log.read_text(encoding="utf-8").splitlines()) if merge_log.exists() else 0
    )
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
