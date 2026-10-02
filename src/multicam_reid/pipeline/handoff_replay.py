"""Replay raw full-scene recordings through the common handoff inference pipeline."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import cv2

from multicam_reid.association.handoff import HandoffIdentityRegistry
from multicam_reid.association.topology import add_overlap_arguments, validate_overlap_pairs
from multicam_reid.detection.yolo import YoloConfig, YoloPersonDetector
from multicam_reid.inputs.handoff_recordings import iter_recorded_frames, recording_camera_ids
from multicam_reid.pipeline.camera_demo import (
    DEFAULT_CONFIG_PATH,
    DemoDisplayConfig,
    DemoModelConfig,
    IdentityDisplay,
    LivePipeline,
    ScreenshotController,
    build_association_registry,
    build_reid_pipelines,
    load_camera_demo_config,
    render_dashboard,
    save_dashboard_screenshot,
    with_overlap_overrides,
)
from multicam_reid.reid.encoder import ReIDEncoder, ReIDEncoderConfig
from multicam_reid.reid.tracklets import TrackKey, TrackletEmbeddingConfig
from multicam_reid.types import FramePacket
from multicam_reid.visualization.tracks import draw_global_boxes

DEFAULT_REPLAY_DISPLAY = DemoDisplayConfig(enabled=False)


def run_handoff_replay(
    recording: Path,
    *,
    pipelines: Mapping[str, LivePipeline],
    models: DemoModelConfig,
    output_dir: Path,
    display: DemoDisplayConfig = DEFAULT_REPLAY_DISPLAY,
    freshness_seconds: float = 0.75,
    frame_packets: Iterable[FramePacket] | None = None,
    camera_ids: Sequence[str] | None = None,
    video_fps: float | None = None,
) -> dict[str, object]:
    """Run deterministically with captured timestamps, even if inference is slow."""
    if not math.isfinite(freshness_seconds) or freshness_seconds <= 0:
        raise ValueError("freshness_seconds must be positive")
    if video_fps is not None and (not math.isfinite(video_fps) or video_fps <= 0):
        raise ValueError("video_fps must be finite and positive")
    camera_ids = tuple(camera_ids) if camera_ids is not None else recording_camera_ids(recording)
    if len(camera_ids) < 2 or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("select at least two distinct camera IDs")
    if set(camera_ids) != set(pipelines):
        raise ValueError("pipelines must match recorded camera IDs")
    registry = build_association_registry(models)
    if not isinstance(registry, HandoffIdentityRegistry):
        raise ValueError("recorded handoff replay requires association_mode=handoff or hybrid")
    validate_overlap_pairs(models.overlap_pairs, camera_ids)
    output = output_dir.expanduser().resolve()
    if output == recording.expanduser().resolve() or output.is_relative_to(recording.resolve()):
        raise ValueError("replay output must be outside the source recording")
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(
        json.dumps(
            {
                "association": {
                    "association_mode": registry.association_mode,
                    "max_cosine_distance": models.max_cosine_distance,
                    "min_stored_samples": models.min_stored_samples,
                    "gallery_ttl_seconds": models.gallery_ttl_seconds,
                    "exit_grace_seconds": models.exit_grace_seconds,
                    "min_travel_seconds": models.min_travel_seconds,
                    "match_margin": models.match_margin,
                    "allowed_transitions": models.allowed_transitions,
                    "overlap_pairs": models.overlap_pairs,
                    "overlap_max_cosine_distance": models.overlap_max_cosine_distance,
                    "overlap_confirmations": models.overlap_confirmations,
                    "reconciliation_enabled": models.reconciliation_enabled,
                    "reconciliation_confirmations": models.reconciliation_confirmations,
                    "reconciliation_min_seconds": models.reconciliation_min_seconds,
                },
                "checkpoint": str(models.checkpoint),
                "freshness_seconds": freshness_seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    latest = {}
    controller = ScreenshotController()
    frames = {}
    counts = dict.fromkeys(camera_ids, 0)
    handoffs = 0
    overlaps = 0
    merges = 0
    completed = False
    update_id = 0
    last_time = 0.0
    started = time.monotonic()
    video_writer = None
    last_progress = started
    if display.enabled:
        cv2.namedWindow(display.window_name, cv2.WINDOW_NORMAL)
    try:
        with (
            (output / "assignments.jsonl").open("w", encoding="utf-8") as assignments_file,
            (output / "handoff_events.jsonl").open("w", encoding="utf-8") as events_file,
            (output / "overlap_events.jsonl").open("w", encoding="utf-8") as overlap_file,
            (output / "merge_events.jsonl").open("w", encoding="utf-8") as merges_file,
            (output / "association_decisions.jsonl").open("w", encoding="utf-8") as decisions_file,
            (output / "processed_frames.jsonl").open("w", encoding="utf-8") as processed_file,
        ):
            packets = (
                frame_packets if frame_packets is not None else iter_recorded_frames(recording)
            )
            for packet in packets:
                last_time = packet.timestamp
                latest[packet.camera_id] = pipelines[packet.camera_id].process(packet)
                counts[packet.camera_id] += 1
                processed_file.write(
                    json.dumps(
                        {
                            "camera_id": packet.camera_id,
                            "source_frame_id": packet.frame_id,
                            "timestamp": packet.timestamp,
                        }
                    )
                    + "\n"
                )
                recent = {
                    camera: result
                    for camera, result in latest.items()
                    if packet.timestamp - result.stage3.packet.timestamp <= freshness_seconds
                }
                visible = tuple(
                    TrackKey(camera, track.local_id)
                    for camera, result in recent.items()
                    for track in result.stage3.tracks
                )
                result = registry.update(
                    tuple(item for view in recent.values() for item in view.appearances),
                    frame_id=update_id,
                    timestamp=packet.timestamp,
                    visible_keys=visible,
                )
                update_id += 1
                identity_rows = {}
                for camera, view in latest.items():
                    if camera not in recent:
                        frames[camera] = view.stage3.packet.frame
                for camera, view in recent.items():
                    frames[camera] = draw_global_boxes(
                        view.stage3.packet.frame,
                        view.stage3.tracks,
                        result.assignments,
                        camera,
                    )
                    identity_rows[camera] = tuple(
                        IdentityDisplay(
                            result.assignments.get(TrackKey(camera, track.local_id)),
                            track.local_id,
                            track.confidence,
                        )
                        for track in view.stage3.tracks
                    )
                    for track in view.stage3.tracks:
                        assignments_file.write(
                            json.dumps(
                                {
                                    "timestamp": packet.timestamp,
                                    "camera_id": camera,
                                    "source_frame_id": view.stage3.packet.frame_id,
                                    "local_id": track.local_id,
                                    "global_id": result.assignments.get(
                                        TrackKey(camera, track.local_id)
                                    ),
                                    "xyxy": track.xyxy,
                                }
                            )
                            + "\n"
                        )
                dashboard = None
                if (
                    result.handoff_events
                    or result.overlap_events
                    or result.merge_events
                    or display.enabled
                    or video_fps is not None
                ):
                    dashboard = render_dashboard(
                        camera_ids,
                        frames,
                        {
                            camera: "recorded | current" if camera in recent else "recorded | stale"
                            for camera in camera_ids
                        },
                        identity_rows,
                        display,
                        controller=controller,
                        session_message=(
                            f"{registry.association_mode} replay | "
                            f"source time {packet.timestamp:.1f}s"
                        ),
                    )
                for event in result.handoff_events:
                    handoffs += 1
                    events_file.write(json.dumps(event.to_dict()) + "\n")
                for event in result.overlap_events:
                    overlaps += 1
                    overlap_file.write(json.dumps(event.to_dict()) + "\n")
                for event in result.merge_events:
                    merges += 1
                    merges_file.write(json.dumps(event.to_dict()) + "\n")
                for decision in result.decisions:
                    decisions_file.write(json.dumps(decision.to_dict()) + "\n")
                if (
                    result.handoff_events or result.overlap_events or result.merge_events
                ) and dashboard is not None:
                    save_dashboard_screenshot(dashboard, output / "screenshots")
                if video_fps is not None and packet.camera_id == camera_ids[-1]:
                    if video_writer is None:
                        height, width = dashboard.shape[:2]
                        video_writer = cv2.VideoWriter(
                            str(output / "dashboard.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            video_fps,
                            (width, height),
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError("cannot create dashboard video")
                    video_writer.write(dashboard)
                if time.monotonic() - last_progress >= 10:
                    print(
                        f"Source time {packet.timestamp:.1f}s | frames {counts} | "
                        f"handoffs {handoffs}",
                        flush=True,
                    )
                    last_progress = time.monotonic()
                if display.enabled and dashboard is not None:
                    cv2.imshow(display.window_name, dashboard)
                    if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                        break
            else:
                completed = True
    finally:
        if video_writer is not None:
            video_writer.release()
        if display.enabled:
            cv2.destroyWindow(display.window_name)
    if not any(counts.values()):
        raise ValueError("recording contains no frames")
    summary = {
        "recording": str(recording.resolve()),
        "frames_by_camera": counts,
        "source_duration_seconds": last_time,
        "processing_seconds": time.monotonic() - started,
        "handoff_events": handoffs,
        "overlap_events": overlaps,
        "merge_events": merges,
        "identity_aliases": registry.identity_aliases,
        "completed": completed,
        "global_ids_issued": registry.total_ids_issued,
        "retained_identities": len(registry),
        "association_mode": registry.association_mode,
        "accuracy": "Not measured; requires independent human ground-truth annotations.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a recorded separate-camera handoff.")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=("solo", "multi_ip"))
    add_overlap_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/handoff/replay"))
    parser.add_argument("--show", action="store_true", help="Display at processing speed.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_camera_demo_config(args.config, mode=args.mode)
    camera_ids = recording_camera_ids(args.recording)
    models = with_overlap_overrides(
        config.models,
        args.overlap,
        camera_ids,
        reconciliation_enabled=False if args.no_reconcile else None,
    )
    if models.association_mode not in {"handoff", "hybrid"}:
        raise ValueError("replay config must select handoff or hybrid association")
    if args.dry_run:
        settings = config.public_dict()
        settings["models"].update(
            association_mode=models.association_mode,
            overlap_pairs=models.overlap_pairs,
            reconciliation_enabled=models.reconciliation_enabled,
        )
        print(json.dumps({"camera_ids": camera_ids, "config": settings}, indent=2))
        return 0
    detector = YoloPersonDetector(
        YoloConfig(
            model=models.detector_model,
            confidence=models.confidence,
            image_size=models.image_size,
            device=models.device,
        )
    )
    encoder = ReIDEncoder.from_checkpoint(
        models.checkpoint,
        ReIDEncoderConfig(device=models.device),
    )
    pipelines = build_reid_pipelines(
        camera_ids,
        detector,
        encoder,
        tracklet_config=(
            TrackletEmbeddingConfig(
                models.history_size,
                models.embedding_interval,
                models.max_idle_updates,
            )
        ),
    )
    summary = run_handoff_replay(
        args.recording,
        pipelines=pipelines,
        models=models,
        output_dir=args.output_dir,
        display=replace(config.display, enabled=args.show),
        freshness_seconds=config.runtime.association_window_seconds,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
