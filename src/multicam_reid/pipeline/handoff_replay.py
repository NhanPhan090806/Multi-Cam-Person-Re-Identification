"""Replay raw full-scene recordings through the common handoff inference pipeline."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import cv2

from multicam_reid.association.handoff import HandoffIdentityRegistry
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
)
from multicam_reid.reid.encoder import ReIDEncoder, ReIDEncoderConfig
from multicam_reid.reid.tracklets import TrackKey, TrackletEmbeddingConfig
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
) -> dict[str, object]:
    """Run deterministically with captured timestamps, even if inference is slow."""
    if freshness_seconds <= 0:
        raise ValueError("freshness_seconds must be positive")
    camera_ids = recording_camera_ids(recording)
    if set(camera_ids) != set(pipelines):
        raise ValueError("pipelines must match recorded camera IDs")
    registry = build_association_registry(models)
    if not isinstance(registry, HandoffIdentityRegistry):
        raise ValueError("recorded handoff replay requires association_mode=handoff")
    output = output_dir.expanduser().resolve()
    if output == recording.expanduser().resolve() or output.is_relative_to(recording.resolve()):
        raise ValueError("replay output must be outside the source recording")
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(
        json.dumps(
            {
                "association": {
                    "max_cosine_distance": models.max_cosine_distance,
                    "min_stored_samples": models.min_stored_samples,
                    "gallery_ttl_seconds": models.gallery_ttl_seconds,
                    "exit_grace_seconds": models.exit_grace_seconds,
                    "min_travel_seconds": models.min_travel_seconds,
                    "match_margin": models.match_margin,
                    "allowed_transitions": models.allowed_transitions,
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
    update_id = 0
    last_time = 0.0
    started = time.monotonic()
    if display.enabled:
        cv2.namedWindow(display.window_name, cv2.WINDOW_NORMAL)
    try:
        with (
            (output / "assignments.jsonl").open("w", encoding="utf-8") as assignments_file,
            (output / "handoff_events.jsonl").open("w", encoding="utf-8") as events_file,
        ):
            for packet in iter_recorded_frames(recording):
                last_time = packet.timestamp
                latest[packet.camera_id] = pipelines[packet.camera_id].process(packet)
                counts[packet.camera_id] += 1
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
                if result.handoff_events or display.enabled:
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
                        session_message=f"handoff replay | source time {packet.timestamp:.1f}s",
                    )
                for event in result.handoff_events:
                    handoffs += 1
                    events_file.write(json.dumps(event.to_dict()) + "\n")
                if result.handoff_events and dashboard is not None:
                    save_dashboard_screenshot(dashboard, output / "screenshots")
                if display.enabled and dashboard is not None:
                    cv2.imshow(display.window_name, dashboard)
                    if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                        break
    finally:
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
        "global_ids_issued": registry.total_ids_issued,
        "retained_identities": len(registry),
        "association_mode": "handoff",
        "accuracy": "Not measured; requires independent human ground-truth annotations.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a recorded separate-camera handoff.")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=("solo", "multi_ip"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/handoff/replay"))
    parser.add_argument("--show", action="store_true", help="Display at processing speed.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_camera_demo_config(args.config, mode=args.mode)
    camera_ids = recording_camera_ids(args.recording)
    if config.models.association_mode != "handoff":
        raise ValueError("replay config must select handoff association")
    if args.dry_run:
        print(json.dumps({"camera_ids": camera_ids, "config": config.public_dict()}, indent=2))
        return 0
    models = config.models
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
