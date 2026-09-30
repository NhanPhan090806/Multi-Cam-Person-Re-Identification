"""Run the complete Stage 3-6 pipeline on synchronized EPFL Laboratory views."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2

from multicam_reid.association.global_registry import (
    AssociationConfig,
    GlobalIdentityRegistry,
)
from multicam_reid.data.epfl_lab import validate_epfl_cameras
from multicam_reid.detection.yolo import YoloConfig, YoloPersonDetector
from multicam_reid.inputs.epfl_lab import EpflLabSource
from multicam_reid.pipeline.epfl_lab_runner import _add_sync_label, _horizontal_composite
from multicam_reid.pipeline.reid_camera import ReIDCameraPipeline, ReIDProcessedFrame
from multicam_reid.pipeline.single_camera import SingleCameraPipeline
from multicam_reid.reid.encoder import DEFAULT_CHECKPOINT, ReIDEncoder, ReIDEncoderConfig
from multicam_reid.reid.tracklets import TrackletEmbeddingConfig, TrackletEmbeddingStore
from multicam_reid.tracking.bytetrack import ByteTrackLocalTracker
from multicam_reid.types import FramePacket
from multicam_reid.visualization.tracks import draw_global_tracks

DEFAULT_OUTPUT_DIR = Path("outputs/stage6/epfl_lab_global_c0_c1")


class ReIDPipeline(Protocol):
    def process(self, packet: FramePacket) -> ReIDProcessedFrame: ...


@dataclass(frozen=True, slots=True)
class GlobalEpflRunConfig:
    """Input, output, and fixed association settings for a Stage 6 replay."""

    dataset_root: Path = Path("data/epfl_lab")
    output_dir: Path = DEFAULT_OUTPUT_DIR
    cameras: tuple[str, ...] = ("C0", "C1")
    start_frame: int = 0
    frame_stride: int = 1
    max_frames: int | None = None
    display: bool = False
    save_video: bool = True
    save_composite: bool = True
    codec: str = "mp4v"
    association: AssociationConfig = AssociationConfig()

    def __post_init__(self) -> None:
        validate_epfl_cameras(self.cameras)
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if len(self.codec) != 4:
            raise ValueError("codec must contain exactly four characters")


@dataclass(frozen=True, slots=True)
class GlobalEpflCameraStats:
    """Per-camera observations and identity counts."""

    frames_processed: int
    detections: int
    track_observations: int
    valid_crops: int
    appearance_observations: int
    unique_local_ids: int
    unique_global_ids: int
    output_video: str | None


@dataclass(frozen=True, slots=True)
class GlobalEpflRunSummary:
    """Reproducible Stage 6 runtime and association artifact summary."""

    dataset_root: str
    cameras: tuple[str, ...]
    start_frame: int
    frame_stride: int
    frames_processed: int
    camera_frames_processed: int
    source_fps: float
    output_fps: float
    elapsed_seconds: float
    synchronized_fps: float
    camera_processing_fps: float
    global_identities_final: int
    global_ids_issued: int
    merge_events: int
    assignment_schema_version: int
    threshold: float
    min_stored_samples: int
    max_idle_frames: int
    camera_stats: dict[str, GlobalEpflCameraStats]
    assignments_log: str
    merge_events_log: str
    composite_video: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_global_epfl_pipelines(
    cameras: Sequence[str],
    detector: Any,
    encoder: Any,
    *,
    tracker_factory: Callable[[str], Any] = ByteTrackLocalTracker,
    tracklet_config: TrackletEmbeddingConfig | None = None,
) -> dict[str, ReIDCameraPipeline]:
    """Share heavy models but isolate each camera's tracker and tracklet memory."""
    selected = validate_epfl_cameras(cameras)
    return {
        camera: ReIDCameraPipeline(
            stage3=SingleCameraPipeline(
                detector=detector,
                tracker=tracker_factory(camera),
            ),
            tracklets=TrackletEmbeddingStore(encoder, tracklet_config),
        )
        for camera in selected
    }


def _open_writer(path: Path, frame: Any, fps: float, codec: str) -> cv2.VideoWriter:
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open Stage 6 output: {path}")
    return writer


def run_global_epfl(
    config: GlobalEpflRunConfig,
    *,
    pipelines: Mapping[str, ReIDPipeline],
    registry: GlobalIdentityRegistry | None = None,
) -> GlobalEpflRunSummary:
    """Run detection through global association and persist auditable evidence."""
    cameras = validate_epfl_cameras(config.cameras)
    if set(pipelines) != set(cameras):
        raise ValueError("pipelines must contain exactly one entry per selected camera")
    if len({id(pipeline) for pipeline in pipelines.values()}) != len(cameras):
        raise ValueError("Each EPFL camera must own a separate pipeline object")

    source = EpflLabSource(
        config.dataset_root,
        cameras=cameras,
        start_frame=config.start_frame,
        frame_stride=config.frame_stride,
    )
    association = registry or GlobalIdentityRegistry(config.association)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "assignments.jsonl"
    events_path = output_dir / "merge_events.jsonl"
    composite_path = output_dir / "synchronized_global_ids.mp4"
    writers: dict[str, cv2.VideoWriter] = {}
    composite_writer: cv2.VideoWriter | None = None
    counts = {
        camera: {
            "frames": 0,
            "detections": 0,
            "tracks": 0,
            "crops": 0,
            "appearances": 0,
            "local_ids": set(),
            "global_ids": set(),
        }
        for camera in cameras
    }
    frames_processed = 0
    merge_count = 0
    started = time.perf_counter()

    try:
        with assignments_path.open("w", encoding="utf-8") as assignments_file, events_path.open(
            "w", encoding="utf-8"
        ) as events_file:
            for synchronized in source:
                if config.max_frames is not None and frames_processed >= config.max_frames:
                    break
                results = {
                    camera: pipelines[camera].process(synchronized.packet(camera))
                    for camera in cameras
                }
                appearances = tuple(
                    appearance
                    for camera in cameras
                    for appearance in results[camera].appearances
                )
                timestamp = synchronized.packets[0].timestamp
                association_result = association.update(
                    appearances,
                    frame_id=synchronized.raw_frame_id,
                    timestamp=timestamp,
                )
                for event in association_result.merge_events:
                    events_file.write(json.dumps(event.to_dict()) + "\n")
                    merge_count += 1

                display_frames = []
                for camera in cameras:
                    result = results[camera]
                    stage3 = result.stage3
                    camera_counts = counts[camera]
                    camera_counts["frames"] += 1
                    camera_counts["detections"] += len(stage3.detections)
                    camera_counts["tracks"] += len(stage3.tracks)
                    camera_counts["crops"] += len(stage3.crops)
                    camera_counts["appearances"] += len(result.appearances)
                    camera_counts["local_ids"].update(
                        track.local_id for track in stage3.tracks
                    )
                    track_by_local_id = {
                        track.local_id: track for track in stage3.tracks
                    }
                    for appearance in result.appearances:
                        global_id = association_result.assignments[appearance.key]
                        track = track_by_local_id[appearance.key.local_id]
                        camera_counts["global_ids"].add(global_id)
                        assignments_file.write(
                            json.dumps(
                                {
                                    "frame_id": synchronized.raw_frame_id,
                                    "timestamp": timestamp,
                                    "camera_id": appearance.key.camera_id,
                                    "local_id": appearance.key.local_id,
                                    "global_id": global_id,
                                    "stored_samples": appearance.stored_samples,
                                    "total_samples": appearance.total_samples,
                                    "xyxy": list(track.xyxy),
                                    "confidence": track.confidence,
                                }
                            )
                            + "\n"
                        )

                    annotated = draw_global_tracks(
                        stage3.packet.frame,
                        stage3.tracks,
                        association_result.assignments,
                        camera,
                    )
                    annotated = _add_sync_label(
                        annotated,
                        camera,
                        synchronized.raw_frame_id,
                        timestamp,
                    )
                    display_frames.append(annotated)
                    if config.save_video:
                        writer = writers.get(camera)
                        if writer is None:
                            output_path = output_dir / f"{camera}_global_ids.mp4"
                            writer = _open_writer(
                                output_path, annotated, source.output_fps, config.codec
                            )
                            writers[camera] = writer
                        writer.write(annotated)

                composite = _horizontal_composite(display_frames)
                if config.save_video and config.save_composite:
                    if composite_writer is None:
                        composite_writer = _open_writer(
                            composite_path, composite, source.output_fps, config.codec
                        )
                    composite_writer.write(composite)
                if config.display:
                    cv2.imshow("Stage 6 - EPFL global identities", composite)
                frames_processed += 1
                if config.display and cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        for writer in writers.values():
            writer.release()
        if composite_writer is not None:
            composite_writer.release()
        if config.display:
            cv2.destroyAllWindows()

    if frames_processed == 0:
        raise ValueError("EPFL Laboratory source did not yield any frames")
    elapsed = time.perf_counter() - started
    camera_stats = {
        camera: GlobalEpflCameraStats(
            frames_processed=int(counts[camera]["frames"]),
            detections=int(counts[camera]["detections"]),
            track_observations=int(counts[camera]["tracks"]),
            valid_crops=int(counts[camera]["crops"]),
            appearance_observations=int(counts[camera]["appearances"]),
            unique_local_ids=len(counts[camera]["local_ids"]),
            unique_global_ids=len(counts[camera]["global_ids"]),
            output_video=(
                str(output_dir / f"{camera}_global_ids.mp4")
                if config.save_video
                else None
            ),
        )
        for camera in cameras
    }
    camera_frames = frames_processed * len(cameras)
    summary = GlobalEpflRunSummary(
        dataset_root=str(source.dataset_root),
        cameras=cameras,
        start_frame=config.start_frame,
        frame_stride=config.frame_stride,
        frames_processed=frames_processed,
        camera_frames_processed=camera_frames,
        source_fps=source.video_fps,
        output_fps=source.output_fps,
        elapsed_seconds=elapsed,
        synchronized_fps=frames_processed / elapsed,
        camera_processing_fps=camera_frames / elapsed,
        global_identities_final=len(association),
        global_ids_issued=association.total_ids_issued,
        merge_events=merge_count,
        assignment_schema_version=2,
        threshold=config.association.max_cosine_distance,
        min_stored_samples=config.association.min_stored_samples,
        max_idle_frames=config.association.max_idle_frames,
        camera_stats=camera_stats,
        assignments_log=str(assignments_path),
        merge_events_log=str(events_path),
        composite_video=(
            str(composite_path)
            if config.save_video and config.save_composite
            else None
        ),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the end-to-end Stage 6 EPFL command-line interface."""
    parser = argparse.ArgumentParser(
        description="Assign shared global IDs across synchronized EPFL Laboratory views."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/epfl_lab"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cameras", nargs="+", default=("C0", "C1"))
    parser.add_argument("--detector-model", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--embedding-interval", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--max-idle-frames", type=int, default=250)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-composite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load the real models and run the complete Stage 6 replay."""
    args = build_parser().parse_args(argv)
    cameras = validate_epfl_cameras(args.cameras)
    detector = YoloPersonDetector(
        YoloConfig(
            model=args.detector_model,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
        )
    )
    encoder = ReIDEncoder.from_checkpoint(
        args.checkpoint,
        ReIDEncoderConfig(device=args.device),
    )
    pipelines = build_global_epfl_pipelines(
        cameras,
        detector,
        encoder,
        tracklet_config=TrackletEmbeddingConfig(
            history_size=args.history_size,
            embedding_interval_frames=args.embedding_interval,
            stale_after_frames=args.max_idle_frames,
        ),
    )
    summary = run_global_epfl(
        GlobalEpflRunConfig(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            cameras=cameras,
            start_frame=args.start_frame,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            display=args.display,
            save_video=not args.no_video,
            save_composite=not args.no_composite,
            association=AssociationConfig(
                max_cosine_distance=args.threshold,
                min_stored_samples=args.min_samples,
                max_idle_frames=args.max_idle_frames,
            ),
        ),
        pipelines=pipelines,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
