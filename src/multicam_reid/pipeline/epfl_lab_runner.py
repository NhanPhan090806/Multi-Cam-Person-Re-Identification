"""Replay sparse synchronized EPFL views through independent Stage 3 pipelines."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from multicam_reid.data.epfl_lab import validate_epfl_cameras
from multicam_reid.detection.yolo import YoloConfig, YoloPersonDetector
from multicam_reid.inputs.epfl_lab import EpflLabSource
from multicam_reid.pipeline.single_camera import ProcessedFrame, SingleCameraPipeline
from multicam_reid.tracking.bytetrack import ByteTrackLocalTracker
from multicam_reid.types import FramePacket

DEFAULT_OUTPUT_DIR = Path("outputs/stage5/epfl_lab_c0_c1")


class CameraPipeline(Protocol):
    def process(self, packet: FramePacket) -> ProcessedFrame: ...


@dataclass(frozen=True, slots=True)
class EpflLabRunConfig:
    """Dataset, camera, output, and stopping settings for sparse replay."""

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
class EpflLabCameraStats:
    """Per-camera detection and local-tracking counts."""

    frames_processed: int
    detections: int
    track_observations: int
    unique_local_ids: int
    valid_crops: int
    output_video: str | None


@dataclass(frozen=True, slots=True)
class EpflLabRunSummary:
    """Measured Stage 5 replay output and synchronization evidence."""

    dataset_root: str
    cameras: tuple[str, ...]
    start_frame: int
    frame_stride: int
    source_fps: float
    output_fps: float
    frames_processed: int
    camera_frames_processed: int
    annotated_frames_seen: int
    unique_ground_truth_ids_seen: int
    elapsed_seconds: float
    synchronized_fps: float
    camera_processing_fps: float
    camera_stats: dict[str, EpflLabCameraStats]
    composite_video: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready nested summary."""
        return asdict(self)


def build_epfl_lab_pipelines(
    cameras: Sequence[str],
    detector: Any,
    *,
    tracker_factory: Callable[[str], Any] = ByteTrackLocalTracker,
) -> dict[str, SingleCameraPipeline]:
    """Share YOLO while assigning independent ByteTrack state per view."""
    selected = validate_epfl_cameras(cameras)
    return {
        camera: SingleCameraPipeline(
            detector=detector,
            tracker=tracker_factory(camera),
        )
        for camera in selected
    }


def _add_sync_label(
    frame: np.ndarray,
    camera_id: str,
    frame_id: int,
    timestamp: float,
) -> np.ndarray:
    annotated = frame.copy()
    text = f"{camera_id} | source frame {frame_id} | {timestamp:.2f}s"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        text,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def _horizontal_composite(frames: Sequence[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("At least one frame is required for a composite")
    target_height = min(frame.shape[0] for frame in frames)
    resized = [
        frame
        if frame.shape[0] == target_height
        else cv2.resize(
            frame,
            (round(frame.shape[1] * target_height / frame.shape[0]), target_height),
        )
        for frame in frames
    ]
    return np.hstack(resized)


def run_epfl_lab(
    config: EpflLabRunConfig,
    *,
    pipelines: Mapping[str, CameraPipeline],
) -> EpflLabRunSummary:
    """Replay synchronized sparse views without cross-camera association."""
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
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, cv2.VideoWriter] = {}
    composite_writer: cv2.VideoWriter | None = None
    composite_path = output_dir / "synchronized_annotated.mp4"
    counts = {
        camera: {
            "frames": 0,
            "detections": 0,
            "tracks": 0,
            "crops": 0,
            "local_ids": set(),
        }
        for camera in cameras
    }
    frames_processed = 0
    annotated_frames_seen = 0
    ground_truth_ids: set[int] = set()
    started = time.perf_counter()

    try:
        for synchronized in source:
            if config.max_frames is not None and frames_processed >= config.max_frames:
                break
            active_positions = tuple(
                position
                for position in synchronized.positions
                if position.state == "active"
            )
            if synchronized.positions:
                annotated_frames_seen += 1
            ground_truth_ids.update(position.person_id for position in active_positions)
            display_frames: list[np.ndarray] = []

            for camera in cameras:
                packet = synchronized.packet(camera)
                result = pipelines[camera].process(packet)
                camera_counts = counts[camera]
                camera_counts["frames"] += 1
                camera_counts["detections"] += len(result.detections)
                camera_counts["tracks"] += len(result.tracks)
                camera_counts["crops"] += len(result.crops)
                camera_counts["local_ids"].update(track.local_id for track in result.tracks)
                annotated = _add_sync_label(
                    result.annotated_frame,
                    camera,
                    synchronized.raw_frame_id,
                    packet.timestamp,
                )
                display_frames.append(annotated)

                if config.save_video:
                    writer = writers.get(camera)
                    if writer is None:
                        height, width = annotated.shape[:2]
                        output_path = output_dir / f"{camera}_annotated.mp4"
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*config.codec),
                            source.output_fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open EPFL output: {output_path}")
                        writers[camera] = writer
                    writer.write(annotated)

            composite = _horizontal_composite(display_frames)
            if config.save_video and config.save_composite:
                if composite_writer is None:
                    height, width = composite.shape[:2]
                    composite_writer = cv2.VideoWriter(
                        str(composite_path),
                        cv2.VideoWriter_fourcc(*config.codec),
                        source.output_fps,
                        (width, height),
                    )
                    if not composite_writer.isOpened():
                        raise RuntimeError(f"Could not open EPFL composite: {composite_path}")
                composite_writer.write(composite)
            if config.display:
                cv2.imshow("EPFL Laboratory Stage 5 - synchronized views", composite)
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
        camera: EpflLabCameraStats(
            frames_processed=int(counts[camera]["frames"]),
            detections=int(counts[camera]["detections"]),
            track_observations=int(counts[camera]["tracks"]),
            unique_local_ids=len(counts[camera]["local_ids"]),
            valid_crops=int(counts[camera]["crops"]),
            output_video=(
                str(output_dir / f"{camera}_annotated.mp4") if config.save_video else None
            ),
        )
        for camera in cameras
    }
    camera_frames = frames_processed * len(cameras)
    summary = EpflLabRunSummary(
        dataset_root=str(source.dataset_root),
        cameras=cameras,
        start_frame=config.start_frame,
        frame_stride=config.frame_stride,
        source_fps=source.video_fps,
        output_fps=source.output_fps,
        frames_processed=frames_processed,
        camera_frames_processed=camera_frames,
        annotated_frames_seen=annotated_frames_seen,
        unique_ground_truth_ids_seen=len(ground_truth_ids),
        elapsed_seconds=elapsed,
        synchronized_fps=frames_processed / elapsed,
        camera_processing_fps=camera_frames / elapsed,
        camera_stats=camera_stats,
        composite_video=(
            str(composite_path)
            if config.save_video and config.save_composite
            else None
        ),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the sparse synchronized Stage 5 replay parser."""
    parser = argparse.ArgumentParser(
        description="Replay sparse EPFL Laboratory views through independent YOLO + ByteTrack."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/epfl_lab"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cameras", nargs="+", default=("C0", "C1"))
    parser.add_argument("--model", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-composite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Construct and run the sparse production Stage 5 replay."""
    args = build_parser().parse_args(argv)
    cameras = validate_epfl_cameras(args.cameras)
    detector = YoloPersonDetector(
        YoloConfig(
            model=args.model,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
        )
    )
    pipelines = build_epfl_lab_pipelines(cameras, detector)
    summary = run_epfl_lab(
        EpflLabRunConfig(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            cameras=cameras,
            start_frame=args.start_frame,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            display=args.display,
            save_video=not args.no_video,
            save_composite=not args.no_composite,
        ),
        pipelines=pipelines,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
