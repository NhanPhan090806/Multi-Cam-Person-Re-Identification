"""Replay synchronized WILDTRACK views through independent Stage 3 pipelines."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2

from multicam_reid.data.wildtrack import ANNOTATION_FPS, validate_wildtrack_cameras
from multicam_reid.detection.yolo import YoloConfig, YoloPersonDetector
from multicam_reid.inputs.wildtrack import WildtrackSource
from multicam_reid.pipeline.single_camera import ProcessedFrame, SingleCameraPipeline
from multicam_reid.tracking.bytetrack import ByteTrackLocalTracker
from multicam_reid.types import FramePacket

DEFAULT_OUTPUT_DIR = Path("outputs/stage5/wildtrack_c1_c2")


class CameraPipeline(Protocol):
    def process(self, packet: FramePacket) -> ProcessedFrame: ...


@dataclass(frozen=True, slots=True)
class WildtrackRunConfig:
    """Dataset, camera, visualization, and stopping settings."""

    dataset_root: Path = Path("data/wildtrack")
    output_dir: Path = DEFAULT_OUTPUT_DIR
    cameras: tuple[str, ...] = ("C1", "C2")
    fps: float = ANNOTATION_FPS
    max_frames: int | None = None
    display: bool = False
    save_video: bool = True
    codec: str = "mp4v"

    def __post_init__(self) -> None:
        validate_wildtrack_cameras(self.cameras)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if len(self.codec) != 4:
            raise ValueError("codec must contain exactly four characters")


@dataclass(frozen=True, slots=True)
class WildtrackCameraStats:
    """Per-camera Stage 3 counts for a synchronized replay."""

    frames_processed: int
    detections: int
    track_observations: int
    unique_local_ids: int
    valid_crops: int
    visible_ground_truth_boxes: int
    output_video: str | None


@dataclass(frozen=True, slots=True)
class WildtrackRunSummary:
    """Measured Stage 5 output and independent per-camera state evidence."""

    dataset_root: str
    cameras: tuple[str, ...]
    frames_processed: int
    camera_frames_processed: int
    unique_ground_truth_ids: int
    elapsed_seconds: float
    synchronized_fps: float
    camera_processing_fps: float
    camera_stats: dict[str, WildtrackCameraStats]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready nested summary."""
        return asdict(self)


def build_wildtrack_pipelines(
    cameras: Sequence[str],
    detector: Any,
    *,
    tracker_factory: Callable[[str], Any] = ByteTrackLocalTracker,
) -> dict[str, SingleCameraPipeline]:
    """Share YOLO while assigning one independent ByteTrack object per camera."""
    selected = validate_wildtrack_cameras(cameras)
    return {
        camera_id: SingleCameraPipeline(
            detector=detector,
            tracker=tracker_factory(camera_id),
        )
        for camera_id in selected
    }


def run_wildtrack(
    config: WildtrackRunConfig,
    *,
    pipelines: Mapping[str, CameraPipeline],
) -> WildtrackRunSummary:
    """Replay synchronized views without performing cross-camera association."""
    cameras = validate_wildtrack_cameras(config.cameras)
    if set(pipelines) != set(cameras):
        raise ValueError("pipelines must contain exactly one entry per selected camera")
    if len({id(pipeline) for pipeline in pipelines.values()}) != len(cameras):
        raise ValueError("Each WILDTRACK camera must own a separate pipeline object")

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, cv2.VideoWriter] = {}
    counts = {
        camera_id: {
            "frames": 0,
            "detections": 0,
            "tracks": 0,
            "crops": 0,
            "local_ids": set(),
            "visible_ground_truth_boxes": 0,
        }
        for camera_id in cameras
    }
    frames_processed = 0
    ground_truth_ids: set[int] = set()
    source = WildtrackSource(config.dataset_root, cameras=cameras, fps=config.fps)
    started = time.perf_counter()

    try:
        for synchronized in source:
            if config.max_frames is not None and frames_processed >= config.max_frames:
                break
            ground_truth_ids.update(
                person.person_id for person in synchronized.people if person.views
            )
            for camera_id in cameras:
                packet = synchronized.packet(camera_id)
                result = pipelines[camera_id].process(packet)
                camera_counts = counts[camera_id]
                camera_counts["frames"] += 1
                camera_counts["detections"] += len(result.detections)
                camera_counts["tracks"] += len(result.tracks)
                camera_counts["crops"] += len(result.crops)
                camera_counts["local_ids"].update(track.local_id for track in result.tracks)
                camera_counts["visible_ground_truth_boxes"] += sum(
                    person.box_for(camera_id) is not None for person in synchronized.people
                )

                if config.save_video:
                    writer = writers.get(camera_id)
                    if writer is None:
                        height, width = result.annotated_frame.shape[:2]
                        output_path = output_dir / f"{camera_id}_annotated.mp4"
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*config.codec),
                            config.fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open WILDTRACK output: {output_path}")
                        writers[camera_id] = writer
                    writer.write(result.annotated_frame)
                if config.display:
                    cv2.imshow(f"WILDTRACK Stage 5 - {camera_id}", result.annotated_frame)
            frames_processed += 1
            if config.display and cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for writer in writers.values():
            writer.release()
        if config.display:
            cv2.destroyAllWindows()

    if frames_processed == 0:
        raise ValueError("WILDTRACK source did not yield any frames")
    elapsed = time.perf_counter() - started
    camera_stats = {
        camera_id: WildtrackCameraStats(
            frames_processed=int(counts[camera_id]["frames"]),
            detections=int(counts[camera_id]["detections"]),
            track_observations=int(counts[camera_id]["tracks"]),
            unique_local_ids=len(counts[camera_id]["local_ids"]),
            valid_crops=int(counts[camera_id]["crops"]),
            visible_ground_truth_boxes=int(
                counts[camera_id]["visible_ground_truth_boxes"]
            ),
            output_video=(
                str(output_dir / f"{camera_id}_annotated.mp4") if config.save_video else None
            ),
        )
        for camera_id in cameras
    }
    camera_frames = frames_processed * len(cameras)
    summary = WildtrackRunSummary(
        dataset_root=str(source.dataset_root),
        cameras=cameras,
        frames_processed=frames_processed,
        camera_frames_processed=camera_frames,
        unique_ground_truth_ids=len(ground_truth_ids),
        elapsed_seconds=elapsed,
        synchronized_fps=frames_processed / elapsed,
        camera_processing_fps=camera_frames / elapsed,
        camera_stats=camera_stats,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 5 synchronized replay parser."""
    parser = argparse.ArgumentParser(
        description="Replay WILDTRACK views through independent YOLO + ByteTrack pipelines."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/wildtrack"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cameras", nargs="+", default=("C1", "C2"))
    parser.add_argument("--model", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Construct and run the production Stage 5 replay."""
    args = build_parser().parse_args(argv)
    cameras = validate_wildtrack_cameras(args.cameras)
    detector = YoloPersonDetector(
        YoloConfig(
            model=args.model,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
        )
    )
    pipelines = build_wildtrack_pipelines(cameras, detector)
    summary = run_wildtrack(
        WildtrackRunConfig(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            cameras=cameras,
            max_frames=args.max_frames,
            display=args.display,
            save_video=not args.no_video,
        ),
        pipelines=pipelines,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
