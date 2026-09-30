"""Run and measure the Stage 3 single-camera video pipeline."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2

from multicam_reid.detection.yolo import YoloConfig, YoloPersonDetector
from multicam_reid.inputs.video import VideoFileSource
from multicam_reid.pipeline.single_camera import (
    CropConfig,
    ProcessedFrame,
    SingleCameraPipeline,
)
from multicam_reid.tracking.bytetrack import ByteTrackConfig, ByteTrackLocalTracker
from multicam_reid.types import FramePacket

DEFAULT_OUTPUT = Path("outputs/stage3/single_camera_annotated.mp4")


class FramePipeline(Protocol):
    def process(self, packet: FramePacket) -> ProcessedFrame: ...


@dataclass(frozen=True, slots=True)
class VideoRunConfig:
    """Input, output, and bounded-run settings for a single camera."""

    source: Path
    output: Path = DEFAULT_OUTPUT
    camera_id: str = "C1"
    codec: str = "mp4v"
    display: bool = False
    max_frames: int | None = None
    crop_dir: Path | None = None
    max_saved_crops: int = 50

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if len(self.codec) != 4:
            raise ValueError("codec must contain exactly four characters")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if self.max_saved_crops < 0:
            raise ValueError("max_saved_crops must not be negative")


@dataclass(frozen=True, slots=True)
class VideoRunSummary:
    """Measured Stage 3 output for one video run."""

    source: str
    output: str
    camera_id: str
    input_fps: float
    frames_processed: int
    total_detections: int
    total_track_observations: int
    unique_local_ids: int
    total_valid_crops: int
    saved_crops: int
    elapsed_seconds: float
    processing_fps: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready summary."""
        return asdict(self)


def _write_crop(path: Path, crop: ProcessedFrame, crop_index: int) -> None:
    person_crop = crop.crops[crop_index]
    filename = (
        f"{crop.packet.camera_id}_f{crop.packet.frame_id:06d}_"
        f"l{person_crop.track.local_id:04d}.jpg"
    )
    encoded, buffer = cv2.imencode(".jpg", person_crop.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not encoded:
        raise RuntimeError(f"Could not encode person crop: {filename}")
    (path / filename).write_bytes(buffer.tobytes())


def run_video(config: VideoRunConfig, pipeline: FramePipeline) -> VideoRunSummary:
    """Process a video, write annotated frames, optional crops, and JSON metrics."""
    output = config.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    crop_dir = config.crop_dir.expanduser().resolve() if config.crop_dir else None
    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    frames_processed = 0
    total_detections = 0
    total_track_observations = 0
    total_valid_crops = 0
    saved_crops = 0
    local_ids: set[int] = set()
    started = time.perf_counter()

    with VideoFileSource(config.source, config.camera_id) as source:
        try:
            for packet in source:
                if config.max_frames is not None and frames_processed >= config.max_frames:
                    break
                result = pipeline.process(packet)
                if writer is None:
                    height, width = result.annotated_frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output),
                        cv2.VideoWriter_fourcc(*config.codec),
                        source.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"OpenCV could not open video output: {output}")
                writer.write(result.annotated_frame)
                frames_processed += 1
                total_detections += len(result.detections)
                total_track_observations += len(result.tracks)
                total_valid_crops += len(result.crops)
                local_ids.update(track.local_id for track in result.tracks)

                if crop_dir is not None and saved_crops < config.max_saved_crops:
                    available = min(len(result.crops), config.max_saved_crops - saved_crops)
                    for crop_index in range(available):
                        _write_crop(crop_dir, result, crop_index)
                    saved_crops += available

                if config.display:
                    cv2.imshow(f"Multi-Cam ReID - {config.camera_id}", result.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        finally:
            if writer is not None:
                writer.release()
            if config.display:
                cv2.destroyAllWindows()

        input_fps = source.fps

    if frames_processed == 0:
        raise ValueError("Video source did not yield any frames")
    elapsed = time.perf_counter() - started
    summary = VideoRunSummary(
        source=str(config.source.expanduser().resolve()),
        output=str(output),
        camera_id=config.camera_id,
        input_fps=input_fps,
        frames_processed=frames_processed,
        total_detections=total_detections,
        total_track_observations=total_track_observations,
        unique_local_ids=len(local_ids),
        total_valid_crops=total_valid_crops,
        saved_crops=saved_crops,
        elapsed_seconds=elapsed,
        processing_fps=frames_processed / elapsed,
    )
    output.with_suffix(".json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 3 single-camera command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run YOLO26n and a camera-owned ByteTrack instance on one video."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera-id", default="C1")
    parser.add_argument("--model", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.1,
        help="YOLO floor; keep <= ByteTrack's 0.10 low threshold.",
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--crop-dir", type=Path)
    parser.add_argument("--max-saved-crops", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Construct and run the Stage 3 pipeline."""
    args = build_parser().parse_args(argv)
    detector = YoloPersonDetector(
        YoloConfig(
            model=args.model,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
        )
    )
    tracker = ByteTrackLocalTracker(args.camera_id, ByteTrackConfig())
    pipeline = SingleCameraPipeline(
        detector=detector,
        tracker=tracker,
        crop_config=CropConfig(),
    )
    summary = run_video(
        VideoRunConfig(
            source=args.source,
            output=args.output,
            camera_id=args.camera_id,
            display=args.display,
            max_frames=args.max_frames,
            crop_dir=args.crop_dir,
            max_saved_crops=args.max_saved_crops,
        ),
        pipeline,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
