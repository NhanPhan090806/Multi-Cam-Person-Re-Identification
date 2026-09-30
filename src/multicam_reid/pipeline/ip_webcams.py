"""Run independent Stage 3 pipelines over one to three IP webcam streams."""

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

from multicam_reid.detection.yolo import YoloConfig, YoloPersonDetector
from multicam_reid.inputs.ip_camera import (
    IpCameraConfig,
    IpCameraStatus,
    IpCameraWorker,
    load_ip_camera_configs,
)
from multicam_reid.pipeline.single_camera import ProcessedFrame, SingleCameraPipeline
from multicam_reid.tracking.bytetrack import ByteTrackLocalTracker
from multicam_reid.types import FramePacket

DEFAULT_CONFIG = Path("configs/ip_webcams.local.yaml")


class CameraPipeline(Protocol):
    def process(self, packet: FramePacket) -> ProcessedFrame: ...


class CameraWorker(Protocol):
    config: IpCameraConfig

    def start(self) -> None: ...

    def snapshot(self, after_frame_id: int = -1) -> FramePacket | None: ...

    def status(self) -> IpCameraStatus: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class IpWebcamRunConfig:
    """Display and stopping behavior for an IP webcam session."""

    display: bool = True
    max_frames_per_camera: int | None = None
    max_runtime_seconds: float | None = None
    startup_timeout_seconds: float = 15.0
    poll_interval_seconds: float = 0.005

    def __post_init__(self) -> None:
        if self.max_frames_per_camera is not None and self.max_frames_per_camera <= 0:
            raise ValueError("max_frames_per_camera must be positive")
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must not be negative")


@dataclass(frozen=True, slots=True)
class IpWebcamRunSummary:
    """Per-camera frame counts and final non-sensitive connection health."""

    elapsed_seconds: float
    total_processing_fps: float
    frames_by_camera: dict[str, int]
    final_status: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_camera_pipelines(
    cameras: Sequence[IpCameraConfig],
    detector: Any,
    *,
    tracker_factory: Callable[[str], Any] = ByteTrackLocalTracker,
) -> dict[str, SingleCameraPipeline]:
    """Share one detector while assigning independent tracker state per camera."""
    return {
        camera.camera_id: SingleCameraPipeline(
            detector=detector,
            tracker=tracker_factory(camera.camera_id),
        )
        for camera in cameras
    }


def _window_name(camera_id: str) -> str:
    return f"Multi-Cam ReID - {camera_id}"


def _initialize_windows(cameras: Sequence[IpCameraConfig]) -> None:
    for camera in cameras:
        name = _window_name(camera.camera_id)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, *camera.window_size)
        cv2.moveWindow(name, *camera.window_position)
        placeholder = np.zeros((camera.window_height, camera.window_width, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            f"Connecting to {camera.camera_id}...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(name, placeholder)


def run_ip_webcams(
    cameras: Sequence[IpCameraConfig],
    *,
    pipelines: Mapping[str, CameraPipeline],
    workers: Mapping[str, CameraWorker] | None = None,
    run_config: IpWebcamRunConfig | None = None,
) -> IpWebcamRunSummary:
    """Process the newest frame from every camera without cross-camera association."""
    if not cameras:
        raise ValueError("At least one IP webcam is required")
    settings = run_config or IpWebcamRunConfig()
    camera_ids = {camera.camera_id for camera in cameras}
    if set(pipelines) != camera_ids:
        raise ValueError("pipelines must contain exactly one entry per camera")
    active_workers: Mapping[str, CameraWorker]
    if workers is None:
        active_workers = {camera.camera_id: IpCameraWorker(camera) for camera in cameras}
    else:
        active_workers = workers
    if set(active_workers) != camera_ids:
        raise ValueError("workers must contain exactly one entry per camera")

    counts = {camera_id: 0 for camera_id in camera_ids}
    last_frame_ids = {camera_id: -1 for camera_id in camera_ids}
    started = time.monotonic()
    if settings.display:
        _initialize_windows(cameras)
    for worker in active_workers.values():
        worker.start()

    try:
        while True:
            received_frame = False
            for camera in cameras:
                camera_id = camera.camera_id
                if (
                    settings.max_frames_per_camera is not None
                    and counts[camera_id] >= settings.max_frames_per_camera
                ):
                    continue
                packet = active_workers[camera_id].snapshot(last_frame_ids[camera_id])
                if packet is None:
                    continue
                received_frame = True
                last_frame_ids[camera_id] = packet.frame_id
                result = pipelines[camera_id].process(packet)
                counts[camera_id] += 1
                if settings.display:
                    cv2.imshow(_window_name(camera_id), result.annotated_frame)

            elapsed = time.monotonic() - started
            if sum(counts.values()) == 0 and elapsed >= settings.startup_timeout_seconds:
                statuses = {key: worker.status() for key, worker in active_workers.items()}
                details = "; ".join(
                    f"{key}: {status.last_error or 'no frames'}" for key, status in statuses.items()
                )
                raise TimeoutError(f"No IP webcam produced a frame: {details}")
            if settings.max_runtime_seconds is not None and elapsed >= settings.max_runtime_seconds:
                break
            if settings.max_frames_per_camera is not None and all(
                count >= settings.max_frames_per_camera for count in counts.values()
            ):
                break
            if settings.display:
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break
            if not received_frame and settings.poll_interval_seconds:
                time.sleep(settings.poll_interval_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        for worker in active_workers.values():
            worker.stop()
        if settings.display:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    statuses = {key: worker.status() for key, worker in active_workers.items()}
    return IpWebcamRunSummary(
        elapsed_seconds=elapsed,
        total_processing_fps=sum(counts.values()) / elapsed if elapsed > 0 else 0.0,
        frames_by_camera=counts,
        final_status={key: asdict(status) for key, status in statuses.items()},
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the multi-IP-webcam command parser."""
    parser = argparse.ArgumentParser(
        description="Detect and locally track people independently on one to three IP webcams."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames-per-camera", type=int)
    parser.add_argument("--max-runtime-seconds", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load camera URLs, share YOLO, and run independent local pipelines."""
    args = build_parser().parse_args(argv)
    cameras = load_ip_camera_configs(args.config)
    detector = YoloPersonDetector(
        YoloConfig(
            model=args.model,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
        )
    )
    pipelines = build_camera_pipelines(cameras, detector)
    summary = run_ip_webcams(
        cameras,
        pipelines=pipelines,
        run_config=IpWebcamRunConfig(
            display=not args.no_display,
            max_frames_per_camera=args.max_frames_per_camera,
            max_runtime_seconds=args.max_runtime_seconds,
        ),
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
