"""Synchronized EPFL Laboratory videos behind the common FramePacket API."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2

from multicam_reid.data.epfl_lab import (
    GROUND_TRUTH_FILENAME,
    EpflLabLayoutError,
    EpflPersonPosition,
    find_epfl_lab_root,
    inspect_epfl_lab,
    load_epfl_ground_truth,
    validate_epfl_cameras,
    video_filename,
)
from multicam_reid.types import FramePacket


@dataclass(frozen=True, slots=True)
class EpflLabSynchronizedFrame:
    """Selected camera packets and identity positions at one source frame."""

    frame_key: str
    raw_frame_id: int
    packets: tuple[FramePacket, ...]
    positions: tuple[EpflPersonPosition, ...]

    def packet(self, camera_id: str) -> FramePacket:
        """Return one selected camera packet by ID."""
        try:
            return next(packet for packet in self.packets if packet.camera_id == camera_id)
        except StopIteration as error:
            raise KeyError(
                f"Camera {camera_id} is not present in frame {self.frame_key}"
            ) from error


class EpflLabSource:
    """Decode synchronized EPFL videos together without losing source time."""

    def __init__(
        self,
        search_root: Path,
        *,
        cameras: Sequence[str] = ("C0", "C1"),
        start_frame: int = 0,
        frame_stride: int = 1,
    ) -> None:
        self.cameras = validate_epfl_cameras(cameras)
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        self.start_frame = start_frame
        self.frame_stride = frame_stride
        self.dataset_root = find_epfl_lab_root(search_root)
        self.summary = inspect_epfl_lab(self.dataset_root, cameras=self.cameras)
        self.video_fps = self.summary.video_fps
        self.output_fps = self.video_fps / self.frame_stride
        if self.start_frame >= self.summary.synchronized_frames:
            raise ValueError(
                "start_frame must be less than the synchronized video frame count"
            )
        self._ground_truth = load_epfl_ground_truth(
            self.dataset_root / GROUND_TRUTH_FILENAME
        )

    def __len__(self) -> int:
        return (
            self.summary.synchronized_frames - self.start_frame + self.frame_stride - 1
        ) // self.frame_stride

    def __iter__(self) -> Iterator[EpflLabSynchronizedFrame]:
        captures = {
            camera: cv2.VideoCapture(str(self.dataset_root / video_filename(camera)))
            for camera in self.cameras
        }
        try:
            failed = [camera for camera, capture in captures.items() if not capture.isOpened()]
            if failed:
                raise EpflLabLayoutError(f"Could not open EPFL videos for cameras: {failed}")
            for capture in captures.values():
                capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            for raw_frame_id in range(self.start_frame, self.summary.synchronized_frames):
                decoded = {}
                for camera, capture in captures.items():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise EpflLabLayoutError(
                            f"Camera {camera} failed at synchronized frame {raw_frame_id}"
                        )
                    decoded[camera] = frame
                if (raw_frame_id - self.start_frame) % self.frame_stride:
                    continue
                timestamp = raw_frame_id / self.video_fps
                packets = tuple(
                    FramePacket(
                        camera_id=camera,
                        frame_id=raw_frame_id,
                        timestamp=timestamp,
                        frame=decoded[camera],
                    )
                    for camera in self.cameras
                )
                yield EpflLabSynchronizedFrame(
                    frame_key=f"{raw_frame_id:06d}",
                    raw_frame_id=raw_frame_id,
                    packets=packets,
                    positions=self._ground_truth.positions_at(raw_frame_id),
                )
        finally:
            for capture in captures.values():
                capture.release()
