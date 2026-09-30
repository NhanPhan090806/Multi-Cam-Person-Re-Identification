"""OpenCV video-file input adapter for the common FramePacket contract."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2

from multicam_reid.types import FramePacket


class VideoFileSource:
    """Read one video as timestamped packets for one logical camera."""

    def __init__(
        self,
        path: Path,
        camera_id: str,
        *,
        capture: Any | None = None,
    ) -> None:
        source_path = path.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Video source does not exist: {source_path}")
        if not camera_id.strip():
            raise ValueError("camera_id must not be empty")
        self.path = source_path
        self.camera_id = camera_id
        self._capture = capture if capture is not None else cv2.VideoCapture(str(source_path))
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"OpenCV could not open video source: {source_path}")
        reported_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.fps = reported_fps if reported_fps > 0 else 25.0

    def __iter__(self) -> Iterator[FramePacket]:
        frame_id = 0
        while True:
            success, frame = self._capture.read()
            if not success:
                break
            position_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = position_ms / 1000.0 if position_ms > 0 else frame_id / self.fps
            yield FramePacket(
                camera_id=self.camera_id,
                frame_id=frame_id,
                timestamp=timestamp,
                frame=frame,
            )
            frame_id += 1

    def close(self) -> None:
        """Release the underlying OpenCV capture handle."""
        self._capture.release()

    def __enter__(self) -> VideoFileSource:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
