"""Synchronized WILDTRACK frames behind the common FramePacket interface."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from multicam_reid.data.wildtrack import (
    ANNOTATION_DIRECTORY,
    ANNOTATION_FPS,
    IMAGE_DIRECTORY,
    WildtrackLayoutError,
    WildtrackPersonAnnotation,
    find_wildtrack_root,
    load_wildtrack_annotations,
    validate_wildtrack_cameras,
)
from multicam_reid.types import FramePacket


@dataclass(frozen=True, slots=True)
class WildtrackSynchronizedFrame:
    """Selected camera packets and persistent-person annotations at one timestamp."""

    frame_key: str
    packets: tuple[FramePacket, ...]
    people: tuple[WildtrackPersonAnnotation, ...]

    def packet(self, camera_id: str) -> FramePacket:
        """Return one selected camera packet by ID."""
        try:
            return next(packet for packet in self.packets if packet.camera_id == camera_id)
        except StopIteration as error:
            raise KeyError(
                f"Camera {camera_id} is not present in frame {self.frame_key}"
            ) from error


def _read_bgr(path: Path) -> np.ndarray:
    try:
        encoded = path.read_bytes()
    except FileNotFoundError as error:
        raise WildtrackLayoutError(f"WILDTRACK frame does not exist: {path}") from error
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise WildtrackLayoutError(f"Could not decode WILDTRACK frame: {path}")
    return frame


class WildtrackSource:
    """Replay annotation-aligned frames from multiple cameras synchronously."""

    def __init__(
        self,
        search_root: Path,
        *,
        cameras: Sequence[str] = ("C1", "C2"),
        fps: float = ANNOTATION_FPS,
    ) -> None:
        self.cameras = validate_wildtrack_cameras(cameras)
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.dataset_root = find_wildtrack_root(search_root)
        self._annotation_paths = tuple(
            sorted((self.dataset_root / ANNOTATION_DIRECTORY).glob("*.json"))
        )
        if not self._annotation_paths:
            raise WildtrackLayoutError("No WILDTRACK JSON annotations were found")
        for annotation_path in self._annotation_paths:
            for camera_id in self.cameras:
                image_path = (
                    self.dataset_root
                    / IMAGE_DIRECTORY
                    / camera_id
                    / f"{annotation_path.stem}.png"
                )
                if not image_path.is_file():
                    raise WildtrackLayoutError(
                        f"Camera {camera_id} is missing synchronized frame {annotation_path.stem}"
                    )

    def __len__(self) -> int:
        return len(self._annotation_paths)

    def __iter__(self) -> Iterator[WildtrackSynchronizedFrame]:
        for frame_id, annotation_path in enumerate(self._annotation_paths):
            timestamp = frame_id / self.fps
            packets = tuple(
                FramePacket(
                    camera_id=camera_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame=_read_bgr(
                        self.dataset_root
                        / IMAGE_DIRECTORY
                        / camera_id
                        / f"{annotation_path.stem}.png"
                    ),
                )
                for camera_id in self.cameras
            )
            yield WildtrackSynchronizedFrame(
                frame_key=annotation_path.stem,
                packets=packets,
                people=load_wildtrack_annotations(
                    annotation_path,
                    cameras=self.cameras,
                ),
            )
