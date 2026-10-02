"""Full-scene WiseNET Set 2 video input; labels are accessed only for scoring."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import cv2

from multicam_reid.types import FramePacket


class WiseNETSequence:
    """Read the locally supplied subset without copying videos into JPEGs.

    Sampling changes inference workload, never source time or source frame IDs.
    Empty/background frames are retained so departure detection still works.
    """

    def __init__(
        self,
        root: Path,
        cameras: tuple[int, ...] = (3, 4),
        *,
        stride: int = 6,
        start: float = 0,
        end: float | None = None,
    ) -> None:
        if len(cameras) < 2 or len(set(cameras)) != len(cameras):
            raise ValueError("select at least two distinct cameras")
        if any(camera not in range(1, 6) for camera in cameras) or stride < 1:
            raise ValueError("camera numbers must be 1..5 and stride must be positive")
        if (
            not math.isfinite(start)
            or start < 0
            or (end is not None and (not math.isfinite(end) or end <= start))
        ):
            raise ValueError("invalid source time range")
        self.root = root.expanduser().resolve()
        self.cameras, self.stride, self.start, self.end = cameras, stride, start, end
        self.camera_ids = tuple(f"C{camera}" for camera in cameras)
        self.paths = {
            f"C{camera}": self.root / f"video_set2/video2_{camera}.avi" for camera in cameras
        }
        self.frame_counts = {}
        rates = []
        for camera, path in self.paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise ValueError(f"cannot open WiseNET video: {path}")
                rates.append(capture.get(cv2.CAP_PROP_FPS))
                self.frame_counts[camera] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()
        if (
            any(not math.isfinite(rate) or rate <= 0 for rate in rates)
            or (max(rates) - min(rates) > 0.001)
            or min(self.frame_counts.values()) <= 0
        ):
            raise ValueError("videos must have positive, equal frame rates and frame counts")
        self.fps = rates[0]

    def frames(self) -> Iterator[FramePacket]:
        """Yield ordered camera packets in dataset time, not inference wall time."""
        captures = {}
        try:
            for camera, path in self.paths.items():
                captures[camera] = cv2.VideoCapture(str(path))
            for frame_id in range(max(self.frame_counts.values())):
                timestamp = frame_id / self.fps
                if self.end is not None and timestamp >= self.end:
                    break
                for camera, capture in captures.items():
                    if frame_id >= self.frame_counts[camera]:
                        continue
                    if not capture.grab():
                        raise ValueError(f"truncated video: {camera}, frame {frame_id}")
                    if timestamp < self.start or frame_id % self.stride:
                        continue
                    success, image = capture.retrieve()
                    if not success:
                        raise ValueError(f"unreadable frame: {camera}, frame {frame_id}")
                    yield FramePacket(camera, frame_id, timestamp, image)
        finally:
            for capture in captures.values():
                capture.release()

    def ground_truth(self, camera: str) -> dict[int, list[tuple[int, tuple[float, ...]]]]:
        """Load clipped manual person boxes; never called by the inference loop."""
        if camera not in self.camera_ids:
            raise ValueError("unknown camera")
        path = self.root / f"annotations_set2/people_detection/video2_{camera[1:]}.json"
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        resolution = data["resolution"][0]
        width, height = resolution["width"], resolution["height"]
        frames = {}
        for frame in data["frames"]:
            boxes = []
            for detection in frame["detections"]:
                x, y, w, h = detection["xywh"]
                box = (max(0, x), max(0, y), min(width, x + w), min(height, y + h))
                if box[2] > box[0] and box[3] > box[1]:
                    boxes.append((int(detection["id"]), box))
            frames[int(frame["frameNumber"])] = boxes
        return frames
