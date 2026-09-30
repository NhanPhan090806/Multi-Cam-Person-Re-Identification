"""Ultralytics YOLO person-only detection behind a stable NumPy contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class DetectionBatch:
    """Person detections in absolute pixel coordinates."""

    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __post_init__(self) -> None:
        if self.xyxy.ndim != 2 or self.xyxy.shape[1:] != (4,):
            raise ValueError("xyxy shape must be Nx4")
        count = self.xyxy.shape[0]
        if self.conf.shape != (count,):
            raise ValueError("conf shape must be N")
        if self.cls.shape != (count,):
            raise ValueError("cls shape must be N")
        if not all(np.all(np.isfinite(values)) for values in (self.xyxy, self.conf, self.cls)):
            raise ValueError("detection values must be finite")

    def __len__(self) -> int:
        return self.xyxy.shape[0]

    def __getitem__(self, index: Any) -> DetectionBatch:
        """Return a filtered batch, preserving its two-dimensional box shape."""
        return DetectionBatch(
            xyxy=np.asarray(self.xyxy[index], dtype=np.float32).reshape(-1, 4),
            conf=np.asarray(self.conf[index], dtype=np.float32).reshape(-1),
            cls=np.asarray(self.cls[index], dtype=np.float32).reshape(-1),
        )

    @property
    def xywh(self) -> np.ndarray:
        """Return center-x, center-y, width, height boxes for ByteTrack."""
        result = np.empty_like(self.xyxy, dtype=np.float32)
        result[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        result[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        result[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        result[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return result

    @classmethod
    def empty(cls) -> DetectionBatch:
        """Create an empty person-detection batch."""
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            conf=np.empty((0,), dtype=np.float32),
            cls=np.empty((0,), dtype=np.float32),
        )


@dataclass(frozen=True, slots=True)
class YoloConfig:
    """Inference settings for the person detector."""

    model: Path = Path("yolo26n.pt")
    confidence: float = 0.1
    iou: float = 0.7
    image_size: int = 640
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1]")
        if not 0.0 < self.iou <= 1.0:
            raise ValueError("iou must be in (0, 1]")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class YoloPersonDetector:
    """Run a pretrained YOLO model for COCO class 0 (person) only."""

    def __init__(self, config: YoloConfig | None = None, *, model: Any | None = None) -> None:
        self.config = config or YoloConfig()
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(self.config.model))
        self._model = model

    def detect(self, frame: np.ndarray) -> DetectionBatch:
        """Return person detections for one uint8 BGR frame."""
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be a uint8 HxWx3 BGR image")
        device: str | int | None
        if self.config.device == "auto":
            device = None
        elif self.config.device == "cuda":
            device = 0
        else:
            device = "cpu"
        result = self._model.predict(
            frame,
            classes=[0],
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or _to_numpy(boxes.xyxy).shape[0] == 0:
            return DetectionBatch.empty()
        return DetectionBatch(
            xyxy=np.ascontiguousarray(_to_numpy(boxes.xyxy), dtype=np.float32),
            conf=np.ascontiguousarray(_to_numpy(boxes.conf), dtype=np.float32),
            cls=np.ascontiguousarray(_to_numpy(boxes.cls), dtype=np.float32),
        )
