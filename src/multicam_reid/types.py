"""Shared data types used across input adapters and processing stages."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FramePacket:
    """A frame plus the metadata needed by the multi-camera pipeline."""

    camera_id: str
    frame_id: int
    timestamp: float
    frame: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must be a non-empty string")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if self.frame.ndim != 3 or self.frame.shape[2] != 3:
            raise ValueError("frame shape must be HxWx3")
        if self.frame.shape[0] == 0 or self.frame.shape[1] == 0:
            raise ValueError("frame must be non-empty")
