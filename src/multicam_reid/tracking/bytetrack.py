"""Camera-owned ByteTrack state behind a stable local-track contract."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from multicam_reid.detection.yolo import DetectionBatch


@dataclass(frozen=True, slots=True)
class LocalTrack:
    """One active track in one camera's local identity namespace."""

    camera_id: str
    local_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    detection_index: int

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if self.local_id <= 0:
            raise ValueError("local_id must be positive")
        if len(self.xyxy) != 4 or not all(np.isfinite(value) for value in self.xyxy):
            raise ValueError("xyxy must contain four finite coordinates")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ByteTrackConfig:
    """Baseline Ultralytics ByteTrack association settings."""

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True

    def __post_init__(self) -> None:
        thresholds = (
            self.track_low_thresh,
            self.track_high_thresh,
            self.new_track_thresh,
            self.match_thresh,
        )
        if not all(0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("tracker thresholds must be in [0, 1]")
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("tracker threshold order requires low <= high")
        if self.track_buffer <= 0:
            raise ValueError("track_buffer must be positive")

    def as_namespace(self) -> SimpleNamespace:
        """Return the attribute interface expected by Ultralytics ByteTrack."""
        return SimpleNamespace(**{
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "fuse_score": self.fuse_score,
        })


class ByteTrackLocalTracker:
    """Own one independent ByteTrack state machine for one camera."""

    def __init__(
        self,
        camera_id: str,
        config: ByteTrackConfig | None = None,
        *,
        tracker: Any | None = None,
    ) -> None:
        if not camera_id.strip():
            raise ValueError("camera_id must not be empty")
        self.camera_id = camera_id
        self.config = config or ByteTrackConfig()
        if tracker is None:
            from ultralytics.trackers.byte_tracker import BYTETracker

            tracker = BYTETracker(self.config.as_namespace())
        self._tracker = tracker

    def update(
        self,
        detections: DetectionBatch,
        frame: np.ndarray,
    ) -> tuple[LocalTrack, ...]:
        """Advance this camera's tracker and return currently active local tracks."""
        raw_tracks = np.asarray(self._tracker.update(detections, img=frame), dtype=np.float32)
        if raw_tracks.size == 0:
            return ()
        if raw_tracks.ndim != 2 or raw_tracks.shape[1] < 8:
            raise RuntimeError("ByteTrack returned an unexpected track array shape")
        return tuple(
            LocalTrack(
                camera_id=self.camera_id,
                local_id=int(row[4]),
                xyxy=tuple(float(value) for value in row[:4]),
                confidence=float(row[5]),
                class_id=int(row[6]),
                detection_index=int(row[7]),
            )
            for row in raw_tracks
        )

    def reset(self) -> None:
        """Clear all state owned by this camera tracker."""
        self._tracker.reset()
