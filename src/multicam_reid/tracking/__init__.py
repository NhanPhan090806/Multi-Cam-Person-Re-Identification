"""Per-camera local tracking interfaces and implementations."""

from multicam_reid.tracking.bytetrack import (
    ByteTrackConfig,
    ByteTrackLocalTracker,
    LocalTrack,
)

__all__ = ["ByteTrackConfig", "ByteTrackLocalTracker", "LocalTrack"]
