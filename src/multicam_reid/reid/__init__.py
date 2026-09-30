"""Person Re-ID checkpoint loading and embedding extraction."""

from multicam_reid.reid.encoder import ReIDEncoder, ReIDEncoderConfig
from multicam_reid.reid.tracklets import (
    TrackKey,
    TrackletAppearance,
    TrackletEmbeddingConfig,
    TrackletEmbeddingStore,
    TrackletSeparationDiagnostic,
    evaluate_tracklet_separation,
)

__all__ = [
    "ReIDEncoder",
    "ReIDEncoderConfig",
    "TrackKey",
    "TrackletAppearance",
    "TrackletEmbeddingConfig",
    "TrackletEmbeddingStore",
    "TrackletSeparationDiagnostic",
    "evaluate_tracklet_separation",
]
