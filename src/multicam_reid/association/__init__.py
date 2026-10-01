"""Cross-camera identity association."""

from multicam_reid.association.global_registry import (
    AssociationConfig,
    AssociationResult,
    DistanceMatrix,
    GlobalIdentityRegistry,
    GlobalIdentitySnapshot,
    IdentityMergeEvent,
    TrackletMatch,
    cosine_distance_matrix,
    match_tracklets,
)
from multicam_reid.association.handoff import HandoffConfig, HandoffIdentityRegistry

__all__ = [
    "AssociationConfig",
    "AssociationResult",
    "DistanceMatrix",
    "GlobalIdentityRegistry",
    "GlobalIdentitySnapshot",
    "HandoffConfig",
    "HandoffIdentityRegistry",
    "IdentityMergeEvent",
    "TrackletMatch",
    "cosine_distance_matrix",
    "match_tracklets",
]
