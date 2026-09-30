"""Conservative cross-camera association of camera-local tracklets.

This module deliberately keeps detection and within-camera tracking out of the
association problem.  It consumes the normalized appearance summaries produced
by Stage 4 and assigns stable project-wide IDs to them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from multicam_reid.reid.tracklets import TrackKey, TrackletAppearance


def _normalized_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have shape (N, D) with N and D greater than zero")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} must not contain zero-length vectors")
    return matrix / norms


def _unit_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    return _normalized_rows(np.asarray(value, dtype=np.float32)[None, :], name=name)[0]


def cosine_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return pairwise cosine distance (``1 - similarity``) for two matrices."""
    left_rows = _normalized_rows(left, name="left embeddings")
    right_rows = _normalized_rows(right, name="right embeddings")
    if left_rows.shape[1] != right_rows.shape[1]:
        raise ValueError("left and right embeddings must have the same dimension")
    return np.clip(1.0 - left_rows @ right_rows.T, 0.0, 2.0).astype(np.float32)


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    """Fixed Stage 6 decision thresholds.

    ``max_cosine_distance=0.35`` is equivalent to requiring cosine similarity
    of at least 0.65.  It is intentionally declared before using EPFL as the
    Stage 6 development replay, so evaluation is not silently tuned per frame.
    """

    max_cosine_distance: float = 0.35
    min_stored_samples: int = 2
    max_idle_frames: int = 250

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_cosine_distance <= 2.0:
            raise ValueError("max_cosine_distance must be between 0 and 2")
        if self.min_stored_samples <= 0:
            raise ValueError("min_stored_samples must be positive")
        if self.max_idle_frames < 0:
            raise ValueError("max_idle_frames must not be negative")


@dataclass(frozen=True, slots=True)
class DistanceMatrix:
    """A labeled distance matrix retained for diagnostics."""

    left_keys: tuple[TrackKey, ...]
    right_keys: tuple[TrackKey, ...]
    values: np.ndarray = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class TrackletMatch:
    """One accepted one-to-one appearance match."""

    left: TrackKey
    right: TrackKey
    cosine_distance: float


@dataclass(frozen=True, slots=True)
class IdentityMergeEvent:
    """Auditable record of two global identities being joined."""

    frame_id: int
    timestamp: float
    survivor_global_id: int
    merged_global_id: int
    left: TrackKey
    right: TrackKey
    cosine_distance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "survivor_global_id": self.survivor_global_id,
            "merged_global_id": self.merged_global_id,
            "left": {"camera_id": self.left.camera_id, "local_id": self.left.local_id},
            "right": {"camera_id": self.right.camera_id, "local_id": self.right.local_id},
            "cosine_distance": self.cosine_distance,
        }


@dataclass(frozen=True, slots=True)
class GlobalIdentitySnapshot:
    """Read-only public view of one global identity."""

    global_id: int
    members: tuple[TrackKey, ...]
    active_members: tuple[TrackKey, ...]
    first_seen_frame_id: int
    last_seen_frame_id: int
    first_seen_timestamp: float
    last_seen_timestamp: float


@dataclass(frozen=True, slots=True)
class AssociationResult:
    """Global assignments and evidence produced for one synchronized instant."""

    assignments: Mapping[TrackKey, int]
    active_global_ids: tuple[int, ...]
    merge_events: tuple[IdentityMergeEvent, ...]
    distance_matrices: tuple[DistanceMatrix, ...]


@dataclass(slots=True)
class _GlobalIdentity:
    global_id: int
    embeddings: dict[TrackKey, np.ndarray]
    sample_counts: dict[TrackKey, int]
    last_seen_by_track: dict[TrackKey, int]
    first_seen_frame_id: int
    last_seen_frame_id: int
    first_seen_timestamp: float
    last_seen_timestamp: float

    @property
    def prototype(self) -> np.ndarray:
        mean = np.mean(np.stack(tuple(self.embeddings.values())), axis=0)
        return _unit_vector(mean, name="global identity prototype")


def match_tracklets(
    left: Sequence[TrackletAppearance],
    right: Sequence[TrackletAppearance],
    *,
    max_cosine_distance: float,
) -> tuple[DistanceMatrix, tuple[TrackletMatch, ...]]:
    """Hungarian-match two non-empty sets of tracklets under a distance gate."""
    if not 0.0 <= max_cosine_distance <= 2.0:
        raise ValueError("max_cosine_distance must be between 0 and 2")
    left_items = tuple(left)
    right_items = tuple(right)
    if not left_items or not right_items:
        raise ValueError("left and right tracklets must both be non-empty")
    if len({item.key for item in (*left_items, *right_items)}) != len(
        left_items
    ) + len(right_items):
        raise ValueError("tracklet keys must be unique across both sides")

    values = cosine_distance_matrix(
        np.stack([item.embedding for item in left_items]),
        np.stack([item.embedding for item in right_items]),
    )
    matrix = DistanceMatrix(
        left_keys=tuple(item.key for item in left_items),
        right_keys=tuple(item.key for item in right_items),
        values=values.copy(),
    )
    left_indices, right_indices = linear_sum_assignment(values)
    matches = tuple(
        TrackletMatch(
            left=left_items[left_index].key,
            right=right_items[right_index].key,
            cosine_distance=float(values[left_index, right_index]),
        )
        for left_index, right_index in zip(left_indices, right_indices, strict=True)
        if values[left_index, right_index] <= max_cosine_distance
    )
    return matrix, matches


class GlobalIdentityRegistry:
    """Maintain global IDs while local tracker IDs appear, vanish, and return."""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        self._next_global_id = 1
        self._global_by_id: dict[int, _GlobalIdentity] = {}
        self._global_by_track: dict[TrackKey, int] = {}

    def __len__(self) -> int:
        return len(self._global_by_id)

    @property
    def total_ids_issued(self) -> int:
        """Return how many numeric IDs have ever been minted, including merged IDs."""
        return self._next_global_id - 1

    def snapshot(
        self,
        global_id: int,
        *,
        active_keys: Collection[TrackKey] = (),
    ) -> GlobalIdentitySnapshot:
        """Return one identity and mark which of its members are active now."""
        identity = self._global_by_id[global_id]
        active = set(active_keys)
        members = tuple(
            sorted(
                key
                for key in identity.embeddings
                if self._global_by_track.get(key) == global_id
            )
        )
        return GlobalIdentitySnapshot(
            global_id=global_id,
            members=members,
            active_members=tuple(member for member in members if member in active),
            first_seen_frame_id=identity.first_seen_frame_id,
            last_seen_frame_id=identity.last_seen_frame_id,
            first_seen_timestamp=identity.first_seen_timestamp,
            last_seen_timestamp=identity.last_seen_timestamp,
        )

    def snapshots(
        self, *, active_keys: Collection[TrackKey] = ()
    ) -> tuple[GlobalIdentitySnapshot, ...]:
        """Return all identities in stable numeric order."""
        return tuple(
            self.snapshot(global_id, active_keys=active_keys)
            for global_id in sorted(self._global_by_id)
        )

    def update(
        self,
        appearances: Sequence[TrackletAppearance],
        *,
        frame_id: int,
        timestamp: float,
    ) -> AssociationResult:
        """Associate all appearances observed at one synchronized instant."""
        if frame_id < 0:
            raise ValueError("frame_id must not be negative")
        if timestamp < 0.0 or not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite and not negative")
        observed = tuple(appearances)
        if len({item.key for item in observed}) != len(observed):
            raise ValueError("appearances must contain unique track keys")
        self._validate_appearances(observed)

        active_keys = {item.key for item in observed}
        self._forget_expired_track_mappings(active_keys, frame_id)
        mature = tuple(
            item for item in observed if item.stored_samples >= self.config.min_stored_samples
        )

        self._reattach_from_gallery(mature, active_keys, frame_id)
        for item in observed:
            if item.key not in self._global_by_track:
                self._create_identity(item, frame_id=frame_id, timestamp=timestamp)
            self._record_appearance(item, frame_id=frame_id, timestamp=timestamp)

        matrices: list[DistanceMatrix] = []
        events: list[IdentityMergeEvent] = []
        by_camera: dict[str, list[TrackletAppearance]] = defaultdict(list)
        for item in mature:
            by_camera[item.key.camera_id].append(item)
        camera_ids = sorted(by_camera)
        for left_index, left_camera in enumerate(camera_ids):
            for right_camera in camera_ids[left_index + 1 :]:
                matrix, matches = match_tracklets(
                    by_camera[left_camera],
                    by_camera[right_camera],
                    max_cosine_distance=self.config.max_cosine_distance,
                )
                matrices.append(matrix)
                for match in matches:
                    event = self._merge_match(
                        match,
                        active_keys=active_keys,
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                    if event is not None:
                        events.append(event)

        assignments = {
            item.key: self._global_by_track[item.key]
            for item in observed
        }
        return AssociationResult(
            assignments=assignments,
            active_global_ids=tuple(sorted(set(assignments.values()))),
            merge_events=tuple(events),
            distance_matrices=tuple(matrices),
        )

    def _validate_appearances(self, observed: Sequence[TrackletAppearance]) -> None:
        dimension: int | None = None
        for item in observed:
            vector = _unit_vector(item.embedding, name=f"appearance for {item.key}")
            if dimension is None:
                dimension = int(vector.size)
            elif vector.size != dimension:
                raise ValueError("all appearance embeddings must have the same dimension")
            if item.stored_samples <= 0 or item.total_samples < item.stored_samples:
                raise ValueError("appearance sample counts are inconsistent")

    def _forget_expired_track_mappings(
        self, active_keys: Collection[TrackKey], frame_id: int
    ) -> None:
        for key, global_id in tuple(self._global_by_track.items()):
            identity = self._global_by_id[global_id]
            if (
                key not in active_keys
                and frame_id - identity.last_seen_by_track[key]
                > self.config.max_idle_frames
            ):
                del self._global_by_track[key]

    def _reattach_from_gallery(
        self,
        mature: Sequence[TrackletAppearance],
        active_keys: Collection[TrackKey],
        frame_id: int,
    ) -> None:
        unmapped_by_camera: dict[str, list[TrackletAppearance]] = defaultdict(list)
        for item in mature:
            if item.key not in self._global_by_track:
                unmapped_by_camera[item.key.camera_id].append(item)

        active_cameras_by_global: dict[int, set[str]] = defaultdict(set)
        for key in active_keys:
            global_id = self._global_by_track.get(key)
            if global_id is not None:
                active_cameras_by_global[global_id].add(key.camera_id)

        claimed_globals: set[int] = set()
        for camera_id in sorted(unmapped_by_camera):
            candidates = [
                identity
                for identity in self._global_by_id.values()
                if frame_id - identity.last_seen_frame_id <= self.config.max_idle_frames
                and camera_id not in active_cameras_by_global[identity.global_id]
                and identity.global_id not in claimed_globals
            ]
            items = unmapped_by_camera[camera_id]
            if not items or not candidates:
                continue
            candidates.sort(key=lambda identity: identity.global_id)
            distances = cosine_distance_matrix(
                np.stack([item.embedding for item in items]),
                np.stack([identity.prototype for identity in candidates]),
            )
            item_indices, candidate_indices = linear_sum_assignment(distances)
            for item_index, candidate_index in zip(
                item_indices, candidate_indices, strict=True
            ):
                if distances[item_index, candidate_index] > self.config.max_cosine_distance:
                    continue
                item = items[item_index]
                global_id = candidates[candidate_index].global_id
                self._global_by_track[item.key] = global_id
                claimed_globals.add(global_id)
                active_cameras_by_global[global_id].add(camera_id)

    def _create_identity(
        self, item: TrackletAppearance, *, frame_id: int, timestamp: float
    ) -> None:
        global_id = self._next_global_id
        self._next_global_id += 1
        self._global_by_track[item.key] = global_id
        self._global_by_id[global_id] = _GlobalIdentity(
            global_id=global_id,
            embeddings={},
            sample_counts={},
            last_seen_by_track={},
            first_seen_frame_id=frame_id,
            last_seen_frame_id=frame_id,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
        )

    def _record_appearance(
        self, item: TrackletAppearance, *, frame_id: int, timestamp: float
    ) -> None:
        identity = self._global_by_id[self._global_by_track[item.key]]
        prior_count = identity.sample_counts.get(item.key, -1)
        if item.total_samples >= prior_count:
            identity.embeddings[item.key] = _unit_vector(
                item.embedding, name=f"appearance for {item.key}"
            )
            identity.sample_counts[item.key] = item.total_samples
        identity.last_seen_by_track[item.key] = frame_id
        identity.last_seen_frame_id = max(identity.last_seen_frame_id, frame_id)
        identity.last_seen_timestamp = max(identity.last_seen_timestamp, timestamp)

    def _merge_match(
        self,
        match: TrackletMatch,
        *,
        active_keys: Collection[TrackKey],
        frame_id: int,
        timestamp: float,
    ) -> IdentityMergeEvent | None:
        left_id = self._global_by_track[match.left]
        right_id = self._global_by_track[match.right]
        if left_id == right_id:
            return None

        active = set(active_keys)
        left_cameras = {
            key.camera_id
            for key, global_id in self._global_by_track.items()
            if global_id == left_id and key in active
        }
        right_cameras = {
            key.camera_id
            for key, global_id in self._global_by_track.items()
            if global_id == right_id and key in active
        }
        if left_cameras & right_cameras:
            return None

        survivor_id, merged_id = sorted((left_id, right_id))
        survivor = self._global_by_id[survivor_id]
        merged = self._global_by_id.pop(merged_id)
        for key, embedding in merged.embeddings.items():
            survivor.embeddings[key] = embedding
            survivor.sample_counts[key] = merged.sample_counts[key]
            survivor.last_seen_by_track[key] = merged.last_seen_by_track[key]
            if self._global_by_track.get(key) == merged_id:
                self._global_by_track[key] = survivor_id
        survivor.first_seen_frame_id = min(
            survivor.first_seen_frame_id, merged.first_seen_frame_id
        )
        survivor.last_seen_frame_id = max(
            survivor.last_seen_frame_id, merged.last_seen_frame_id
        )
        survivor.first_seen_timestamp = min(
            survivor.first_seen_timestamp, merged.first_seen_timestamp
        )
        survivor.last_seen_timestamp = max(
            survivor.last_seen_timestamp, merged.last_seen_timestamp
        )
        return IdentityMergeEvent(
            frame_id=frame_id,
            timestamp=timestamp,
            survivor_global_id=survivor_id,
            merged_global_id=merged_id,
            left=match.left,
            right=match.right,
            cosine_distance=match.cosine_distance,
        )
