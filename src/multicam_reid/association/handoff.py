"""Appearance-gallery handoff between cameras watching separate locations.

Tracks wait for usable appearance evidence. An arriving track can claim only an
inactive identity, after a departure grace period, within a seconds-based TTL.
No synchronized cross-camera observations or ground-plane calibration is needed.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from multicam_reid.association.global_registry import (
    GlobalIdentitySnapshot,
    cosine_distance_matrix,
)
from multicam_reid.reid.tracklets import TrackKey, TrackletAppearance


@dataclass(frozen=True, slots=True)
class HandoffConfig:
    max_cosine_distance: float = 0.35
    min_stored_samples: int = 2
    gallery_ttl_seconds: float = 120.0
    exit_grace_seconds: float = 1.0
    min_travel_seconds: float = 0.0
    match_margin: float = 0.05
    allowed_transitions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "max_cosine_distance",
            "gallery_ttl_seconds",
            "exit_grace_seconds",
            "min_travel_seconds",
            "match_margin",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.gallery_ttl_seconds == 0:
            raise ValueError("gallery_ttl_seconds must be positive")
        if self.max_cosine_distance > 2 or self.match_margin > 2:
            raise ValueError("cosine distance and margin must be at most two")
        if self.min_stored_samples <= 0:
            raise ValueError("min_stored_samples must be positive")
        if any(
            len(edge) != 2 or not all(isinstance(v, str) and v.strip() for v in edge)
            for edge in self.allowed_transitions
        ):
            raise ValueError("allowed_transitions must contain pairs of nonempty camera IDs")


@dataclass(frozen=True, slots=True)
class HandoffEvent:
    frame_id: int
    timestamp: float
    global_id: int
    from_camera: str
    from_local_id: int
    to_camera: str
    to_local_id: int
    gap_seconds: float
    cosine_distance: float

    def to_dict(self) -> dict[str, object]:
        return {"event": "handoff", **asdict(self)}


@dataclass(frozen=True, slots=True)
class HandoffResult:
    assignments: Mapping[TrackKey, int]
    active_global_ids: tuple[int, ...]
    handoff_events: tuple[HandoffEvent, ...]
    # Same serving interface as the historical overlap registry.
    merge_events: tuple = ()
    distance_matrices: tuple = ()


@dataclass(slots=True)
class _Identity:
    global_id: int
    current_key: TrackKey
    templates: dict[str, np.ndarray]
    members: set[TrackKey]
    first_frame: int
    last_frame: int
    first_seen: float
    last_seen: float
    absent_since: float | None = None


class HandoffIdentityRegistry:
    """Remember departed people and recover their global ID on later arrivals."""

    def __init__(self, config: HandoffConfig | None = None) -> None:
        self.config = config or HandoffConfig()
        self._identities: dict[int, _Identity] = {}
        self._by_track: dict[TrackKey, int] = {}
        self._next_id = 1
        self._last_timestamp = -1.0
        self._dimension: int | None = None

    def __len__(self) -> int:
        return len(self._identities)

    @property
    def total_ids_issued(self) -> int:
        return self._next_id - 1

    def snapshots(self, *, active_keys: Collection[TrackKey] = ()):
        active = set(active_keys)
        return tuple(
            GlobalIdentitySnapshot(
                item.global_id,
                tuple(sorted(item.members)),
                (item.current_key,) if item.current_key in active else (),
                item.first_frame,
                item.last_frame,
                item.first_seen,
                item.last_seen,
            )
            for item in sorted(self._identities.values(), key=lambda item: item.global_id)
        )

    def update(
        self,
        appearances: Sequence[TrackletAppearance],
        *,
        frame_id: int,
        timestamp: float,
        visible_keys: Collection[TrackKey] | None = None,
    ) -> HandoffResult:
        if frame_id < 0 or not np.isfinite(timestamp) or timestamp < self._last_timestamp:
            raise ValueError("frame_id must be nonnegative and timestamps monotonic and finite")
        if timestamp < 0:
            raise ValueError("timestamp must be nonnegative")
        keys = [item.key for item in appearances]
        if len(keys) != len(set(keys)):
            raise ValueError("appearances must contain unique keys")
        vectors: dict[TrackKey, np.ndarray] = {}
        dimension = self._dimension
        for item in appearances:
            vector = np.asarray(item.embedding, dtype=np.float32)
            if (
                vector.ndim != 1
                or not vector.size
                or not np.all(np.isfinite(vector))
                or np.linalg.norm(vector) <= 1e-12
            ):
                raise ValueError("appearance must be a finite nonzero vector")
            if dimension is not None and vector.size != dimension:
                raise ValueError("appearance dimensions must agree across updates")
            if item.stored_samples <= 0 or item.total_samples < item.stored_samples:
                raise ValueError("appearance sample counts are inconsistent")
            dimension = vector.size
            vectors[item.key] = vector / np.linalg.norm(vector)
        self._dimension = dimension
        self._last_timestamp = timestamp
        visible = set(keys) if visible_keys is None else set(visible_keys) | set(keys)

        for global_id, identity in tuple(self._identities.items()):
            if identity.current_key in visible:
                identity.last_seen = timestamp
                identity.last_frame = frame_id
                identity.absent_since = None
            else:
                if identity.absent_since is None:
                    identity.absent_since = timestamp
                if timestamp - identity.last_seen > self.config.gallery_ttl_seconds:
                    del self._identities[global_id]
                    self._by_track.pop(identity.current_key, None)

        arrivals = sorted(
            (
                item
                for item in appearances
                if item.key not in self._by_track
                and item.stored_samples >= self.config.min_stored_samples
            ),
            key=lambda item: item.key,
        )
        candidates = sorted(
            (
                item
                for item in self._identities.values()
                if item.current_key not in visible and item.absent_since is not None
            ),
            key=lambda item: item.global_id,
        )
        events: list[HandoffEvent] = []
        ambiguous: set[TrackKey] = set()
        if arrivals and candidates:
            distances = np.full((len(arrivals), len(candidates)), 3.0, dtype=np.float32)
            for row, arrival in enumerate(arrivals):
                for col, identity in enumerate(candidates):
                    edge = (identity.current_key.camera_id, arrival.key.camera_id)
                    if (
                        self.config.allowed_transitions
                        and edge[0] != edge[1]
                        and edge not in self.config.allowed_transitions
                    ):
                        continue
                    distance = cosine_distance_matrix(
                        vectors[arrival.key][None, :],
                        np.stack(tuple(identity.templates.values())),
                    ).min()
                    if (
                        timestamp - identity.last_seen < self.config.min_travel_seconds
                        or timestamp - identity.absent_since < self.config.exit_grace_seconds
                    ):
                        if distance <= self.config.max_cosine_distance:
                            ambiguous.add(arrival.key)
                        continue
                    distances[row, col] = distance
                ordered = np.sort(distances[row])
                if (
                    len(ordered) >= 2
                    and ordered[0] <= self.config.max_cosine_distance
                    and ordered[1] - ordered[0] < self.config.match_margin
                ):
                    ambiguous.add(arrival.key)
                if arrival.key in ambiguous:
                    distances[row] = 3.0
            # Gate before assignment; dummy columns permit unmatched arrivals.
            costs = np.where(distances <= self.config.max_cosine_distance, distances, 3.0)
            dummy = np.full((len(arrivals), len(arrivals)), self.config.max_cosine_distance + 0.001)
            rows, cols = linear_sum_assignment(np.hstack((costs, dummy)))
            for row, col in zip(rows, cols, strict=True):
                if col >= len(candidates) or costs[row, col] > self.config.max_cosine_distance:
                    continue
                arrival, identity = arrivals[row], candidates[col]
                previous = identity.current_key
                events.append(
                    HandoffEvent(
                        frame_id,
                        timestamp,
                        identity.global_id,
                        previous.camera_id,
                        previous.local_id,
                        arrival.key.camera_id,
                        arrival.key.local_id,
                        timestamp - identity.last_seen,
                        float(distances[row, col]),
                    )
                )
                self._by_track.pop(previous, None)
                identity.current_key = arrival.key
                identity.members.add(arrival.key)
                identity.absent_since = None
                self._by_track[arrival.key] = identity.global_id

        for item in arrivals:
            if item.key not in self._by_track and item.key not in ambiguous:
                global_id = self._next_id
                self._next_id += 1
                self._by_track[item.key] = global_id
                self._identities[global_id] = _Identity(
                    global_id,
                    item.key,
                    {},
                    {item.key},
                    frame_id,
                    frame_id,
                    timestamp,
                    timestamp,
                )
        for item in appearances:
            global_id = self._by_track.get(item.key)
            if global_id is None:
                continue
            identity = self._identities[global_id]
            # Avoid replacing a mature template with one low-quality sample.
            if item.stored_samples >= self.config.min_stored_samples:
                identity.templates[item.key.camera_id] = vectors[item.key].copy()
            identity.last_seen = timestamp
            identity.last_frame = frame_id
            identity.absent_since = None
        assignments = {key: self._by_track[key] for key in visible if key in self._by_track}
        return HandoffResult(assignments, tuple(sorted(set(assignments.values()))), tuple(events))
