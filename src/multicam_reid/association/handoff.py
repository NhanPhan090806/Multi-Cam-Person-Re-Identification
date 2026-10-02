"""Appearance-gallery handoff, with optional pair-scoped simultaneous matching.

Tracks wait for usable appearance evidence. An arriving track can claim only an
inactive identity, after a departure grace period, within a seconds-based TTL.
No ground-plane calibration is needed. Declared overlapping views require fresh
appearance confirmations and pairwise-compatible camera occupancy.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import groupby

import numpy as np
from scipy.optimize import linear_sum_assignment

from multicam_reid.association.global_registry import (
    GlobalIdentitySnapshot,
    IdentityMergeEvent,
    cosine_distance_matrix,
)
from multicam_reid.association.reconciliation import AssociationDecision, OverlapReconciler
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
    overlap_pairs: tuple[tuple[str, str], ...] = ()
    overlap_max_cosine_distance: float = 0.25
    overlap_confirmations: int = 3
    reconciliation_enabled: bool = True
    reconciliation_confirmations: int = 5
    reconciliation_min_seconds: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "max_cosine_distance",
            "gallery_ttl_seconds",
            "exit_grace_seconds",
            "min_travel_seconds",
            "match_margin",
            "overlap_max_cosine_distance",
            "reconciliation_min_seconds",
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
        if self.overlap_max_cosine_distance > 2:
            raise ValueError("overlap_max_cosine_distance must be at most two")
        if not isinstance(self.overlap_confirmations, int) or self.overlap_confirmations <= 0:
            raise ValueError("overlap_confirmations must be a positive integer")
        if (
            not isinstance(self.reconciliation_confirmations, int)
            or self.reconciliation_confirmations <= 0
        ):
            raise ValueError("reconciliation_confirmations must be a positive integer")
        if not isinstance(self.reconciliation_enabled, bool):
            raise ValueError("reconciliation_enabled must be boolean")
        if any(
            len(edge) != 2 or not all(isinstance(v, str) and v.strip() for v in edge)
            for edge in self.allowed_transitions
        ):
            raise ValueError("allowed_transitions must contain pairs of nonempty camera IDs")
        if any(
            len(edge) != 2
            or not all(isinstance(v, str) and v.strip() for v in edge)
            or edge[0] == edge[1]
            for edge in self.overlap_pairs
        ):
            raise ValueError("overlap_pairs must contain pairs of distinct nonempty camera IDs")
        object.__setattr__(
            self,
            "overlap_pairs",
            tuple(sorted({tuple(sorted(edge)) for edge in self.overlap_pairs})),
        )


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
    event: str = "handoff"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HandoffResult:
    assignments: Mapping[TrackKey, int]
    active_global_ids: tuple[int, ...]
    handoff_events: tuple[HandoffEvent, ...]
    overlap_events: tuple[HandoffEvent, ...] = ()
    # Same serving interface as the historical overlap registry.
    merge_events: tuple[IdentityMergeEvent, ...] = ()
    distance_matrices: tuple = ()
    decisions: tuple[AssociationDecision, ...] = ()


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
    bound_keys: set[TrackKey] = field(default_factory=set)


class HandoffIdentityRegistry:
    """Recover departed people; opt into simultaneous matches for declared pairs."""

    def __init__(self, config: HandoffConfig | None = None) -> None:
        self.config = config or HandoffConfig()
        self._identities: dict[int, _Identity] = {}
        self._by_track: dict[TrackKey, int] = {}
        self._next_id = 1
        self._last_timestamp = -1.0
        self._dimension: int | None = None
        # (candidate global ID, confirmation count, last distinct embedding sample).
        self._overlap_evidence: dict[TrackKey, tuple[int, int, int]] = {}
        self._aliases: dict[int, int] = {}
        self._decisions: list[AssociationDecision] = []
        self._reconciler = OverlapReconciler(
            self.config.overlap_pairs,
            threshold=min(self.config.max_cosine_distance, self.config.overlap_max_cosine_distance),
            confirmations=self.config.reconciliation_confirmations,
            min_seconds=self.config.reconciliation_min_seconds,
            margin=self.config.match_margin,
            min_samples=self.config.min_stored_samples,
        )

    @property
    def identity_aliases(self) -> dict[int, int]:
        """Final aliases for audit only; previously emitted assignments are not rewritten."""
        result = {}
        for old, target in self._aliases.items():
            while target in self._aliases:
                target = self._aliases[target]
            result[old] = target
        return result

    def _log(
        self,
        reason: str,
        keys: Sequence[TrackKey] = (),
        global_ids: tuple[int, ...] = (),
        distance: float | None = None,
        threshold: float | None = None,
        count: int | None = None,
    ) -> None:
        self._decisions.append(
            AssociationDecision(
                self._frame_id,
                self._last_timestamp,
                "arrival",
                reason,
                tuple(keys),
                global_ids,
                float(distance) if distance is not None else None,
                threshold,
                count,
            )
        )

    @property
    def association_mode(self) -> str:
        return "hybrid" if self.config.overlap_pairs else "handoff"

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
                tuple(sorted(item.bound_keys & active)),
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
        self._frame_id = frame_id
        self._decisions = []
        visible = set(keys) if visible_keys is None else set(visible_keys) | set(keys)

        for global_id, identity in tuple(self._identities.items()):
            active = identity.bound_keys & visible
            if active:
                if identity.current_key not in active:
                    identity.current_key = min(active)
                identity.last_seen = timestamp
                identity.last_frame = frame_id
                identity.absent_since = None
            else:
                if identity.absent_since is None:
                    identity.absent_since = timestamp
                if timestamp - identity.last_seen > self.config.gallery_ttl_seconds:
                    self._log("gallery_expired", tuple(identity.bound_keys), (global_id,))
                    del self._identities[global_id]
                    for key in identity.bound_keys:
                        self._by_track.pop(key, None)

        self._overlap_evidence = {
            key: evidence for key, evidence in self._overlap_evidence.items() if key in visible
        }
        # Use this update's mature templates, not a stale previous camera observation.
        for item in appearances:
            global_id = self._by_track.get(item.key)
            if global_id is not None and item.stored_samples >= self.config.min_stored_samples:
                self._identities[global_id].templates[item.key.camera_id] = vectors[item.key].copy()

        arrivals = sorted(
            (
                item
                for item in appearances
                if item.key not in self._by_track
                and item.stored_samples >= self.config.min_stored_samples
            ),
            key=lambda item: item.key,
        )
        for item in appearances:
            if (
                item.key not in self._by_track
                and item.stored_samples < self.config.min_stored_samples
            ):
                self._log("insufficient_appearance", (item.key,))
        events: list[HandoffEvent] = []
        overlaps: list[HandoffEvent] = []
        if self.config.overlap_pairs:
            # Per-camera assignment allows several permitted views of one identity,
            # but never two tracks from the same camera. Recompute occupancy each batch.
            batches = [list(items) for _, items in groupby(arrivals, key=lambda a: a.key.camera_id)]
        else:
            batches = [arrivals]
        for batch in batches:
            self._assign_arrivals(batch, vectors, visible, frame_id, timestamp, events, overlaps)

        for item in appearances:
            global_id = self._by_track.get(item.key)
            if global_id is None:
                continue
            identity = self._identities[global_id]
            if item.stored_samples >= self.config.min_stored_samples:
                identity.templates[item.key.camera_id] = vectors[item.key].copy()
            identity.last_seen = timestamp
            identity.last_frame = frame_id
            identity.absent_since = None
        merges = []
        if self.config.overlap_pairs and self.config.reconciliation_enabled:
            proposals, decisions = self._reconciler.propose(
                {gid: identity.bound_keys & visible for gid, identity in self._identities.items()},
                {item.key: item for item in appearances},
                frame_id=frame_id,
                timestamp=timestamp,
            )
            self._decisions.extend(decisions)
            for proposal in proposals:
                survivor = self._identities[proposal.left_id]
                merged = self._identities[proposal.right_id]
                left, right = min(survivor.bound_keys & visible), min(merged.bound_keys & visible)
                merges.append(
                    IdentityMergeEvent(
                        frame_id,
                        timestamp,
                        survivor.global_id,
                        merged.global_id,
                        left,
                        right,
                        proposal.cosine_distance,
                    )
                )
                active = (survivor.bound_keys | merged.bound_keys) & visible
                for key in (survivor.bound_keys | merged.bound_keys) - active:
                    self._by_track.pop(key, None)
                for key in active:
                    self._by_track[key] = survivor.global_id
                survivor.bound_keys = active
                survivor.members |= merged.members
                survivor.templates = {**merged.templates, **survivor.templates}
                for key in active:
                    survivor.templates[key.camera_id] = vectors[key].copy()
                survivor.current_key = left
                survivor.first_frame = min(survivor.first_frame, merged.first_frame)
                survivor.first_seen = min(survivor.first_seen, merged.first_seen)
                self._aliases[merged.global_id] = survivor.global_id
                del self._identities[merged.global_id]
                self._reconciler.record_merge(survivor.global_id, merged.global_id)
                # Confirmations naming either changed group must start again.
                self._overlap_evidence = {
                    key: evidence
                    for key, evidence in self._overlap_evidence.items()
                    if evidence[0] not in {survivor.global_id, merged.global_id}
                }
        assignments = {key: self._by_track[key] for key in visible if key in self._by_track}
        return HandoffResult(
            assignments,
            tuple(sorted(set(assignments.values()))),
            tuple(events),
            tuple(overlaps),
            merge_events=tuple(merges),
            decisions=tuple(self._decisions),
        )

    def _assign_arrivals(
        self,
        arrivals: Sequence[TrackletAppearance],
        vectors: Mapping[TrackKey, np.ndarray],
        visible: set[TrackKey],
        frame_id: int,
        timestamp: float,
        events: list[HandoffEvent],
        overlaps: list[HandoffEvent],
    ) -> None:
        """Match a batch one-to-one and leave uncertain overlap arrivals pending."""
        candidates = sorted(
            (
                item
                for item in self._identities.values()
                if self.config.overlap_pairs or item.absent_since is not None
            ),
            key=lambda item: item.global_id,
        )
        ambiguous: set[TrackKey] = set()
        pending_overlap: set[TrackKey] = set()
        if arrivals and candidates:
            distances = np.full((len(arrivals), len(candidates)), 3.0, dtype=np.float32)
            limits = np.full_like(distances, self.config.max_cosine_distance)
            for row, arrival in enumerate(arrivals):
                for col, identity in enumerate(candidates):
                    active = identity.bound_keys & visible
                    if active:
                        if any(key.camera_id == arrival.key.camera_id for key in active):
                            self._log("same_camera_conflict", (arrival.key,), (identity.global_id,))
                            continue
                        # All concurrently occupied pairs must be explicitly allowed.
                        if any(
                            tuple(sorted((key.camera_id, arrival.key.camera_id)))
                            not in self.config.overlap_pairs
                            for key in active
                        ):
                            self._log("overlap_not_allowed", (arrival.key,), (identity.global_id,))
                            continue
                        references = [identity.templates[key.camera_id] for key in sorted(active)]
                        distance = float(
                            cosine_distance_matrix(
                                vectors[arrival.key][None, :],
                                np.stack(references),
                            ).max()
                        )
                        limits[row, col] = min(
                            self.config.max_cosine_distance,
                            self.config.overlap_max_cosine_distance,
                        )
                        if distance <= limits[row, col]:
                            distances[row, col] = distance
                        self._log(
                            "candidate_eligible"
                            if distance <= limits[row, col]
                            else "distance_rejected",
                            (arrival.key,),
                            (identity.global_id,),
                            distance,
                            float(limits[row, col]),
                        )
                        continue
                    edge = (identity.current_key.camera_id, arrival.key.camera_id)
                    if (
                        self.config.allowed_transitions
                        and edge[0] != edge[1]
                        and edge not in self.config.allowed_transitions
                    ):
                        self._log("transition_not_allowed", (arrival.key,), (identity.global_id,))
                        continue
                    distance = cosine_distance_matrix(
                        vectors[arrival.key][None, :],
                        np.stack(tuple(identity.templates.values())),
                    ).min()
                    if timestamp - identity.last_seen < self.config.min_travel_seconds or (
                        identity.absent_since is not None
                        and timestamp - identity.absent_since < self.config.exit_grace_seconds
                    ):
                        if distance <= self.config.max_cosine_distance:
                            ambiguous.add(arrival.key)
                        self._log(
                            "waiting_exit_or_travel"
                            if distance <= self.config.max_cosine_distance
                            else "distance_rejected",
                            (arrival.key,),
                            (identity.global_id,),
                            distance,
                            self.config.max_cosine_distance,
                        )
                        continue
                    if distance <= limits[row, col]:
                        distances[row, col] = distance
                    self._log(
                        "candidate_eligible"
                        if distance <= limits[row, col]
                        else "distance_rejected",
                        (arrival.key,),
                        (identity.global_id,),
                        distance,
                        float(limits[row, col]),
                    )
                ordered = np.sort(distances[row])
                if (
                    len(ordered) >= 2
                    and ordered[0] <= self.config.max_cosine_distance
                    and ordered[1] - ordered[0] < self.config.match_margin
                ):
                    ambiguous.add(arrival.key)
                    self._log("ambiguous_identity", (arrival.key,))
                if arrival.key in ambiguous:
                    distances[row] = 3.0
            if self.config.overlap_pairs:
                # A column competition is ambiguous too: do not choose arbitrarily
                # between two near-identical people arriving in the same camera.
                for col in range(len(candidates)):
                    ordered = np.sort(distances[:, col])
                    if (
                        len(ordered) >= 2
                        and ordered[0] <= self.config.max_cosine_distance
                        and ordered[1] - ordered[0] < self.config.match_margin
                    ):
                        ambiguous.update(
                            arrivals[row].key
                            for row in range(len(arrivals))
                            if distances[row, col] <= ordered[0] + self.config.match_margin
                        )
                        self._log("ambiguous_arrivals", global_ids=(candidates[col].global_id,))
                for row, arrival in enumerate(arrivals):
                    if arrival.key in ambiguous:
                        distances[row] = 3.0
            # Gate before assignment; dummy columns permit unmatched arrivals.
            costs = np.where(distances <= limits, distances, 3.0)
            dummy = np.full((len(arrivals), len(arrivals)), self.config.max_cosine_distance + 0.001)
            rows, cols = linear_sum_assignment(np.hstack((costs, dummy)))
            for row, col in zip(rows, cols, strict=True):
                if col >= len(candidates) or costs[row, col] > self.config.max_cosine_distance:
                    continue
                arrival, identity = arrivals[row], candidates[col]
                active = identity.bound_keys & visible
                if active:
                    old_id, count, sample = self._overlap_evidence.get(arrival.key, (-1, 0, -1))
                    if old_id != identity.global_id:
                        count, sample = 0, -1
                    if arrival.total_samples > sample:
                        count += 1
                    self._overlap_evidence[arrival.key] = (
                        identity.global_id,
                        count,
                        arrival.total_samples,
                    )
                    pending_overlap.add(arrival.key)
                    if count < self.config.overlap_confirmations:
                        self._log(
                            "waiting_overlap_confirmation",
                            (arrival.key,),
                            (identity.global_id,),
                            float(distances[row, col]),
                            float(limits[row, col]),
                            count,
                        )
                        continue
                previous = identity.current_key
                self._log(
                    "overlap_assigned" if active else "handoff_assigned",
                    (arrival.key,),
                    (identity.global_id,),
                    float(distances[row, col]),
                    float(limits[row, col]),
                )
                (overlaps if active else events).append(
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
                        "overlap" if active else "handoff",
                    )
                )
                # Remove departed bindings to avoid phantom ownership on return trips.
                for key in identity.bound_keys - active:
                    self._by_track.pop(key, None)
                identity.bound_keys = active | {arrival.key}
                identity.current_key = arrival.key
                identity.members.add(arrival.key)
                identity.absent_since = None
                self._by_track[arrival.key] = identity.global_id
                identity.templates[arrival.key.camera_id] = vectors[arrival.key].copy()
                identity.last_seen = timestamp
                identity.last_frame = frame_id
                self._overlap_evidence.pop(arrival.key, None)

        for item in arrivals:
            if item.key not in pending_overlap:
                self._overlap_evidence.pop(item.key, None)
            if (
                item.key not in self._by_track
                and item.key not in ambiguous
                and item.key not in pending_overlap
            ):
                global_id = self._next_id
                self._next_id += 1
                self._by_track[item.key] = global_id
                self._log("new_identity", (item.key,), (global_id,))
                self._identities[global_id] = _Identity(
                    global_id,
                    item.key,
                    {item.key.camera_id: vectors[item.key].copy()},
                    {item.key},
                    frame_id,
                    frame_id,
                    timestamp,
                    timestamp,
                    bound_keys={item.key},
                )
