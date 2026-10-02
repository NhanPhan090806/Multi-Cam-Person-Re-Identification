"""Conservative proposals to reconcile existing IDs, without mutating the registry."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from multicam_reid.association.global_registry import cosine_distance_matrix
from multicam_reid.reid.tracklets import TrackKey, TrackletAppearance


@dataclass(frozen=True, slots=True)
class AssociationDecision:
    frame_id: int
    timestamp: float
    phase: str
    reason: str
    keys: tuple[TrackKey, ...] = ()
    global_ids: tuple[int, ...] = ()
    cosine_distance: float | None = None
    threshold: float | None = None
    confirmations: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {"event": "association_decision", **asdict(self)}


@dataclass(frozen=True, slots=True)
class MergeProposal:
    left_id: int
    right_id: int
    cosine_distance: float


@dataclass(frozen=True, slots=True)
class _Evidence:
    samples: Mapping[TrackKey, int]
    count: int
    first_timestamp: float
    last_timestamp: float


class OverlapReconciler:
    """Require mutual-best, repeated two-sided evidence and a conflict-free topology."""

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        threshold: float,
        confirmations: int,
        min_seconds: float,
        margin: float,
        min_samples: int,
    ) -> None:
        self.pairs = set(pairs)
        self.threshold = threshold
        self.confirmations = confirmations
        self.min_seconds = min_seconds
        self.margin = margin
        self.min_samples = min_samples
        self._evidence: dict[tuple[int, int], _Evidence] = {}
        self._conflicts: set[tuple[int, int]] = set()

    def observe(self, groups: Mapping[int, set[TrackKey]]) -> None:
        """Remember identities seen together in one camera, even after they depart."""
        self._conflicts = {pair for pair in self._conflicts if all(gid in groups for gid in pair)}
        by_camera = defaultdict(set)
        for gid, keys in groups.items():
            for key in keys:
                by_camera[key.camera_id].add(gid)
        for ids in by_camera.values():
            self._conflicts.update(combinations(sorted(ids), 2))

    def record_merge(self, survivor: int, merged: int) -> None:
        """Carry exclusion evidence to the survivor; discard stale merge confirmations."""
        self._conflicts = {
            tuple(sorted(survivor if gid == merged else gid for gid in pair))
            for pair in self._conflicts
            if not set(pair) <= {survivor, merged}
        }
        self._evidence = {
            pair: evidence
            for pair, evidence in self._evidence.items()
            if survivor not in pair and merged not in pair
        }

    def propose(
        self,
        groups: Mapping[int, set[TrackKey]],
        appearances: Mapping[TrackKey, TrackletAppearance],
        *,
        frame_id: int,
        timestamp: float,
    ) -> tuple[tuple[MergeProposal, ...], tuple[AssociationDecision, ...]]:
        self.observe(groups)
        distances: dict[tuple[int, int], float] = {}
        decisions: list[AssociationDecision] = []

        def log(reason, pair, distance=None, count=None):
            decisions.append(
                AssociationDecision(
                    frame_id,
                    timestamp,
                    "reconciliation",
                    reason,
                    tuple(sorted(groups[pair[0]] | groups[pair[1]])),
                    pair,
                    distance,
                    self.threshold,
                    count,
                )
            )

        active = sorted(gid for gid, keys in groups.items() if keys)
        for pair in combinations(active, 2):
            left, right = (groups[gid] for gid in pair)
            if {key.camera_id for key in left} & {key.camera_id for key in right}:
                log("same_camera_conflict", pair)
                continue
            if pair in self._conflicts:
                log("historical_same_camera_conflict", pair)
                continue
            if any(
                tuple(sorted((a.camera_id, b.camera_id))) not in self.pairs
                for a in left
                for b in right
            ):
                log("overlap_not_allowed", pair)
                continue
            if any(
                key not in appearances or appearances[key].stored_samples < self.min_samples
                for key in left | right
            ):
                log("insufficient_appearance", pair)
                continue
            distance = float(
                cosine_distance_matrix(
                    np.stack([appearances[key].embedding for key in sorted(left)]),
                    np.stack([appearances[key].embedding for key in sorted(right)]),
                ).max()
            )
            distances[pair] = distance
            if distance > self.threshold:
                log("distance_rejected", pair, distance)

        partners = defaultdict(list)
        for (left, right), distance in distances.items():
            partners[left].append((distance, right))
            partners[right].append((distance, left))
        preferred = {}
        for gid, choices in partners.items():
            choices.sort()
            if len(choices) < 2 or choices[1][0] - choices[0][0] >= self.margin:
                preferred[gid] = choices[0][1]

        next_evidence = {}
        proposals = []
        for pair, distance in sorted(distances.items()):
            if distance > self.threshold:
                continue
            if preferred.get(pair[0]) != pair[1] or preferred.get(pair[1]) != pair[0]:
                log("ambiguous_reconciliation", pair, distance)
                continue
            samples = {key: appearances[key].total_samples for gid in pair for key in groups[gid]}
            previous = self._evidence.get(pair)
            if previous is None or samples.keys() != previous.samples.keys():
                evidence = _Evidence(samples, 1, timestamp, timestamp)
            elif all(samples[key] > previous.samples[key] for key in samples):
                evidence = _Evidence(
                    samples,
                    previous.count + 1,
                    previous.first_timestamp,
                    timestamp,
                )
            else:
                evidence = previous
            next_evidence[pair] = evidence
            if (
                evidence.count < self.confirmations
                or evidence.last_timestamp - evidence.first_timestamp < self.min_seconds
            ):
                log(
                    "waiting_reconciliation_confirmation"
                    if evidence.count < self.confirmations
                    else "waiting_reconciliation_duration",
                    pair,
                    distance,
                    evidence.count,
                )
                continue
            proposals.append(MergeProposal(*pair, distance))
            log("reconciliation_accepted", pair, distance, evidence.count)
        self._evidence = next_evidence
        return tuple(proposals), tuple(decisions)
