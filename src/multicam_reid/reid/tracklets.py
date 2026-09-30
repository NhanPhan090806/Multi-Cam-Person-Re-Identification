"""Rolling OSNet appearance representations for camera-local person tracks."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from multicam_reid.pipeline.single_camera import PersonCrop


class CropEncoder(Protocol):
    """Minimal encoder contract implemented by :class:`ReIDEncoder`."""

    def encode(self, crops: Sequence[np.ndarray]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True, order=True)
class TrackKey:
    """A local ID qualified by its camera namespace."""

    camera_id: str
    local_id: int

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if self.local_id <= 0:
            raise ValueError("local_id must be positive")


@dataclass(frozen=True, slots=True)
class TrackletEmbeddingConfig:
    """Sampling and retention settings for rolling tracklet embeddings."""

    history_size: int = 10
    embedding_interval_frames: int = 5
    stale_after_frames: int = 150

    def __post_init__(self) -> None:
        if self.history_size <= 0:
            raise ValueError("history_size must be positive")
        if self.embedding_interval_frames <= 0:
            raise ValueError("embedding_interval_frames must be positive")
        if self.stale_after_frames < 0:
            raise ValueError("stale_after_frames must not be negative")


@dataclass(frozen=True, slots=True, eq=False)
class TrackletAppearance:
    """The current normalized appearance summary for one local tracklet."""

    key: TrackKey
    embedding: np.ndarray
    stored_samples: int
    total_samples: int
    first_seen_frame_id: int
    last_seen_frame_id: int
    first_seen_timestamp: float
    last_seen_timestamp: float


@dataclass(frozen=True, slots=True)
class TrackletSeparationDiagnostic:
    """Cosine-similarity evidence for the Stage 4 quality gate."""

    same_track_mean: float
    different_track_mean: float
    margin: float
    same_track_pairs: int
    different_track_pairs: int
    passed: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        """Return JSON-safe scalar values."""
        return {
            "same_track_mean": self.same_track_mean,
            "different_track_mean": self.different_track_mean,
            "margin": self.margin,
            "same_track_pairs": self.same_track_pairs,
            "different_track_pairs": self.different_track_pairs,
            "passed": self.passed,
        }


@dataclass(slots=True)
class _TrackletState:
    history: deque[np.ndarray]
    total_samples: int
    first_seen_frame_id: int
    last_seen_frame_id: int
    last_sample_frame_id: int
    first_seen_timestamp: float
    last_seen_timestamp: float


def _unit_vector(vector: np.ndarray, *, context: str) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{context} embedding must be a non-empty 1D vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{context} embedding must contain finite values")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise RuntimeError(f"{context} embedding has zero length")
    return np.ascontiguousarray(result / norm, dtype=np.float32)


class TrackletEmbeddingStore:
    """Sample crop embeddings and maintain one rolling average per local track."""

    def __init__(
        self,
        encoder: CropEncoder,
        config: TrackletEmbeddingConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.config = config or TrackletEmbeddingConfig()
        self._states: dict[TrackKey, _TrackletState] = {}
        self._embedding_dimension: int | None = None

    def __len__(self) -> int:
        return len(self._states)

    @property
    def embedding_dimension(self) -> int | None:
        """Return the learned dimension after the first encoder call."""
        return self._embedding_dimension

    def update(
        self,
        crops: Sequence[PersonCrop],
        *,
        current_frame_id: int | None = None,
    ) -> tuple[TrackletAppearance, ...]:
        """Update visible tracklets, encoding all crops currently due as one batch."""
        visible = tuple(crops)
        if current_frame_id is None and visible:
            current_frame_id = max(crop.frame_id for crop in visible)
        if current_frame_id is not None:
            if current_frame_id < 0:
                raise ValueError("current_frame_id must not be negative")
            self.prune(current_frame_id)

        crop_by_key: dict[TrackKey, PersonCrop] = {}
        due: list[tuple[TrackKey, PersonCrop]] = []
        for crop in visible:
            key = TrackKey(crop.track.camera_id, crop.track.local_id)
            if key in crop_by_key:
                raise ValueError(f"Duplicate crop for track {key} in one update")
            crop_by_key[key] = crop
            state = self._states.get(key)
            if state is None or (
                crop.frame_id - state.last_sample_frame_id
                >= self.config.embedding_interval_frames
            ):
                due.append((key, crop))

        normalized_embeddings: list[np.ndarray] = []
        if due:
            encoded = np.asarray(
                self.encoder.encode([crop.image for _, crop in due]),
                dtype=np.float32,
            )
            if encoded.ndim != 2 or encoded.shape[0] != len(due) or encoded.shape[1] == 0:
                raise ValueError(
                    "Encoder embedding output must have shape "
                    f"({len(due)}, D); received {encoded.shape}"
                )
            if self._embedding_dimension is None:
                self._embedding_dimension = int(encoded.shape[1])
            elif encoded.shape[1] != self._embedding_dimension:
                raise ValueError(
                    "Encoder embedding dimension changed from "
                    f"{self._embedding_dimension} to {encoded.shape[1]}"
                )
            normalized_embeddings = [
                _unit_vector(vector, context="Encoder") for vector in encoded
            ]

        for (key, crop), embedding in zip(due, normalized_embeddings, strict=True):
            state = self._states.get(key)
            if state is None:
                state = _TrackletState(
                    history=deque(maxlen=self.config.history_size),
                    total_samples=0,
                    first_seen_frame_id=crop.frame_id,
                    last_seen_frame_id=crop.frame_id,
                    last_sample_frame_id=crop.frame_id,
                    first_seen_timestamp=crop.timestamp,
                    last_seen_timestamp=crop.timestamp,
                )
                self._states[key] = state
            state.history.append(embedding)
            state.total_samples += 1
            state.last_sample_frame_id = crop.frame_id

        for key, crop in crop_by_key.items():
            state = self._states.get(key)
            if state is not None:
                state.last_seen_frame_id = crop.frame_id
                state.last_seen_timestamp = crop.timestamp

        return tuple(
            appearance
            for key in crop_by_key
            if (appearance := self.get(key)) is not None
        )

    def get(self, key: TrackKey) -> TrackletAppearance | None:
        """Return a copy of one tracklet's current rolling representation."""
        state = self._states.get(key)
        if state is None:
            return None
        mean = np.mean(np.stack(state.history), axis=0)
        embedding = _unit_vector(mean, context="Tracklet average")
        return TrackletAppearance(
            key=key,
            embedding=embedding,
            stored_samples=len(state.history),
            total_samples=state.total_samples,
            first_seen_frame_id=state.first_seen_frame_id,
            last_seen_frame_id=state.last_seen_frame_id,
            first_seen_timestamp=state.first_seen_timestamp,
            last_seen_timestamp=state.last_seen_timestamp,
        )

    def all(self) -> tuple[TrackletAppearance, ...]:
        """Return all retained tracklets in deterministic key order."""
        appearances: list[TrackletAppearance] = []
        for key in sorted(self._states):
            appearance = self.get(key)
            if appearance is not None:
                appearances.append(appearance)
        return tuple(appearances)

    def history(self, key: TrackKey) -> np.ndarray:
        """Return a defensive copy of one tracklet's normalized samples."""
        state = self._states.get(key)
        if state is None:
            dimension = self._embedding_dimension or 0
            return np.empty((0, dimension), dtype=np.float32)
        return np.stack(state.history).astype(np.float32, copy=True)

    def histories(self) -> dict[TrackKey, np.ndarray]:
        """Return defensive copies of all retained sample histories."""
        return {key: self.history(key) for key in sorted(self._states)}

    def prune(self, current_frame_id: int) -> tuple[TrackKey, ...]:
        """Forget tracks not observed within the configured frame TTL."""
        if current_frame_id < 0:
            raise ValueError("current_frame_id must not be negative")
        removed = tuple(
            key
            for key, state in self._states.items()
            if current_frame_id - state.last_seen_frame_id > self.config.stale_after_frames
        )
        for key in removed:
            del self._states[key]
        return removed


def evaluate_tracklet_separation(
    histories: Mapping[TrackKey, np.ndarray],
) -> TrackletSeparationDiagnostic:
    """Compare within-track cosine similarity with between-track similarity."""
    normalized: dict[TrackKey, np.ndarray] = {}
    for key, history in histories.items():
        values = np.asarray(history, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError(f"History for {key} must have shape (N, D) with N > 0")
        normalized[key] = np.stack(
            [_unit_vector(row, context=f"History for {key}") for row in values]
        )

    same_scores = [
        float(left @ right)
        for history in normalized.values()
        for left, right in combinations(history, 2)
    ]
    different_scores = [
        float(left @ right)
        for (_, left_history), (_, right_history) in combinations(normalized.items(), 2)
        for left in left_history
        for right in right_history
    ]
    if not same_scores:
        raise ValueError("Diagnostic needs at least one track with two embedding samples")
    if not different_scores:
        raise ValueError("Diagnostic needs at least two different local tracks")

    same_mean = float(np.mean(same_scores))
    different_mean = float(np.mean(different_scores))
    margin = same_mean - different_mean
    return TrackletSeparationDiagnostic(
        same_track_mean=same_mean,
        different_track_mean=different_mean,
        margin=margin,
        same_track_pairs=len(same_scores),
        different_track_pairs=len(different_scores),
        passed=margin > 0.0,
    )
