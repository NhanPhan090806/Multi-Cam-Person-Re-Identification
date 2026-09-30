from __future__ import annotations

import numpy as np
import pytest

from multicam_reid.association.global_registry import (
    AssociationConfig,
    GlobalIdentityRegistry,
    cosine_distance_matrix,
    match_tracklets,
)
from multicam_reid.reid.tracklets import TrackKey, TrackletAppearance


def appearance(
    camera_id: str,
    local_id: int,
    vector: tuple[float, ...],
    *,
    samples: int = 2,
    frame_id: int = 10,
) -> TrackletAppearance:
    embedding = np.asarray(vector, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)
    return TrackletAppearance(
        key=TrackKey(camera_id, local_id),
        embedding=embedding,
        stored_samples=samples,
        total_samples=samples,
        first_seen_frame_id=0,
        last_seen_frame_id=frame_id,
        first_seen_timestamp=0.0,
        last_seen_timestamp=frame_id / 25.0,
    )


def test_cosine_distance_matrix_has_zero_for_same_direction() -> None:
    left = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    right = np.asarray([[2.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    distances = cosine_distance_matrix(left, right)

    assert distances.shape == (2, 2)
    assert distances[0, 0] == pytest.approx(0.0)
    assert distances[0, 1] == pytest.approx(1.0 - 1.0 / np.sqrt(2.0))
    assert distances[1, 0] == pytest.approx(1.0)


def test_hungarian_matching_is_one_to_one_and_thresholded() -> None:
    left = (
        appearance("C0", 1, (1.0, 0.0)),
        appearance("C0", 2, (0.0, 1.0)),
    )
    right = (
        appearance("C1", 7, (0.99, 0.01)),
        appearance("C1", 8, (0.1, 0.9)),
    )

    matrix, matches = match_tracklets(left, right, max_cosine_distance=0.1)

    assert matrix.left_keys == (TrackKey("C0", 1), TrackKey("C0", 2))
    assert matrix.right_keys == (TrackKey("C1", 7), TrackKey("C1", 8))
    assert [(match.left, match.right) for match in matches] == [
        (TrackKey("C0", 1), TrackKey("C1", 7)),
        (TrackKey("C0", 2), TrackKey("C1", 8)),
    ]
    assert all(match.cosine_distance <= 0.1 for match in matches)


def test_registry_merges_matching_cross_camera_tracks_into_one_global_id() -> None:
    registry = GlobalIdentityRegistry(AssociationConfig(max_cosine_distance=0.2))

    result = registry.update(
        (
            appearance("C0", 1, (1.0, 0.0)),
            appearance("C1", 4, (0.99, 0.01)),
        ),
        frame_id=10,
        timestamp=0.4,
    )

    assert result.assignments[TrackKey("C0", 1)] == result.assignments[TrackKey("C1", 4)]
    assert len(result.merge_events) == 1
    assert result.active_global_ids == (1,)
    snapshot = registry.snapshot(1, active_keys=result.assignments)
    assert {member.camera_id for member in snapshot.members} == {"C0", "C1"}


def test_registry_keeps_dissimilar_people_as_separate_global_ids() -> None:
    registry = GlobalIdentityRegistry(AssociationConfig(max_cosine_distance=0.2))

    result = registry.update(
        (
            appearance("C0", 1, (1.0, 0.0)),
            appearance("C1", 4, (0.0, 1.0)),
        ),
        frame_id=10,
        timestamp=0.4,
    )

    assert result.assignments[TrackKey("C0", 1)] != result.assignments[TrackKey("C1", 4)]
    assert result.merge_events == ()
    assert result.active_global_ids == (1, 2)


def test_registry_waits_for_minimum_samples_then_merges_provisional_ids() -> None:
    registry = GlobalIdentityRegistry(
        AssociationConfig(max_cosine_distance=0.2, min_stored_samples=2)
    )

    first = registry.update(
        (
            appearance("C0", 1, (1.0, 0.0), samples=1, frame_id=0),
            appearance("C1", 4, (1.0, 0.0), samples=1, frame_id=0),
        ),
        frame_id=0,
        timestamp=0.0,
    )
    second = registry.update(
        (
            appearance("C0", 1, (1.0, 0.0), samples=2, frame_id=5),
            appearance("C1", 4, (1.0, 0.0), samples=2, frame_id=5),
        ),
        frame_id=5,
        timestamp=0.2,
    )

    assert len(set(first.assignments.values())) == 2
    assert len(set(second.assignments.values())) == 1
    assert second.merge_events[0].frame_id == 5


def test_registry_never_merges_two_active_tracks_from_the_same_camera() -> None:
    registry = GlobalIdentityRegistry(AssociationConfig(max_cosine_distance=0.2))
    registry.update(
        (
            appearance("C0", 1, (1.0, 0.0)),
            appearance("C0", 2, (1.0, 0.0)),
        ),
        frame_id=10,
        timestamp=0.4,
    )

    result = registry.update(
        (
            appearance("C0", 1, (1.0, 0.0), frame_id=11),
            appearance("C0", 2, (1.0, 0.0), frame_id=11),
            appearance("C1", 8, (1.0, 0.0), frame_id=11),
        ),
        frame_id=11,
        timestamp=0.44,
    )

    c0_globals = {
        result.assignments[TrackKey("C0", 1)],
        result.assignments[TrackKey("C0", 2)],
    }
    assert len(c0_globals) == 2


def test_new_track_can_rejoin_recent_inactive_global_identity() -> None:
    registry = GlobalIdentityRegistry(
        AssociationConfig(max_cosine_distance=0.2, max_idle_frames=20)
    )
    first = registry.update(
        (appearance("C0", 1, (1.0, 0.0), frame_id=0),),
        frame_id=0,
        timestamp=0.0,
    )

    returned = registry.update(
        (appearance("C1", 9, (0.99, 0.01), frame_id=10),),
        frame_id=10,
        timestamp=0.4,
    )

    assert returned.assignments[TrackKey("C1", 9)] == first.assignments[TrackKey("C0", 1)]


def test_expired_global_identity_is_not_reused() -> None:
    registry = GlobalIdentityRegistry(
        AssociationConfig(max_cosine_distance=0.2, max_idle_frames=5)
    )
    first = registry.update(
        (appearance("C0", 1, (1.0, 0.0), frame_id=0),),
        frame_id=0,
        timestamp=0.0,
    )

    returned = registry.update(
        (appearance("C1", 9, (1.0, 0.0), frame_id=10),),
        frame_id=10,
        timestamp=0.4,
    )

    assert returned.assignments[TrackKey("C1", 9)] != first.assignments[TrackKey("C0", 1)]


def test_stale_local_mapping_expires_while_same_global_is_active_elsewhere() -> None:
    registry = GlobalIdentityRegistry(
        AssociationConfig(max_cosine_distance=0.2, max_idle_frames=5)
    )
    first = registry.update(
        (
            appearance("C0", 1, (1.0, 0.0), frame_id=0),
            appearance("C1", 2, (1.0, 0.0), frame_id=0),
        ),
        frame_id=0,
        timestamp=0.0,
    )
    original_global = first.assignments[TrackKey("C1", 2)]
    for frame_id in range(1, 11):
        registry.update(
            (appearance("C1", 2, (1.0, 0.0), frame_id=frame_id),),
            frame_id=frame_id,
            timestamp=frame_id / 25.0,
        )

    reused = registry.update(
        (
            appearance("C0", 1, (0.0, 1.0), frame_id=11),
            appearance("C1", 2, (1.0, 0.0), frame_id=11),
        ),
        frame_id=11,
        timestamp=0.44,
    )

    assert reused.assignments[TrackKey("C1", 2)] == original_global
    assert reused.assignments[TrackKey("C0", 1)] != original_global


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_cosine_distance": -0.1},
        {"max_cosine_distance": 2.1},
        {"min_stored_samples": 0},
        {"max_idle_frames": -1},
    ],
)
def test_association_config_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AssociationConfig(**kwargs)
