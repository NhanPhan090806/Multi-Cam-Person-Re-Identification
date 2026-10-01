from __future__ import annotations

import numpy as np
import pytest

from multicam_reid.association.handoff import HandoffConfig, HandoffIdentityRegistry
from multicam_reid.reid.tracklets import TrackKey, TrackletAppearance


def appearance(camera: str, local_id: int = 1, vector=(1.0, 0.0), samples=2):
    return TrackletAppearance(
        TrackKey(camera, local_id),
        np.asarray(vector, dtype=np.float32),
        samples,
        samples,
        0,
        0,
        0.0,
        0.0,
    )


def registry(**settings):
    return HandoffIdentityRegistry(HandoffConfig(exit_grace_seconds=1.0, **settings))


def test_a_blind_interval_b_preserves_global_id_and_records_handoff():
    identities = registry()
    first = identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=1, timestamp=1.0)
    later = identities.update([appearance("B", 9)], frame_id=2, timestamp=10.0)
    assert later.assignments[TrackKey("B", 9)] == first.assignments[TrackKey("A", 1)]
    assert len(later.handoff_events) == 1
    event = later.handoff_events[0].to_dict()
    assert event["from_camera"] == "A" and event["to_camera"] == "B"
    assert event["gap_seconds"] == 10.0


def test_new_track_waits_for_samples_then_reconnects_instead_of_getting_stuck():
    identities = registry()
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=1, timestamp=1.0)
    pending = identities.update([appearance("B", samples=1)], frame_id=2, timestamp=5.0)
    assert TrackKey("B", 1) not in pending.assignments
    mature = identities.update([appearance("B")], frame_id=3, timestamp=6.0)
    assert mature.assignments[TrackKey("B", 1)] == 1
    assert identities.total_ids_issued == 1


def test_simultaneously_visible_lookalikes_are_not_joined():
    identities = registry()
    result = identities.update([appearance("A"), appearance("B")], frame_id=0, timestamp=0.0)
    assert result.assignments[TrackKey("A", 1)] != result.assignments[TrackKey("B", 1)]
    assert not result.handoff_events


def test_different_person_gets_new_id_after_departure():
    identities = registry()
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=1, timestamp=1.0)
    result = identities.update([appearance("B", vector=(0.0, 1.0))], frame_id=2, timestamp=5.0)
    assert result.assignments[TrackKey("B", 1)] == 2


def test_retention_uses_seconds_instead_of_number_of_updates():
    identities = registry(gallery_ttl_seconds=60.0)
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=10000, timestamp=1.0)
    result = identities.update([appearance("B")], frame_id=20000, timestamp=30.0)
    assert result.assignments[TrackKey("B", 1)] == 1


def test_expired_identity_is_removed_and_ids_are_never_recycled():
    identities = registry(gallery_ttl_seconds=5.0)
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=1, timestamp=1.0)
    result = identities.update([appearance("B")], frame_id=2, timestamp=6.0)
    assert result.assignments[TrackKey("B", 1)] == 2
    assert len(identities) == 1


def test_active_identity_survives_without_a_valid_crop_and_blocks_handoff():
    identities = registry(gallery_ttl_seconds=5.0)
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    result = identities.update(
        [appearance("B")],
        frame_id=1,
        timestamp=10.0,
        visible_keys=[TrackKey("A", 1), TrackKey("B", 1)],
    )
    assert result.assignments[TrackKey("A", 1)] == 1
    assert result.assignments[TrackKey("B", 1)] == 2


def test_ambiguous_appearance_stays_pending_until_candidates_separate():
    identities = registry(match_margin=0.05)
    identities.update(
        [appearance("A"), appearance("A", 2, (0.999, 0.01))], frame_id=0, timestamp=0.0
    )
    identities.update([], frame_id=1, timestamp=1.0)
    result = identities.update([appearance("B")], frame_id=2, timestamp=5.0)
    assert TrackKey("B", 1) not in result.assignments
    assert identities.total_ids_issued == 2


def test_two_arrivals_cannot_claim_one_departed_identity():
    identities = registry(match_margin=0.0)
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=1, timestamp=1.0)
    result = identities.update([appearance("B"), appearance("C")], frame_id=2, timestamp=5.0)
    assert len(set(result.assignments.values())) == 2
    assert len(result.handoff_events) == 1


def test_local_track_change_in_same_camera_can_recover_an_inactive_id():
    identities = registry()
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=1, timestamp=1.0)
    result = identities.update([appearance("A", 7)], frame_id=2, timestamp=5.0)
    assert result.assignments[TrackKey("A", 7)] == 1
    assert result.handoff_events[0].from_camera == "A"


def test_topology_and_minimum_travel_time_are_enforced():
    identities = registry(min_travel_seconds=3.0, allowed_transitions=(("A", "B"),))
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=1, timestamp=0.5)
    early = identities.update([appearance("B")], frame_id=2, timestamp=2.0)
    assert TrackKey("B", 1) not in early.assignments
    identities.update([], frame_id=3, timestamp=3.0)
    disallowed = identities.update([appearance("C")], frame_id=4, timestamp=10.0)
    assert disallowed.assignments[TrackKey("C", 1)] == 2


def test_arrival_during_departure_grace_waits_and_can_match_later():
    identities = registry()
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=1, timestamp=1.0)
    early = identities.update([appearance("B")], frame_id=2, timestamp=1.1)
    assert TrackKey("B", 1) not in early.assignments
    later = identities.update([appearance("B")], frame_id=3, timestamp=2.1)
    assert later.assignments[TrackKey("B", 1)] == 1


def test_snapshot_tracks_departed_and_current_members():
    identities = registry()
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=1, timestamp=1.0)
    result = identities.update([appearance("B")], frame_id=2, timestamp=5.0)
    snapshot = identities.snapshots(active_keys=result.assignments)[0]
    assert snapshot.members == (TrackKey("A", 1), TrackKey("B", 1))
    assert snapshot.active_members == (TrackKey("B", 1),)


@pytest.mark.parametrize(
    "settings",
    [
        {"gallery_ttl_seconds": 0},
        {"exit_grace_seconds": -1},
        {"min_travel_seconds": -1},
        {"match_margin": -0.1},
        {"max_cosine_distance": 3},
        {"min_stored_samples": 0},
        {"gallery_ttl_seconds": float("nan")},
        {"allowed_transitions": (("", "B"),)},
    ],
)
def test_invalid_config_is_rejected(settings):
    with pytest.raises(ValueError):
        HandoffConfig(**settings)


def test_time_cannot_go_backwards_or_be_nonfinite():
    identities = registry()
    identities.update([], frame_id=0, timestamp=5.0)
    with pytest.raises(ValueError):
        identities.update([], frame_id=1, timestamp=4.0)
    with pytest.raises(ValueError):
        identities.update([], frame_id=1, timestamp=float("nan"))


def test_duplicate_keys_and_invalid_embeddings_are_rejected():
    identities = registry()
    with pytest.raises(ValueError):
        identities.update([appearance("A"), appearance("A")], frame_id=0, timestamp=0.0)
    with pytest.raises(ValueError):
        identities.update([appearance("A", vector=(0.0, 0.0))], frame_id=0, timestamp=0.0)


def test_return_trip_does_not_leave_a_stale_mapping_in_previous_camera():
    identities = registry()
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=1, timestamp=1.0)
    identities.update([appearance("B")], frame_id=2, timestamp=5.0)
    identities.update([], frame_id=3, timestamp=6.0)
    result = identities.update([appearance("A", 5)], frame_id=4, timestamp=10.0)
    assert result.assignments[TrackKey("A", 5)] == 1
    assert result.handoff_events[0].from_camera == "B"


def test_negative_time_changed_dimension_and_bad_sample_counts_are_rejected():
    from dataclasses import replace

    identities = registry()
    with pytest.raises(ValueError):
        identities.update([], frame_id=0, timestamp=-1.0)
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    with pytest.raises(ValueError):
        identities.update([appearance("B", vector=(1.0, 0.0, 0.0))], frame_id=1, timestamp=1.0)
    with pytest.raises(ValueError):
        identities.update([replace(appearance("B"), total_samples=1)], frame_id=1, timestamp=1.0)


def test_known_visible_track_does_not_replace_template_with_an_immature_sample():
    identities = registry()
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([appearance("A", vector=(0.0, 1.0), samples=1)], frame_id=1, timestamp=1.0)
    identities.update([], frame_id=2, timestamp=2.0)
    result = identities.update([appearance("B")], frame_id=3, timestamp=5.0)
    assert result.assignments[TrackKey("B", 1)] == 1


def test_different_person_is_not_blocked_by_another_persons_departure_grace():
    identities = registry()
    identities.update([appearance("A")], frame_id=0, timestamp=0.0)
    identities.update([], frame_id=1, timestamp=1.0)
    result = identities.update(
        [appearance("B", vector=(0.0, 1.0))], frame_id=2, timestamp=1.1,
    )
    assert result.assignments[TrackKey("B", 1)] == 2
