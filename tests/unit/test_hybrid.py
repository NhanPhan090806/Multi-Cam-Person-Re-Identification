from dataclasses import replace

import numpy as np
import pytest

from multicam_reid.association.handoff import HandoffConfig, HandoffIdentityRegistry
from multicam_reid.association.topology import parse_overlap_pairs
from multicam_reid.reid.tracklets import TrackKey, TrackletAppearance


def appearance(camera, local_id=1, vector=(1.0, 0.0), samples=2):
    return TrackletAppearance(
        TrackKey(camera, local_id),
        np.asarray(vector, dtype=np.float32),
        samples,
        samples,
        0,
        samples,
        0.0,
        float(samples),
    )


def hybrid(pairs=(("A", "B"),), **kwargs):
    return HandoffIdentityRegistry(HandoffConfig(overlap_pairs=pairs, **kwargs))


def settle(registry, views, start=0):
    for step in range(3):
        result = registry.update(
            [replace(view, total_samples=view.total_samples + step) for view in views],
            frame_id=start + step,
            timestamp=float(start + step),
        )
    return result


def test_declared_overlap_joins_after_fresh_confirmations_and_keeps_both_bindings():
    registry = hybrid()
    first = registry.update([appearance("A"), appearance("B")], frame_id=0, timestamp=0)
    assert first.assignments == {TrackKey("A", 1): 1}
    # Replaying cached views must not count as new evidence.
    cached = registry.update([appearance("A"), appearance("B")], frame_id=1, timestamp=0)
    assert TrackKey("B", 1) not in cached.assignments
    joined = settle(registry, [appearance("A"), appearance("B")], start=2)
    assert joined.assignments == {TrackKey("A", 1): 1, TrackKey("B", 1): 1}
    assert registry.total_ids_issued == 1
    assert not joined.handoff_events
    snapshot = registry.snapshots(active_keys=joined.assignments)[0]
    assert snapshot.active_members == (TrackKey("A", 1), TrackKey("B", 1))


def test_overlap_is_bidirectional_and_unlisted_pairs_stay_strict():
    registry = hybrid((("B", "A"),))
    joined = settle(registry, [appearance("A"), appearance("B"), appearance("C")])
    assert joined.assignments[TrackKey("A", 1)] == joined.assignments[TrackKey("B", 1)]
    assert joined.assignments[TrackKey("C", 1)] != joined.assignments[TrackKey("A", 1)]


def test_pair_permissions_do_not_spread_transitively():
    registry = hybrid((("A", "B"), ("B", "C")), overlap_confirmations=1)
    registry.update([appearance("A"), appearance("B")], frame_id=0, timestamp=0)
    joined = registry.update(
        [appearance("A"), appearance("B"), appearance("C")],
        frame_id=1,
        timestamp=1,
    )
    assert joined.assignments[TrackKey("A", 1)] == joined.assignments[TrackKey("B", 1)]
    assert joined.assignments[TrackKey("C", 1)] != joined.assignments[TrackKey("A", 1)]


def test_two_people_in_one_camera_never_share_an_id_even_if_embeddings_identical():
    registry = hybrid(overlap_confirmations=1, match_margin=0)
    result = settle(registry, [appearance("A"), appearance("A", 2), appearance("B")])
    assert result.assignments[TrackKey("A", 1)] != result.assignments[TrackKey("A", 2)]
    assert len(set(result.assignments.values())) == 2


def test_two_similar_arrivals_competing_for_one_id_both_remain_pending():
    registry = hybrid(overlap_confirmations=1)
    registry.update([appearance("A")], frame_id=0, timestamp=0)
    result = settle(registry, [appearance("A"), appearance("B"), appearance("B", 2)], start=1)
    assert result.assignments == {TrackKey("A", 1): 1}


def test_one_arrival_with_two_similar_candidates_remains_pending():
    registry = hybrid(overlap_confirmations=1)
    result = settle(registry, [appearance("A"), appearance("A", 2), appearance("B")])
    assert TrackKey("B", 1) not in result.assignments


def test_stricter_overlap_threshold_does_not_replace_handoff_threshold():
    registry = hybrid(overlap_confirmations=1, overlap_max_cosine_distance=0.1)
    # Cosine distance about .2: acceptable for handoff, not active overlap.
    result = registry.update(
        [appearance("A"), appearance("B", vector=(0.8, 0.6))],
        frame_id=0,
        timestamp=0,
    )
    assert result.assignments[TrackKey("A", 1)] != result.assignments[TrackKey("B", 1)]


def test_group_can_handoff_only_after_every_member_has_departed():
    registry = hybrid()
    settle(registry, [appearance("A"), appearance("B")])
    blocked = registry.update(
        [appearance("C")],
        frame_id=3,
        timestamp=3,
        visible_keys=[TrackKey("B", 1), TrackKey("C", 1)],
    )
    assert blocked.assignments[TrackKey("C", 1)] != 1
    registry.update([], frame_id=4, timestamp=4)
    recovered = registry.update([appearance("D")], frame_id=5, timestamp=6)
    # Two identical inactive identities make the later arrival ambiguous too.
    assert TrackKey("D", 1) not in recovered.assignments


def test_overlap_group_survives_one_departure_then_handoffs_after_blind_gap():
    registry = hybrid()
    settle(registry, [appearance("A"), appearance("B")])
    only_a = registry.update([appearance("A", samples=6)], frame_id=3, timestamp=3)
    assert only_a.assignments == {TrackKey("A", 1): 1}
    registry.update([], frame_id=4, timestamp=4)
    recovered = registry.update([appearance("C")], frame_id=5, timestamp=8)
    assert recovered.assignments == {TrackKey("C", 1): 1}
    assert len(recovered.handoff_events) == 1
    assert recovered.handoff_events[0].from_camera == "A"
    assert not recovered.overlap_events
    returned = registry.update(
        [appearance("A"), appearance("C")],
        frame_id=6,
        timestamp=9,
    )
    assert returned.assignments[TrackKey("A", 1)] != returned.assignments[TrackKey("C", 1)]


def test_visible_member_without_crop_still_blocks_an_unlisted_camera():
    registry = hybrid(gallery_ttl_seconds=2)
    settle(registry, [appearance("A"), appearance("B")])
    result = registry.update(
        [appearance("C")],
        frame_id=3,
        timestamp=10,
        visible_keys=[TrackKey("B", 1)],
    )
    assert result.assignments[TrackKey("B", 1)] == 1
    assert result.assignments[TrackKey("C", 1)] != 1


def test_overlap_events_are_distinct_from_handoffs():
    registry = hybrid(overlap_confirmations=1)
    result = registry.update([appearance("A"), appearance("B")], frame_id=0, timestamp=0)
    assert len(result.overlap_events) == 1
    assert result.overlap_events[0].to_dict()["event"] == "overlap"
    assert not result.handoff_events


@pytest.mark.parametrize(
    "settings",
    [
        {"overlap_pairs": (("A", "A"),)},
        {"overlap_pairs": (("", "B"),)},
        {"overlap_max_cosine_distance": -1},
        {"overlap_max_cosine_distance": 3},
        {"overlap_confirmations": 0},
    ],
)
def test_invalid_hybrid_settings(settings):
    with pytest.raises(ValueError):
        HandoffConfig(**settings)


def test_cli_overlap_pairs_are_normalized_deduplicated_and_validated():
    assert parse_overlap_pairs(["2,3", "3,2", "4,5"], ("C2", "C3", "C4", "C5"), numeric=True) == (
        ("C2", "C3"),
        ("C4", "C5"),
    )
    assert parse_overlap_pairs(["PHONE,LAPTOP"], ("PHONE", "LAPTOP")) == (("LAPTOP", "PHONE"),)


@pytest.mark.parametrize("value", ["2", "2,2", "2,9", "2,3,4", ",3"])
def test_cli_rejects_bad_or_unselected_pairs(value):
    with pytest.raises(ValueError):
        parse_overlap_pairs([value], ("C2", "C3"), numeric=True)


def test_complete_three_camera_overlap_requires_every_pair():
    registry = hybrid((("A", "B"), ("A", "C"), ("B", "C")), overlap_confirmations=1)
    result = registry.update(
        [appearance("A"), appearance("B"), appearance("C")],
        frame_id=0,
        timestamp=0,
    )
    assert set(result.assignments.values()) == {1}
    assert len(result.overlap_events) == 2


def test_confirmations_reset_when_evidence_disagrees_then_recovers():
    registry = hybrid()
    registry.update([appearance("A"), appearance("A", 2, (0, 1))], frame_id=0, timestamp=0)
    registry.update(
        [appearance("A"), appearance("A", 2, (0, 1)), appearance("B")],
        frame_id=1,
        timestamp=1,
    )
    # Switch candidate: old A=1 confirmation must not count towards A=2.
    for frame in (2, 3):
        result = registry.update(
            [
                appearance("A"),
                appearance("A", 2, (0, 1)),
                appearance("B", vector=(0, 1), samples=frame + 2),
            ],
            frame_id=frame,
            timestamp=frame,
        )
        assert TrackKey("B", 1) not in result.assignments
    result = registry.update(
        [
            appearance("A"),
            appearance("A", 2, (0, 1)),
            appearance("B", vector=(0, 1), samples=6),
        ],
        frame_id=4,
        timestamp=4,
    )
    assert result.assignments[TrackKey("B", 1)] == 2


def test_departed_group_can_handoff_then_join_a_new_overlapping_camera_same_update():
    registry = hybrid((("A", "B"), ("C", "D")), overlap_confirmations=1)
    registry.update([appearance("A"), appearance("B")], frame_id=0, timestamp=0)
    registry.update([], frame_id=1, timestamp=1)
    result = registry.update([appearance("C"), appearance("D")], frame_id=2, timestamp=5)
    assert result.assignments == {TrackKey("C", 1): 1, TrackKey("D", 1): 1}
    assert len(result.handoff_events) == len(result.overlap_events) == 1


def test_expired_group_clears_all_bindings():
    registry = hybrid(gallery_ttl_seconds=2, overlap_confirmations=1)
    registry.update([appearance("A"), appearance("B")], frame_id=0, timestamp=0)
    registry.update([], frame_id=1, timestamp=1)
    registry.update([], frame_id=2, timestamp=4)
    result = registry.update([appearance("B")], frame_id=3, timestamp=5)
    assert result.assignments[TrackKey("B", 1)] == 2


def test_overlap_candidate_must_agree_with_every_active_view_template():
    registry = hybrid(
        (("A", "B"), ("A", "C"), ("B", "C")),
        overlap_confirmations=1,
        overlap_max_cosine_distance=0.25,
    )
    registry.update([appearance("A"), appearance("B")], frame_id=0, timestamp=0)
    # Active B drifts: matching A alone must not be enough for C to join the group.
    result = registry.update(
        [
            appearance("A"),
            appearance("B", vector=(0, 1)),
            appearance("C"),
        ],
        frame_id=1,
        timestamp=1,
    )
    assert result.assignments[TrackKey("C", 1)] != result.assignments[TrackKey("A", 1)]
