import numpy as np
import pytest

from multicam_reid.association.handoff import HandoffConfig, HandoffIdentityRegistry
from multicam_reid.reid.tracklets import TrackKey, TrackletAppearance


def view(camera, local=1, vector=(1, 0), samples=2):
    return TrackletAppearance(
        TrackKey(camera, local),
        np.asarray(vector, np.float32),
        samples,
        samples,
        0,
        samples,
        0,
        float(samples),
    )


def registry(pairs=(("A", "B"),), **settings):
    return HandoffIdentityRegistry(HandoffConfig(overlap_pairs=pairs, **settings))


def split(registry):
    result = registry.update([view("A"), view("B", vector=(0, 1))], frame_id=0, timestamp=0)
    assert set(result.assignments.values()) == {1, 2}


def converge(registry, cameras=("A", "B"), start=1, steps=7):
    for step in range(start, start + steps):
        result = registry.update(
            [view(c, samples=step + 2) for c in cameras], frame_id=step, timestamp=float(step)
        )
    return result


def test_existing_ids_are_reconciled_after_fresh_mutual_evidence():
    ids = registry()
    split(ids)
    for step in range(1, 5):
        result = ids.update(
            [view("A", samples=step + 2), view("B", samples=step + 2)],
            frame_id=step,
            timestamp=float(step),
        )
        assert set(result.assignments.values()) == {1, 2}
    result = ids.update([view("A", samples=7), view("B", samples=7)], frame_id=5, timestamp=5)
    assert set(result.assignments.values()) == {1}
    assert ids.total_ids_issued == 2 and len(ids) == 1
    event = result.merge_events[0].to_dict()
    assert event["survivor_global_id"] == 1 and event["merged_global_id"] == 2
    assert ids.identity_aliases == {2: 1}
    assert not result.handoff_events and not result.overlap_events


def test_repeated_cached_or_one_sided_samples_cannot_confirm_a_merge():
    ids = registry()
    split(ids)
    for step in range(1, 12):
        result = ids.update(
            [view("A", samples=step + 2), view("B")], frame_id=step, timestamp=float(step)
        )
        assert not result.merge_events
    assert len(ids) == 2


@pytest.mark.parametrize("enabled,pairs", [(False, (("A", "B"),)), (True, (("A", "C"),))])
def test_disabled_or_unlisted_reconciliation_keeps_ids_separate(enabled, pairs):
    ids = registry(pairs, reconciliation_enabled=enabled)
    split(ids)
    assert set(converge(ids).assignments.values()) == {1, 2}


def test_historical_same_camera_people_cannot_be_merged_later():
    ids = registry()
    ids.update([view("A"), view("A", 2, (0, 1))], frame_id=0, timestamp=0)
    ids.update([view("A")], frame_id=1, timestamp=1)
    ids.update([view("A"), view("B", vector=(0, 1))], frame_id=2, timestamp=3)
    result = converge(ids, start=4)
    assert set(result.assignments.values()) == {1, 2}
    assert any(d.to_dict()["reason"] == "historical_same_camera_conflict" for d in result.decisions)


def test_ambiguous_three_way_lookalikes_are_not_reconciled():
    ids = registry((("A", "B"), ("A", "C"), ("B", "C")))
    ids.update(
        [view("A"), view("B", vector=(0, 1)), view("C", vector=(-1, 0))], frame_id=0, timestamp=0
    )
    result = converge(ids, cameras=("A", "B", "C"))
    assert len(set(result.assignments.values())) == 3
    assert any(d.to_dict()["reason"] == "ambiguous_reconciliation" for d in result.decisions)


def test_bad_evidence_resets_confirmation_streak():
    ids = registry()
    split(ids)
    converge(ids, steps=3)
    ids.update([view("A", samples=6), view("B", vector=(0, 1), samples=6)], frame_id=4, timestamp=4)
    result = converge(ids, start=5, steps=4)
    assert len(set(result.assignments.values())) == 2
    result = converge(ids, start=9, steps=1)
    assert len(set(result.assignments.values())) == 1


def test_blind_handoff_after_reconciliation_keeps_surviving_id():
    ids = registry()
    split(ids)
    converge(ids)
    ids.update([], frame_id=8, timestamp=8)
    result = ids.update([view("C")], frame_id=9, timestamp=12)
    assert result.assignments == {TrackKey("C", 1): 1}
    assert len(result.handoff_events) == 1


def test_minimum_confirmation_duration_and_missing_crop_are_enforced():
    ids = registry(reconciliation_confirmations=2, reconciliation_min_seconds=3)
    split(ids)
    for step in (1, 2):
        result = ids.update(
            [view("A", samples=step + 2), view("B", samples=step + 2)],
            frame_id=step,
            timestamp=step / 10,
        )
        assert not result.merge_events
    missing = ids.update(
        [view("A", samples=5)],
        frame_id=3,
        timestamp=1,
        visible_keys=[TrackKey("A", 1), TrackKey("B", 1)],
    )
    assert any(d.to_dict()["reason"] == "insufficient_appearance" for d in missing.decisions)
    for step in (4, 5):
        result = ids.update(
            [view("A", samples=step + 2), view("B", samples=step + 2)],
            frame_id=step,
            timestamp=float(step),
        )
        assert not result.merge_events
    result = ids.update([view("A", samples=8), view("B", samples=8)], frame_id=7, timestamp=7)
    assert len(result.merge_events) == 1


@pytest.mark.parametrize(
    "settings",
    [
        {"reconciliation_confirmations": 0},
        {"reconciliation_min_seconds": -1},
        {"reconciliation_enabled": "yes"},
    ],
)
def test_invalid_reconciliation_settings(settings):
    with pytest.raises(ValueError):
        HandoffConfig(**settings)


def test_alias_chains_and_departed_bindings_are_cleaned_after_group_merges():
    ids = registry(
        (("A", "B"), ("A", "C"), ("B", "C")),
        reconciliation_confirmations=1,
        reconciliation_min_seconds=0,
    )
    ids.update(
        [view("A", vector=(1, 0, 0)), view("B", vector=(0, 1, 0)), view("C", vector=(0, 0, 1))],
        frame_id=0,
        timestamp=0,
    )
    result = ids.update(
        [
            view("A", vector=(1, 0, 0), samples=3),
            view("B", vector=(0, 1, 0), samples=3),
            view("C", vector=(0, 1, 0), samples=3),
        ],
        frame_id=1,
        timestamp=1,
    )
    assert ids.identity_aliases == {3: 2}
    result = ids.update(
        [view("A", vector=(1, 0, 0), samples=4), view("B", vector=(1, 0, 0), samples=4)],
        frame_id=2,
        timestamp=2,
    )
    assert ids.identity_aliases == {3: 1, 2: 1}
    assert set(result.assignments.values()) == {1}
    ids.update([], frame_id=3, timestamp=3)
    returned = ids.update([view("D", vector=(1, 0, 0))], frame_id=4, timestamp=6)
    assert returned.assignments == {TrackKey("D", 1): 1}


def test_historical_conflicts_follow_the_survivor_of_another_merge():
    ids = registry(
        (("A", "B"), ("A", "C"), ("B", "C")),
        reconciliation_confirmations=1,
        reconciliation_min_seconds=0,
    )
    ids.update(
        [view("A"), view("B", vector=(0, 1)), view("B", 2, (-1, 0))], frame_id=0, timestamp=0
    )
    ids.update(
        [view("A", samples=3), view("B", samples=3), view("B", 2, (-1, 0), 3)],
        frame_id=1,
        timestamp=1,
    )
    assert ids.identity_aliases == {2: 1}
    ids.update([view("A", samples=4)], frame_id=2, timestamp=2)
    recovered = ids.update(
        [view("A", samples=5), view("C", vector=(-1, 0))], frame_id=3, timestamp=4
    )
    assert recovered.assignments[TrackKey("C", 1)] == 3
    result = converge(ids, cameras=("A", "C"), start=5)
    assert set(result.assignments.values()) == {1, 3}


def test_arrival_route_rejections_are_logged():
    ids = registry(allowed_transitions=(("A", "B"),))
    ids.update([view("A")], frame_id=0, timestamp=0)
    ids.update([], frame_id=1, timestamp=1)
    result = ids.update([view("C")], frame_id=2, timestamp=5)
    assert any(decision.reason == "transition_not_allowed" for decision in result.decisions)


def test_new_same_camera_track_cannot_claim_an_active_id_and_reason_is_logged():
    ids = registry()
    ids.update([view("A")], frame_id=0, timestamp=0)
    result = ids.update([view("A"), view("A", 2)], frame_id=1, timestamp=1)
    assert result.assignments[TrackKey("A", 1)] != result.assignments[TrackKey("A", 2)]
    assert any(decision.reason == "same_camera_conflict" for decision in result.decisions)


def test_cached_time_does_not_inflate_fresh_evidence_duration():
    ids = registry(reconciliation_confirmations=2, reconciliation_min_seconds=1)
    split(ids)
    ids.update([view("A", samples=3), view("B", samples=3)], frame_id=1, timestamp=0.1)
    ids.update([view("A", samples=4), view("B", samples=4)], frame_id=2, timestamp=0.2)
    cached = ids.update([view("A", samples=4), view("B", samples=4)], frame_id=3, timestamp=2)
    assert not cached.merge_events
    assert any(d.reason == "waiting_reconciliation_duration" for d in cached.decisions)
    fresh = ids.update([view("A", samples=5), view("B", samples=5)], frame_id=4, timestamp=3)
    assert len(fresh.merge_events) == 1
