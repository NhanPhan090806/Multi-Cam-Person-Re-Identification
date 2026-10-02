# Correction to separate-location camera handoffs

This document records the original handoff migration. For current commands and
arguments, see [run.md](../run.md); current progress is in [plan.md](../plan.md).

The old scope selected overlapping cameras. The intended mission is A -> leave ->
unseen travel -> B, with the original anonymous global ID recovered in B.

## Audit and reuse

YOLO, independent ByteTrack, validated crops, trained OSNet, rolling embeddings,
camera workers, black ID panels and screenshots remain appropriate. Their inputs
do not require simultaneous camera visibility. No retraining is needed just to
change association policy.

The former registry mainly matched active camera pairs. Its historical gallery
only considered mature tracks without mappings, but immature tracks immediately
received provisional IDs. Once they matured, gallery recovery did not reconsider
them. This could prevent sequential handoffs. Gallery age was also measured in
updates rather than seconds, and expired identity objects stayed allocated.

The live runner used `registry or ...`, which discarded explicitly supplied empty
registries because they define `__len__`. It now checks `is not None` explicitly.

## Modules and use

| Module | Role |
|---|---|
| `association/handoff.py` | Pure appearance memory and arrival/departure decisions |
| `inputs/handoff_recordings.py` | Record raw frames with timestamps; replay FramePackets |
| `pipeline/camera_demo.py` | Live orchestration; solo/multi-IP default to handoff |
| `pipeline/handoff_replay.py` | Same model pipeline applied to recorded event times |
| `main_handoff_replay.py` | Surface command for offline debugging |
| `scripts/smoke_handoff.py` | Actual models on an artificial A -> unseen -> B schedule |

New tracks wait for enough samples. Known tracks keep their IDs while visible.
Departed identities remain in the gallery for 120 seconds by default. Arrivals
match only inactive identities after the departure grace period, using cosine
distance, ambiguity margin, optional routes/travel-time gates and one-to-one
assignment. Ambiguous arrivals stay pending. Expired identities are removed;
numeric IDs are never recycled during the session. The gallery does not persist
across application restarts.

The old overlap registry/runners remain for reproducing historical results. Select
`--association-mode overlap` explicitly for simultaneous same-scene views.

## Data decisions

Keep Market-1501 and the checkpoint. Keep EPFL as an optional overlap regression
dataset (approximately 149 MiB). WILDTRACK's former download was already removed;
its optional downloader remains. This migration deletes no data or checkpoints.

The migration initially proposed consented phone/laptop recordings from separate
areas. The project now also uses WiseNET Set 2 as its public full-scene development
test. Hardware recordings remain the live demonstration input: record entrances,
exits and background frames across a blind gap.

[PRID2011 publisher](https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/prid11/)
provides single/multi-shot person crops, including 200 shared identities, but its
original videos require contacting the publisher. Crops are an optional appearance
check, not a full-scene detector/tracker replacement. PRID is not required or
downloaded here.

## Evidence boundary

Software tests establish policy/data-flow behavior. The actual-model artificial
smoke establishes that YOLO, ByteTrack and trained OSNet execute through the new
path. Repeated images do not prove independent-camera accuracy. Real validation
still requires an A -> B -> A recording, followed by a second participant and
negative cases. Human labels must be independent of predicted global IDs.

EPFL's 67.3% association F1 remains an overlap result. It cannot be relabelled as
handoff accuracy. The inherited threshold and new margin are starting values,
not calibrated values for the phone/laptop setting. Lighting/viewpoint shifts,
similar clothing and tracker failures can still cause errors. A stream dropout
can resemble absence after observations expire; retain camera-health evidence.
