# Person Re-Identification Across Separate Camera Locations

Updated: 2026-10-01. This is a university engineering group project.

## 1. Mission and correction

The assignment is: "Person Re-identification: recognize the same person across multiple cameras."

The intended demonstration is a person entering camera A, leaving its view,
remaining unseen while travelling, and appearing in camera B at a different
location. The system should recover the original global ID using appearance
evidence. Cameras do not need to see the person simultaneously.

The previous plan selected overlapping views and EPFL Laboratory. That tested a
valid but different scenario. Its code/results remain useful as integration
history; they are not evidence that the required camera handoff works.

## 2. Scope

Required:

- Two fixed cameras watching different locations, with a blind area between them.
- Solo sources: one phone IP Webcam and one laptop webcam.
- Group sources: two or three IP-camera streams.
- Pretrained YOLO person detector, independent ByteTrack state per camera.
- Existing trained OSNet-x0.25, normalized 512-D embeddings and rolling averages.
- A time-limited gallery remembering people after they leave a camera.
- Conservative appearance-based recovery of global IDs on later arrivals.
- Visible global IDs, handoff messages, saved evidence and reproducible replay.
- A recorded demonstration and an honest account of failures/limitations.

Optional: directed camera routes and fixed minimum travel time; a third camera;
Blender/prerecorded simulation of two separate locations after the real demo.

Out of scope: learning camera topology/travel-time distributions, retraining the
detector, face recognition, real-world names, clothing changes, physics/robotics
simulation, distributed inference, or a new research model.

## 3. What survives the correction

| Component | Decision |
|---|---|
| Market-1501 and training notebooks | Keep; cross-camera appearance learning still fits |
| 60-epoch OSNet checkpoint | Keep; no retraining required by this architecture change |
| Market query/gallery evaluation | Keep; measures crop retrieval, not whole-scene handoff |
| YOLO + ByteTrack + crop validation | Keep unchanged, independently per camera |
| Tracklet embeddings and inference preprocessing | Keep |
| Live phone/laptop and multi-IP workers | Keep |
| Black ID panels and screenshot controls | Keep |
| Overlap global registry | Retain behind explicit association_mode: overlap |
| Default global registry | Replace with departure/arrival handoff policy |
| EPFL and WILDTRACK pipelines | Optional historical overlap diagnostics |
| EPFL association accuracy report | Preserve but label as overlap-only |

## 4. Data decision

### Training and image-level evaluation: Market-1501

Keep the existing dataset and official train/query/gallery protocol. No new
training dataset is required. The selected checkpoint is:

outputs/train/osnet_x0_25_market1501_results/model/model.pth.tar-60

Training-engine result: 64.6% mAP and 84.0% Rank-1. Independent application
preprocessing result: 64.50% mAP and 83.97% Rank-1. These results do not guarantee
accuracy on the phone/laptop environment. There was no separate validation split
in the completed baseline; do not describe that run as validation-selected.

### Primary full-scene handoff data: consented phone/laptop recordings

Use the devices already available. Put the phone in another room or around a
corner; keep the laptop fixed. Keep both streams running, including background
frames when the participant is absent. Walk A -> blind area -> B -> blind area -> A.

Run main_camera_demo.py with --record-dir recordings/handoff to create a fresh
session folder with raw JPEG frames, frames.jsonl and manifest.json. Camera IDs
and actual session timestamps are saved. URLs/credentials are excluded. These
are processed observations, not every hardware capture frame. JPEG recording
costs storage and can reduce throughput.

Replay via main_handoff_replay.py --recording <session-folder>. Replay uses
recorded event timestamps, independent of inference speed, with the same models,
separate per-camera trackers, and handoff registry.

These recordings are demonstration/development data. Accuracy claims require
independent human labels. Self-predicted IDs are never ground truth.

### Optional public crop sequences: PRID2011

Publisher: https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/prid11/

The publisher supplies single-shot and multi-shot person crops from two cameras,
with 200 shared identities. Original full-scene videos require contacting the
publisher. PRID is an optional appearance-sequence check, not an automatic-download
replacement for the full YOLO + tracking demo. It is not required, downloaded,
or used by the current refactor.

### Existing overlap data

EPFL C0/C1 stays on disk as an optional regression dataset; its downloader and
evaluation code remain. Its current local subset is approximately 149 MiB.
WILDTRACK's former local download was already removed; its optional code stays.
No datasets or trained checkpoints are deleted by this migration.

## 5. Handoff architecture

~~~text
Each camera: frames -> YOLO persons -> ByteTrack -> validated crops -> OSNet
                  -> rolling normalized tracklet embedding

New arrival -> enough samples?
             no: pending
             yes: compare against INACTIVE identity gallery
                  -> appearance + time + optional route gates
                  -> ambiguity margin + one-to-one assignment
                  -> recover old global ID, or create new ID

Known visible track -> keep its assigned global ID
Departure -> retain identity for a configurable TTL in seconds
TTL expires -> remove remembered identity; never recycle its numeric ID
~~~

HandoffIdentityRegistry is a separate module consuming tracklet appearances and
returning assignments and handoff events. It does not open cameras or load models.
Unit tests exercise the handoff rules independently.

Defaults in configs/camera_demo.yaml:

| Setting | Default / meaning |
|---|---|
| association_mode | handoff; independent of solo/multi-IP source mode |
| min_stored_samples | 2 OSNet samples before assigning a new track |
| max_cosine_distance | 0.35; inherited threshold, not calibrated for this environment |
| gallery_ttl_seconds | 120 seconds after last visibility |
| exit_grace_seconds | 1 second of absence before permitting a transfer |
| min_travel_seconds | 0; optional fixed minimum since last visibility |
| match_margin | 0.05 separation from the second-best candidate |
| allowed_transitions | Empty: any route; optional directed camera pairs |

One global ID is active on at most one local track in handoff mode. Simultaneous
look-alikes in different cameras are not merged. Ambiguous matches and arrivals
during a plausible departure grace period remain pending and are reconsidered.
Each identity retains a template per camera to accommodate viewpoint changes.

The 0.75-second association_window_seconds only limits cached observation freshness.
It does not limit the travel gap or require synchronized views. Gallery retention
uses elapsed seconds rather than frames/association updates. The gallery is
session-local and does not survive application restarts.

The black UI shows Global/Local IDs and det, which is YOLO detection confidence,
not a Re-ID probability. Handoff events include cosine distance and time gap.

## 6. Progress after correction

- [x] Stage 0: environment and repository.
- [x] Stage 1: Market-1501 OSNet training.
- [x] Stage 2: independent retrieval evaluation and checkpoint verification.
- [x] Stage 3: detection, independent local tracking, live input adapters.
- [x] Stage 4: OSNet tracklet embeddings.
- [x] Stage 5 software: record/replay separate-camera full-scene data.
- [ ] Stage 5 data: collect an actual A -> unseen -> B recording.
- [x] Stage 6 software: handoff memory, maturation, time/route gates,
  ambiguity handling, one-to-one recovery and handoff event logs.
- [x] Stage 6 tests: blind gaps, late maturity, wrong person, similar simultaneous
  people, TTL, return trips, local fragmentation, time/route constraints.
- [x] Stage 6 model wiring: synthetic schedule through actual YOLO, ByteTrack and
  trained OSNet; explicitly not independent-camera accuracy.
- [ ] Stage 7: label separate-location recordings and measure handoff errors.
- [x] Stage 8 software: solo/multi-IP default to handoff, ID panels, transition
  messages, screenshots and optional raw recording.
- [ ] Stage 8 hardware: verify and record handoffs on the actual phone/laptop.
- [ ] Stage 9 optional: simulation with separate locations and a blind interval.

Historical overlap work remains completed: EPFL synchronized replay (Stage 5),
same-time global matching (Stage 6), and EPFL evaluation (Stage 7). Its reported
67.3% association F1, 65.5% precision and 69.1% recall are conditional on calibrated
EPFL attribution. Do not reuse those numbers as non-overlapping handoff results.

Actual-model synthetic smoke on 2026-10-01: 62 full-frame observations processed
in 3.19 seconds, one global ID issued and one A -> B handoff after a blind gap.
This verifies the new wiring, not independent-camera accuracy.

## 7. Verification and evaluation

Software verification: association unit tests; recording/replay round-trip;
integration replay with A departure, blind gap and B arrival; existing regression
suite; real-model synthetic wiring smoke. Duplicated source footage tests wiring
only, not independent-camera performance.

For real recordings, human-label each participant's visits and transitions. Keep
a small development recording separate from the final demonstration recording.
Do not tune the threshold repeatedly on the final recording.

Report expected handoffs, correctly recovered IDs, missed handoffs, false joins,
ID changes, pending/unidentified visits and processing throughput. Explain
detector misses, local fragmentation and appearance ambiguity separately.
Retain the standard Market retrieval metrics in their own section.

## 8. Demo acceptance procedure

1. Keep both sources connected, pointing at different places.
2. Stand in A long enough for two valid embedding samples and note the ID.
3. Leave A fully; spend several seconds outside both views.
4. Enter B within the gallery TTL; confirm the recovered ID and A -> B message.
5. Repeat B -> A. Confirm new local IDs map back to the same global ID.
6. Test a different participant, a longer gap, occlusion and similar clothing.
7. Save screenshots and record the raw session. Replay the session offline.

The solo test validates continuity, not multi-person discrimination. A second
participant is required before claiming that the system distinguishes people.

## 9. Limits and submission

Clothing similarity, body crops, lighting/viewpoint shifts and local tracker errors
can cause wrong IDs. Strong appearance evidence is necessary; temporal rules do
not guarantee a match. Similar candidates can remain pending. Network dropouts
can resemble absence after cached observations expire; report camera health
alongside a handoff. Numeric IDs are anonymous and session-local.

Submit training/evaluation evidence, corrected architecture, separate-place
handoff recording/replay, measured errors and limits, and team contributions.
Simulation and a third camera are optional. Keep recordings/data/weights and
credentials out of Git; record consented participants and cite publishers.
