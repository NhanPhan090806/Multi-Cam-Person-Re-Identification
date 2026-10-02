# Multi-Camera Person Re-Identification

Course project for recovering a person's global ID after they leave one camera,
travel through an unseen area, and enter a camera at a different location.

The frozen scope and design decisions are in [plan.md](plan.md).

## Current Status

- [x] Stage 0 repository and environment foundation
- [x] Market-1501 downloader and integrity inspection
- [x] Typed Torchreid training configuration
- [x] One-epoch smoke-training mode
- [x] Kaggle/Colab training notebook
- [x] Full Market-1501 cloud training
- [x] Safe checkpoint loading and normalized crop embeddings
- [x] Standalone Re-ID evaluation and retrieval visualization
- [x] YOLO + ByteTrack single-camera pipeline
- [x] OSNet embeddings and rolling averages for actual ByteTrack tracklets
- [x] Sparse synchronized EPFL Laboratory C0/C1 replay with independent local trackers
- [x] Cross-camera global association with auditable global-ID logs and video

## Separate-Location Camera Handoff

The original EPFL stages tested overlapping views. Their results remain available
as historical diagnostics and do not establish handoff accuracy. The live app now
defaults to `models.association_mode: handoff`. `solo`/`multi_ip` select devices;
`handoff`/`hybrid`/`overlap` select the identity policy independently.

Keep Market-1501 and the trained OSNet checkpoint. Put the phone and laptop in
different areas with a blind region between them. Keep both streams running,
including empty views. Stand in A for a few seconds, leave both views, then enter B.
The registry remembers departed people for 120 seconds by default and attempts
to recover their original global ID. It does not merge simultaneous look-alikes.

`Global pending` means insufficient samples, ambiguous appearance, or a plausible
match awaiting the departure/time gate. It is reconsidered on later observations.
A successful recovery produces a black-area message such as
`Global 1: PHONE -> LAPTOP | gap 5.0s` plus a `handoff_events.jsonl` entry.

Record a short consented session:

~~~powershell
python main_camera_demo.py `
  --mode solo `
  --camera-url PHONE=http://192.168.1.23:8080/video `
  --record-dir recordings/handoff
~~~

Each recording creates a new session folder with raw JPEG frames, an ordered
timeline and a manifest. Its path appears as `recording_dir` in the session
summary. These are processed observations, not every hardware frame. Recording
consumes disk space and can lower throughput; use short sessions.

~~~powershell
python main_handoff_replay.py `
  --recording recordings/handoff/session_YOUR_SESSION `
  --output-dir outputs/handoff/my_replay `
  --show
~~~

Replay reuses the same detector, independent trackers, OSNet and handoff registry.
It uses recorded timestamps even when inference runs at a different speed.
The optional display runs at processing speed. Outputs include assignments,
handoff events/screenshots, settings and runtime counts. Accuracy requires human
labels; predicted IDs are not ground truth.

Configure `gallery_ttl_seconds`, `exit_grace_seconds`, `min_travel_seconds`,
`match_margin`, and optional directed `allowed_transitions` in the camera config.
The 0.75-second observation freshness window does not limit the travel gap.
No ground-plane calibration or synchronized views are needed.

Model-backed wiring smoke (requires the existing Stage 3 smoke input):

~~~powershell
python scripts/smoke_handoff.py
~~~

It schedules repeated smoke images as A -> blank -> B. This tests actual models
through the new path, but is artificial footage, not an accuracy benchmark.
See [migration notes](docs/handoff_migration.md) and [updated plan](plan.md).

## Environment

For a command-focused guide with arguments and examples, see [run.md](run.md).
All commands assume you run from the repository root with your Python environment
activated; `python` refers to that environment's interpreter.

The verified local environment is Python 3.12. The package supports Python 3.10-3.12 so it can run on current Kaggle and Colab images.

- PyTorch 2.5.1 + CUDA 12.1
- torchvision 0.20.1
- OpenCV 4.10.0
- Ultralytics 8.4.64
- Torchreid 0.2.5

PyTorch should be installed separately using the build appropriate for the target GPU. On Kaggle and Colab, use the preinstalled PyTorch unless there is a demonstrated incompatibility.

Install this repository and its development tools:

~~~powershell
python -m pip install ".[dev]"
~~~

This repository lives under a Windows path containing Vietnamese characters. Use a normal install as shown above: current setuptools editable installs (`-e`) cannot encode that path into their `.pth` file. Reinstall after changing package source locally; pytest reads directly from `src/`.

On Windows, installation also creates small `.exe` console launchers inside the
venv's `Scripts` directory. They are wrappers that invoke this package with the
venv's Python interpreter, not separately compiled applications. Every command
can alternatively be run through its corresponding script in `scripts/`.

## Verify the Repository

~~~powershell
python -m pytest
python -m ruff check .
~~~

## Download Market-1501

The downloader uses the public Kaggle mirror **pengcw1/market-1501**, extracts it into the Torchreid-compatible layout, and prints integrity statistics.

~~~powershell
python scripts/download_market1501.py --output-dir data/reid/market1501
~~~

Expected layout:

~~~text
data/reid/
└── market1501/
    └── Market-1501-v15.09.15/
        ├── bounding_box_train/
        ├── bounding_box_test/
        └── query/
~~~

## Train the Re-ID Baseline

Validate configuration and dataset paths without starting training:

~~~powershell
python scripts/train_market1501.py --dry-run
~~~

Run the one-epoch smoke configuration:

~~~powershell
python scripts/train_market1501.py --smoke
~~~

Run the full configuration:

~~~powershell
python scripts/train_market1501.py
~~~

The default settings are in [configs/train_market1501.yaml](configs/train_market1501.yaml). Training outputs and model checkpoints are ignored by Git.

## Verify the Trained Re-ID Checkpoint

The epoch-60 checkpoint is the selected model: 64.6% mAP, 84.0% Rank-1, 94.1% Rank-5, and 96.6% Rank-10. Verify that it loads and produces a unit-normalized embedding:

~~~powershell
python scripts/verify_reid_checkpoint.py
~~~

To encode a real OpenCV-compatible person crop instead of the synthetic smoke-test crop:

~~~powershell
python scripts/verify_reid_checkpoint.py --image path/to/person_crop.jpg
~~~

The application API accepts OpenCV BGR crops and returns 512-dimensional unit vectors:

~~~python
from pathlib import Path

from multicam_reid.reid import ReIDEncoder

encoder = ReIDEncoder.from_checkpoint(
    Path("outputs/train/osnet_x0_25_market1501_results/model/model.pth.tar-60")
)
embedding = encoder.encode_one(person_crop_bgr)
~~~

## Evaluate Re-ID Independently

Run the selected checkpoint through the project's application preprocessing and
embedding API, then independently calculate the official Market-1501 metrics:

~~~powershell
python scripts/evaluate_market1501.py --device cuda
~~~

The verified epoch-60 result is **64.50% mAP, 83.97% Rank-1, 94.12% Rank-5,
and 96.59% Rank-10** across all 3,368 queries. This independently reproduces
the rounded training-engine result closely while exercising the actual OpenCV
inference path that the camera pipeline will use.

Generated files are saved under `outputs/evaluation/market1501/`:

- `query_embeddings.npz` and `gallery_embeddings.npz`: normalized 512-D
  embeddings plus person IDs, camera IDs, and source paths;
- `embedding_manifest.json`: dataset path, checkpoint SHA-256, model, input size,
  epoch, and device provenance;
- `metrics.json`: mAP, CMC, counts, runtime, and artifact paths;
- `retrieval_good.jpg` and `retrieval_bad.jpg`: reproducible retrieval contact
  sheets;
- `retrieval_examples.json`: the exact queries and gallery rankings shown.

In the contact sheets, blue marks the query, green marks a correct identity, and
red marks an incorrect identity. Re-run metrics and visualizations without GPU
inference by reusing artifacts whose dataset and checkpoint provenance matches:

~~~powershell
python scripts/evaluate_market1501.py --reuse-embeddings
~~~

If either the dataset location or checkpoint SHA-256 differs, reuse fails closed
instead of silently evaluating stale embeddings.

The evaluator is also an independent Python module, so notebooks or later
pipeline code do not need to invoke a shell command:

~~~python
from pathlib import Path

from multicam_reid.evaluation import EvaluationConfig, run_evaluation

metrics = run_evaluation(
    EvaluationConfig(
        device="cuda",
        output_dir=Path("outputs/evaluation/market1501"),
    )
)
print(metrics["mAP"], metrics["cmc"])
~~~

This module is the Stage 2 quality gate. It proves that the saved checkpoint,
serving-time crop preprocessing, embeddings, cosine ranking, and evaluation
protocol work together before Stage 3 introduces YOLO and ByteTrack errors.

## Run the Single-Camera Pipeline

Stage 3 connects a person-only YOLO26n detector to one camera-owned ByteTrack
instance, validates person crops, draws camera-local IDs, and writes an annotated
video plus JSON runtime summary:

~~~powershell
python scripts/run_single_camera.py `
  --source path/to/input.mp4 `
  --output outputs/stage3/annotated.mp4 `
  --camera-id C1 `
  --device cuda `
  --crop-dir outputs/stage3/crops
~~~

Add `--display` for a live window and press Escape to stop. The YOLO confidence
floor defaults to `0.10` so ByteTrack receives both its high- and low-confidence
detection groups; ByteTrack starts new tracks at `0.25`.

The same implementation is available as separate modules:

~~~python
from multicam_reid.detection import YoloPersonDetector
from multicam_reid.pipeline import SingleCameraPipeline
from multicam_reid.tracking import ByteTrackLocalTracker

detector = YoloPersonDetector()
tracker = ByteTrackLocalTracker(camera_id="C1")
pipeline = SingleCameraPipeline(detector=detector, tracker=tracker)
processed = pipeline.process(frame_packet)
~~~

`processed.tracks` contains local IDs and boxes. `processed.crops` is the clean
boundary consumed by OSNet in Stage 4. One tracker object belongs to exactly one
camera; future cameras must get their own tracker while sharing the detector and
Re-ID models.

The box label starts with `person` because YOLO is deliberately called with only
COCO class `0`. The remaining text is the camera-local ByteTrack ID and detector
confidence, not a Re-ID or global identity.

The verified 60-frame diagnostic run maintained one local ID for all 60 tracked
frames, produced 60 valid crops, and processed 15.74 FPS on the GTX 1650. This is
a wiring smoke test using a simple generated video, not a claim about tracking
accuracy on difficult real footage.

## Connect One to Three IP Webcams

The IP-webcam adapter runs the same Stage 3 pipeline independently for every
enabled URL. It shares one YOLO model to conserve GPU memory, but every camera
owns a separate ByteTrack state and therefore a separate local-ID namespace.
There is still no cross-camera Re-ID at this point.

Create your ignored local configuration from the safe example:

~~~powershell
Copy-Item configs/ip_webcams.example.yaml configs/ip_webcams.local.yaml
~~~

Edit `configs/ip_webcams.local.yaml`. For the Android app named **IP Webcam**,
the stream URL is commonly:

~~~text
http://PHONE_IP:8080/video
~~~

Enable only the cameras currently available. If one entry is enabled, one
worker and window are created. If two or three are enabled, all are connected
and processed independently. Each entry has editable window geometry:

~~~yaml
- camera_id: C1
  url: http://192.168.1.101:8080/video
  enabled: true
  window_width: 640
  window_height: 360
  window_x: 0
  window_y: 0
~~~

Run the cameras after activating the venv:

~~~powershell
python scripts/run_ip_webcams.py `
  --config configs/ip_webcams.local.yaml `
  --device cuda
~~~

Press `q` or Escape to stop. Each stream has a background latest-frame reader,
short network timeouts, and automatic reconnection, so a disconnected phone
does not freeze the other cameras. YOLO inference runs sequentially on the
shared GTX 1650; this avoids loading duplicate models but total FPS will fall as
cameras are added.

Phone and computer must normally be on the same LAN. Start the phone's server,
keep its screen/app awake, allow Windows Firewall access on the private network,
and test the `/video` URL in a browser if OpenCV cannot connect. Do not commit
URLs containing usernames, passwords, or other credentials; the local YAML is
Git-ignored.

## Run Stage 4 Re-ID Integration

Stage 4 takes the validated person crops produced by Stage 3 and runs the trained
OSNet checkpoint on them. It samples each local track every few frames, keeps the
latest normalized embeddings, averages them, and normalizes the average again.
This rolling representation is less sensitive to one blurred or partially
occluded crop than a single-frame embedding.

Run the standalone integration diagnostic on a controlled clip containing at
least two clearly different people with stable tracks:

~~~powershell
python scripts/run_reid_diagnostic.py `
  --source path/to/two_people.mp4 `
  --device cuda `
  --embedding-interval 5 `
  --display
~~~

The display still shows `person` and local IDs. Its extra Re-ID line reports the
number of represented tracklets, stored samples, and embedding dimension; it
does not claim that cross-camera identity matching has happened. The command
writes these ignored artifacts under `outputs/stage4/reid_diagnostic/`:

- `diagnostic.json`: runtime plus mean same-track and different-track cosine
  similarities;
- `tracklet_embeddings.npz`: pickle-free 512-D rolling representations and the
  normalized samples used to build them.

The diagnostic treats different local IDs in the controlled clip as different
people, so do not use arbitrary footage with ByteTrack ID switches as ground
truth. The verified two-person run processed 60 frames into two stable
tracklets, sampled 24 embeddings, and measured **0.974 same-track cosine
similarity versus 0.403 different-track similarity** (margin **0.571**) at
24.16 FPS on the GTX 1650.

The reusable Python composition is:

~~~python
from pathlib import Path

from multicam_reid.pipeline import ReIDCameraPipeline, SingleCameraPipeline
from multicam_reid.reid import ReIDEncoder, TrackletEmbeddingStore

stage3 = SingleCameraPipeline(detector=detector, tracker=tracker)
encoder = ReIDEncoder.from_checkpoint(
    Path("outputs/train/osnet_x0_25_market1501_results/model/model.pth.tar-60")
)
stage4 = ReIDCameraPipeline(
    stage3=stage3,
    tracklets=TrackletEmbeddingStore(encoder),
)
result = stage4.process(frame_packet)

for appearance in result.appearances:
    print(appearance.key, appearance.embedding.shape, appearance.stored_samples)
~~~

`appearance.key` is only `(camera_id, local_id)`. Stage 6 compares these rolling
vectors across cameras and assigns global IDs; OSNet itself does not perform
that decision.

Run Stage 4's isolated unit tests without invoking the repository-wide coverage
gate:

~~~powershell
python -m pytest tests/unit/test_tracklet_embeddings.py --no-cov
~~~

## Run Stage 5 EPFL Laboratory C0/C1 Replay (Historical Overlap Diagnostic)

The original Stage 5 used the EPFL Laboratory six-person sequence as its primary real
multi-camera input. It contains four synchronized ordinary-perspective videos of
the same room, with people entering sequentially rather than appearing as a
crowd. The default C0/C1 subset is about 156 MB and is deliberately chosen for
debugging and a visually understandable class demonstration.

Download C0, C1, the camera homographies, and the identity-labelled ground-grid
positions directly from EPFL without an account or manual browser step:

~~~powershell
python scripts/download_epfl_lab.py `
  --output-dir data/epfl_lab `
  --cameras C0 C1
~~~

The validator checks equal video lengths, frame rates and dimensions; parses the
old EPFL ground-truth format; and verifies calibration for every selected camera.
The local C0/C1 subset contains:

- 2,955 synchronized frames at 25 FPS and 360x288;
- 119 identity-labelled timestamps, one every 25 frames;
- six persistent identity columns and at most six people in the room;
- 476 active ground-position labels;
- 155,830,968 bytes on disk.

Run the verified two-person debugging window beginning at source frame 400:

~~~powershell
python scripts/run_epfl_lab.py `
  --dataset-root data/epfl_lab `
  --output-dir outputs/stage5/epfl_lab_preview `
  --cameras C0 C1 `
  --device cuda `
  --start-frame 400 `
  --max-frames 150 `
  --display
~~~

The command writes `C0_annotated.mp4`, `C1_annotated.mp4`, `summary.json`, and
`synchronized_annotated.mp4`. The last file places the views side by side at the
same source frame and timestamp, which makes it much easier to verify visually
that the same person really occurs in both cameras. `--frame-stride N` permits a
faster sampled replay without changing source timestamps; `--no-video` performs
measurement without rendering files.

The verified 150-pair preview processed 300 camera frames in 9.42 seconds
(31.86 camera frames/s). It saw two ground-truth identities; C0 produced 261
detections and 231 track observations, while C1 produced 325 detections and 275
track observations. A single YOLO detector is shared, but C0 and C1 still own
separate ByteTrack state and local-ID namespaces.

This stage still does not assign cross-camera global IDs. Stage 6 combines this
synchronized input with Stage 4 OSNet tracklet embeddings, cosine distance,
Hungarian matching, and a global identity registry.

The older WILDTRACK downloader and runner remain available as optional crowd
stress-test tools. Their 2.06 GiB local download was removed after this migration
and can be recovered by rerunning `scripts/download_wildtrack.py`.

## Run Stage 6 Global Association (Historical EPFL Overlap Diagnostic)

Stage 6 is the first complete multi-camera Re-ID pipeline. For every synchronized
instant it runs:

~~~text
C0/C1 frames -> YOLO -> separate ByteTrack instances -> person crops
             -> shared OSNet -> rolling tracklet embeddings
             -> cosine-distance matrices -> Hungarian one-to-one matches
             -> global-ID registry -> matching labels/colors in both views
~~~

Run the verified two-person development window:

~~~powershell
python scripts/run_global_epfl.py `
  --dataset-root data/epfl_lab `
  --output-dir outputs/stage6/epfl_lab_dev_400_549 `
  --cameras C0 C1 `
  --device cuda `
  --start-frame 400 `
  --max-frames 150 `
  --display
~~~

The default cosine-distance threshold is `0.35`, equivalent to requiring cosine
similarity of at least `0.65`. A tracklet needs two stored OSNet samples before
it may merge with another identity. Hungarian assignment makes matches one to
one for each camera pair, and the registry refuses any merge that would put two
simultaneously active tracks from the same camera into one global identity.
Inactive identities remain available for 250 source frames so an interrupted or
returning track can reattach.

The output directory contains:

- `C0_global_ids.mp4` and `C1_global_ids.mp4`: separate annotated views;
- `synchronized_global_ids.mp4`: side-by-side evidence at the same source time;
- `assignments.jsonl`: one machine-readable local-to-global assignment per
  visible appearance and frame;
- `merge_events.jsonl`: every accepted merge, its track keys, frame, and cosine
  distance;
- `summary.json`: settings, counts, output paths, and measured throughput.

In the verified 150-pair run, 300 camera frames were processed in 11.20 seconds
(26.79 camera frames/s). The two main people were assigned consistently across
C0 and C1 as Global 1 and Global 3. Global 3 also survived a C1 ByteTrack switch
from local 3 to local 12. Six provisional numeric IDs were issued and three
cross-camera merges left three final identities; the third was a four-frame C0
local track, illustrating that detector/tracker fragmentation propagates into
global association. These are Stage 6 development observations, not final
association precision/recall; Stage 7 will attribute tracks to EPFL ground truth
before reporting quantitative accuracy.

The reusable association module is independent of EPFL:

~~~python
from multicam_reid.association import AssociationConfig, GlobalIdentityRegistry

registry = GlobalIdentityRegistry(
    AssociationConfig(max_cosine_distance=0.35, min_stored_samples=2)
)
result = registry.update(
    appearances_from_all_cameras,
    frame_id=source_frame_id,
    timestamp=source_timestamp,
)
global_id = result.assignments[track_key]
~~~

That separation matters for Stage 8: the EPFL source can be replaced by two
IP-camera workers while the association policy and global registry stay the same.

## Run Stage 7 Evaluation (Historical EPFL Overlap Diagnostic)

Stage 7 does not train another model and does not alter Stage 6 IDs. It is an
offline measurement module: it takes `assignments.jsonl` from a completed Stage
6 run, projects each tracked bounding-box footpoint onto the EPFL ground plane,
matches predictions to the sparse identity-labelled positions, and scores the
result. This separation makes evaluation quick to repeat without running YOLO,
ByteTrack, or OSNet again.

Run the full sequence once with the frozen Stage 6 settings:

~~~powershell
python scripts/run_global_epfl.py `
  --dataset-root data/epfl_lab `
  --output-dir outputs/stage6/epfl_lab_full `
  --cameras C0 C1 `
  --device cuda `
  --no-video
~~~

Then evaluate its saved assignments:

~~~powershell
python scripts/evaluate_epfl_global.py `
  --dataset-root data/epfl_lab `
  --stage6-run-dir outputs/stage6/epfl_lab_full `
  --output-dir outputs/stage7/epfl_lab_full `
  --cameras C0 C1 `
  --max-ground-distance-cells 5 `
  --market1501-metrics outputs/evaluation/market1501/metrics.json
~~~

The evaluator writes `metrics.json` (complete machine-readable results and
SHA-256 provenance), `attributions.jsonl` (track-to-person matches and rejected
predictions), `contingency.csv` (person/global-ID counts), and `report.md` (a
short report-ready summary).

The verified full C0/C1 run processed 2,955 synchronized frames (5,910 camera
frames) in 246.80 seconds: 11.97 synchronized pairs/s or 23.95 camera frames/s.
At the 119 annotated frames, 761/860 eligible observations were spatially
attributed and 450/476 scene-person opportunities were covered. Cross-camera
association reached **65.5% precision, 69.1% recall, and 67.3% F1** (215 TP, 113
FP, 96 FN). This beats both deliberately trivial baselines: cameras kept wholly
separate score 0 F1, while assigning everyone one ID scores 35.5% F1.

The result is usable but not strong. There were 118 sparse local-ID switches and
115 sparse global-ID switches; ByteTrack fragmentation is therefore a major
upstream limitation, and OSNet/global association only repairs part of it. EPFL
provides sparse scene-level ground positions rather than per-camera boxes and
visibility flags, so these are calibrated project metrics—not official
MOTChallenge MOTA or IDF1 scores. The Stage 6 cosine threshold remained frozen
at 0.35; the reporting sequence was not used for repeated threshold tuning.

## Run Stage 8 Live Camera Demo

Stage 8 replaces synchronized dataset files with independently threaded live
sources while reusing YOLO, ByteTrack and OSNet with the handoff association
modules. The repository-surface command supports two source arrangements:

- `solo`: exactly one phone IP Webcam plus one laptop webcam, suitable for one
  participant operating both cameras;
- `multi_ip`: two or more phone/IP-camera streams, suitable for a group demo.

First start the IP Webcam server on the phone and place every device on the same
local network. Android IP Webcam commonly displays an address such as
`http://192.168.1.23:8080`; its OpenCV MJPEG endpoint is normally `/video`.

Validate configuration without opening models or cameras:

~~~powershell
python main_camera_demo.py `
  --mode solo `
  --camera-url PHONE=http://192.168.1.23:8080/video `
  --dry-run
~~~

Run the one-person phone plus laptop demonstration:

~~~powershell
python main_camera_demo.py `
  --mode solo `
  --camera-url PHONE=http://192.168.1.23:8080/video `
  --webcam-index 0
~~~

If Windows exposes the laptop camera under another index, try
`--webcam-index 1`. For a multi-IP session:

~~~powershell
python main_camera_demo.py `
  --mode multi_ip `
  --camera-url PHONE_A=http://192.168.1.23:8080/video `
  --camera-url PHONE_B=http://192.168.1.24:8080/video
~~~

The dashboard keeps camera pixels clear except for colored bounding boxes. Each
camera has a separate black information panel underneath it showing connection
health and explicit `Global N | Local N | det 0.xx` rows in the same color as
the corresponding box. Click **Save screenshot [S]** or press `S` to save the
complete annotated dashboard. A green success message appears for one second
only after the clean screenshot has been written, so the confirmation itself is
not captured. Press `Q` or Escape to stop. Screenshots, non-sensitive session
settings, assignments, handoff/merge events, and runtime counts are written under
`outputs/stage8/live_demo/`. Stream URLs are deliberately excluded.

Edit `configs/camera_demo.yaml` to change model paths, capture dimensions,
dashboard tile size, active-camera tolerance, or the two camera lists. The live
loop consumes only each worker's newest frame, so a slow network stream cannot
build an unbounded queue. A 0.75-second window limits cached observation freshness.
It does not limit time spent travelling between cameras: the identity gallery
retention controls that. `det` is detection confidence, not a Re-ID probability.

## WiseNET handoff test

Run the existing YOLO + ByteTrack + trained OSNet + handoff gallery on full-scene
WiseNET Set 2 videos (default cameras 3/4). Labels are loaded only after inference
for evaluation; they never determine predicted identities.

~~~powershell
python main_wisenet_test.py `
  --output-dir outputs/handoff/wisenet_set2_run2
~~~

Choose a new output directory on each run; previous predictions are never silently
overwritten. Add `--show` for a preview at processing speed, `--stride 3` for 10 FPS,
or `--start 15 --end 90` for a shorter source interval. Default stride 6 processes
5 FPS per camera while retaining original timestamps and background frames.
The input uses the local `data/wisenet_set2/video_set2/` and
`annotations_set2/people_detection/` layout. Other camera pairs may overlap.

For the four-camera mixed network, explicitly permit simultaneous matching only
between the views you know overlap:

~~~powershell
python main_wisenet_test.py `
  --cameras 2 3 4 5 --overlap 2,3 4,5 `
  --output-dir outputs/handoff/wisenet_hybrid --show
~~~

`--overlap` automatically selects **hybrid** association. Pairs are bidirectional:
`2,3` equals `3,2`; all other simultaneously occupied pairs remain prohibited.
Permissions are not transitive: to share one ID among three simultaneous views,
list all three pairs. No overlap option means the existing strict handoff default
(unless your config explicitly selects hybrid). `--no-overlap` clears configured
pairs and restores strict handoff. Use `--dry-run` to validate without inference.
You must still verify that a doorway/room actually provides shared visibility;
the program does not infer camera geometry from numbers or dataset labels.

The same option works for live cameras and raw recording replay, using their
configured IDs instead of WiseNET numbers:

~~~powershell
python main_camera_demo.py `
  --mode solo --overlap PHONE,LAPTOP
python main_handoff_replay.py `
  --recording recordings/handoff/session_YOUR_SESSION --overlap PHONE,LAPTOP `
  --output-dir outputs/handoff/hybrid_replay
~~~

For persistent live configuration, set `models.association_mode: hybrid` and
`models.overlap_pairs: [[PHONE, LAPTOP]]` in `configs/camera_demo.yaml`. Named pairs
must reference cameras in the selected live mode. CLI pairs override the config.

Hybrid safeguards: at most one active track per camera per global ID; every
simultaneously occupied camera pair must be allowed; cosine distance <= 0.25
(also capped by the handoff threshold); three consecutive accepted comparisons
on distinct new embedding samples; and ambiguity rejection for both competing
identities and competing arrivals. A group can perform a non-overlapping handoff
only after **all** its visible members have departed and time/route gates pass.
Already-issued IDs can now be reconciled, but only when both identities are
visible in permitted overlapping cameras. The match must be mutual-best and
unambiguous, with five accepted comparisons on fresh samples from **both** sides,
spanning at least one second. The same strict 0.25 overlap threshold applies.
Current or historically observed same-camera conflicts prohibit merging. The
smaller ID survives; the obsolete ID is recorded as an alias. Earlier screenshots
and assignment logs are not rewritten. Use `--no-reconcile` to compare with the
previous arrival-only policy, or set `models.reconciliation_enabled: false`.
Overlap thresholds are uncalibrated defaults, not guaranteed accuracy settings.
Identical-looking people in permitted pairs can still be falsely joined; topology
and repeated appearance evidence reduce risk, but do not prove physical identity.
Duplicate local boxes are a separate tracker issue, not repaired by this policy.

The standalone `association/reconciliation.py` module consumes existing identity
groups and fresh Stage 4 appearances, returning merge proposals plus decision
diagnostics. It does not load YOLO/OSNet or open cameras. The Stage 6 registry
applies approved merges; Stage 8 live/replay callers render the resulting IDs.
Run its isolated checks with:

~~~powershell
python -m pytest `
  tests/unit/test_reconciliation.py --no-cov -q
~~~

Outputs include `dashboard.mp4` (colored boxes plus Global/Local IDs in black
panels), `assignments.jsonl`, `handoff_events.jsonl`, `overlap_events.jsonl`,
`merge_events.jsonl`, `association_decisions.jsonl`, `processed_frames.jsonl`,
screenshots, `settings.json`,
`summary.json`, and `evaluation.json`. The first baseline is in
`outputs/handoff/wisenet_set2/`. To re-score saved predictions without model inference:

~~~powershell
python main_wisenet_test.py --evaluate-only
~~~

When re-scoring a custom run, repeat its `--output-dir`, `--cameras`, `--stride`,
`--start`, and `--end` values. The evaluator uses one-to-one IoU >= 0.5 matching
against clipped manual boxes, deduplicates cached-view log entries, and separately
reports prompt recovery (within two seconds), delayed recovery, ID fragmentation,
shared IDs, and tracked-box coverage. Same-camera recovery events are not counted
as cross-camera transitions. This is a two-person development diagnostic, not a
general-purpose benchmark or proof of live phone/laptop performance.
Simultaneous overlap matches are logged separately and are not counted as
departure/arrival recoveries. Existing WiseNET transition scoring remains a
development diagnostic, particularly for mixed overlapping networks.

`association_decisions.jsonl` records candidate distances/thresholds, rejected
camera pairs, exit/travel waiting, ambiguous candidates, confirmation waiting,
and accepted reconciliations. It contains no saved OSNet embedding vectors.
`processed_frames.jsonl` includes empty views too: if you quit with Q/Esc, scoring
excludes unprocessed footage. Older WiseNET runs are limited using their saved
per-camera counts and sampling settings; repeat those settings when re-scoring.
`global_ids_by_person` and continuity scores use original, causal predictions;
`resolved_global_ids_by_person` and `resolved_shared_global_ids` separately audit
final aliases, without retroactively making early mistakes disappear.

Verified final untuned reconciliation replay: `outputs/handoff/wisenet_reconciled_final`.
At 76.0s, C4 Global 3 was reconciled into C5 Global 1 (distance about 0.185).
The early C2/C3 example remains separate: its measured distances about
0.324-0.343 exceed the fixed 0.25 threshold. Three IDs were issued historically,
but only two distinct identities remained after the accepted merge. Duplicate
C5 tracks can still create/retain splits; this change does not repair the tracker.

## Cloud Training

Use [train_market1501_kaggle.ipynb](train_market1501_kaggle.ipynb) on Kaggle, or [train_market1501.ipynb](train_market1501.ipynb) when the repository is already available locally. The notebooks:

1. verifies GPU availability;
2. installs the repository package;
3. downloads and validates Market-1501;
4. resolves the training configuration;
5. exposes separate smoke and full-training switches.

Use a GPU runtime. Copy the best checkpoint out of the ephemeral runtime after training.

## Dataset Rules

- Do not commit or redistribute Market-1501, EPFL Laboratory, or WILDTRACK.
- Do not commit checkpoints, recordings, credentials, or RTSP URLs.
- Use the datasets for non-commercial educational work and cite their original papers.
