# Running the project

Run commands from the repository root, using an activated Python environment
with the project dependencies installed. All examples use `python`; no
machine-specific environment path is required. The multi-line examples use
PowerShell backticks. On another shell, put the command on one line instead.

For the current project, choose one workflow:

| Goal | Command |
|---|---|
| Test existing WiseNET videos | `python main_wisenet_test.py ...` |
| Run phone/laptop or multiple IP cameras live | `python main_camera_demo.py ...` |
| Replay a recorded live session | `python main_handoff_replay.py ...` |

These run YOLO detection, per-camera ByteTrack, trained OSNet embeddings, and
global-ID association together. You do not need to run earlier stages separately
or retrain OSNet before every demo. See [plan.md](plan.md) for project status and
[README.md](README.md) for architecture, results, and earlier-stage diagnostics.

## 1. Environment and prerequisites

Check which environment `python` uses, then install the project if needed:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip install ".[dev]"
python scripts/verify_reid_checkpoint.py --device auto
```

Use a normal install, not `-e`, for this repository's non-ASCII Windows path.
The root-level demo scripts import the current `src/` code directly; reinstall
after source changes if using installed package commands elsewhere. PyTorch and
torchvision must already be installed with compatible builds; the project install
does not select a CUDA build for you.

The default config is [configs/camera_demo.yaml](configs/camera_demo.yaml). It uses:

- Detector: `yolo26n.pt` (a missing model may trigger a download).
- Re-ID checkpoint: `outputs/train/osnet_x0_25_market1501_results/model/model.pth.tar-60`.
- WiseNET input: `data/wisenet_set2/` with `video_set2/` and annotations.

Keep the checkpoint as-is: PyTorch loads it directly; no `.tar` extraction is
needed. WiseNET evaluation needs its annotations, not just video files. Live
operation does not require Market-1501 or WiseNET to be loaded.

## 2. WiseNET video testing

Separate-room baseline (cameras 3 and 4), forcing strict handoff:

```powershell
python main_wisenet_test.py --cameras 3 4 --no-overlap `
  --output-dir outputs/handoff/wisenet_rooms_run1 --show
```

Four-camera hybrid test with only the declared overlapping pairs permitted:

```powershell
python main_wisenet_test.py --cameras 2 3 4 5 --overlap "2,3" "4,5" `
  --output-dir outputs/handoff/wisenet_hybrid_run1 --show
```

Validate without model inference:

```powershell
python main_wisenet_test.py --cameras 2 3 4 5 --overlap "2,3" "4,5" --dry-run
```

Process only part of the footage, at 10 sampled frames per second per camera:

```powershell
python main_wisenet_test.py --cameras 3 4 --no-overlap `
  --start 15 --end 90 --stride 3 `
  --output-dir outputs/handoff/wisenet_short_run1 --show
```

Choose a **new output directory for each inference run**: WiseNET refuses to
overwrite an existing one. Preview playback runs at processing speed; press
`Q` or Escape to stop early. Scoring then covers only processed frames.

Re-score saved predictions without running YOLO or OSNet again:

```powershell
python main_wisenet_test.py --cameras 2 3 4 5 `
  --output-dir outputs/handoff/wisenet_hybrid_run1 --evaluate-only
```

Repeat the original `--cameras`, `--data-root`, `--stride`, `--start`, and `--end`
settings when evaluating a saved run. Re-scoring overwrites its `evaluation.json`,
not its predictions or videos. Labels are used only after inference for scoring.

<!-- AUTO-GENERATED: argument reference from main_wisenet_test.py and association/topology.py -->

| Argument | Default / meaning | Example |
|---|---|---|
| `--data-root PATH` | `data/wisenet_set2`; dataset root | `--data-root data/wisenet_set2` |
| `--config PATH` | `configs/camera_demo.yaml`; models/association/display settings | `--config configs/camera_demo.yaml` |
| `--cameras N ...` | `3 4`; select available WiseNET cameras 1-5 | `--cameras 2 3 4 5` |
| `--stride N` | `6`; process every Nth frame; source FPS is 30 | `--stride 3` gives 10 FPS |
| `--start SECONDS` | `0`; start of source-time interval | `--start 15` |
| `--end SECONDS` | No explicit end; use available video duration | `--end 90` |
| `--output-dir PATH` | `outputs/handoff/wisenet_set2` | `--output-dir outputs/handoff/run2` |
| `--show` | Off; display the dashboard | `--show` |
| `--evaluate-only` | Off; re-score existing assignments | `--evaluate-only` |
| `--dry-run` | Off; validate inputs/config, without inference | `--dry-run` |

Shared overlap arguments are listed in section 5. `-h` / `--help` prints help.

<!-- END AUTO-GENERATED -->

## 3. Live phone/laptop and multi-IP demo

Start each phone's IP Webcam server and put the phones and laptop on the same
LAN. Replace the example IP addresses with those shown by your phones. A typical
MJPEG URL ends in `/video`. Leave cameras running even when their views are empty.

Solo mode: one phone plus one laptop webcam, watching different places:

```powershell
python main_camera_demo.py --mode solo `
  --camera-url PHONE=http://192.168.1.23:8080/video `
  --webcam-index 0 --no-overlap `
  --output-dir outputs/stage8/solo_run1
```

Walk PHONE -> unseen area -> LAPTOP -> unseen area -> PHONE. If the laptop camera
is unavailable at index 0, try `--webcam-index 1`. Add `--dry-run` to check the
selected config without opening cameras or models; this does not test connectivity.

Multiple IP cameras:

```powershell
python main_camera_demo.py --mode multi_ip `
  --camera-url PHONE_A=http://192.168.1.23:8080/video `
  --camera-url PHONE_B=http://192.168.1.24:8080/video `
  --no-overlap --output-dir outputs/stage8/multi_ip_run1
```

`--camera-url` replaces an existing configured camera's URL; it does not add a
camera or enable a disabled one. To use a third phone, enable `PHONE_C` in the
YAML and set its URL. Solo mode expects exactly one IP camera and one webcam;
multi-IP mode expects at least two enabled IP cameras.

For overlapping live views, replace `--no-overlap` with the appropriate named
pair, e.g. `--overlap "PHONE,LAPTOP"` or `--overlap "PHONE_A,PHONE_B"`.

In the live dashboard:

- Black panels show `Global`, camera-local `Local`, and `det` (YOLO confidence,
  not a Re-ID probability). Matching global IDs use matching box colors.
- Click **Save screenshot [S]** or press `S` to save the full dashboard. The
  one-second success message appears after saving and is not in the screenshot.
- Press `Q` or Escape to stop.

To record raw processed frames as well, add:

```powershell
python main_camera_demo.py --mode solo `
  --camera-url PHONE=http://192.168.1.23:8080/video --no-overlap `
  --record-dir recordings/handoff --output-dir outputs/stage8/recorded_run1
```

Each recording creates a new session folder. Find its exact path in the printed
summary's `recording_dir`. Recording JPEGs consumes disk space and can slow the
demo; these are processed observations, not every hardware capture frame.

<!-- AUTO-GENERATED: argument reference from pipeline/camera_demo.py and association/topology.py -->

| Argument | Default / meaning | Example |
|---|---|---|
| `--config PATH` | `configs/camera_demo.yaml` | `--config configs/camera_demo.yaml` |
| `--mode {solo,multi_ip}` | YAML `active_mode` (currently `solo`) | `--mode multi_ip` |
| `--camera-url ID=URL` | Repeatable; override a configured IP camera URL | `--camera-url PHONE=http://192.168.1.23:8080/video` |
| `--webcam-index N` | YAML webcam index (currently `0`) | `--webcam-index 1` |
| `--device {auto,cpu,cuda}` | YAML device (currently `auto`) | `--device cpu` |
| `--detector-model PATH` | YAML detector model | `--detector-model yolo26n.pt` |
| `--checkpoint PATH` | YAML OSNet checkpoint | `--checkpoint outputs/train/osnet_x0_25_market1501_results/model/model.pth.tar-60` |
| `--association-mode {handoff,hybrid,overlap}` | YAML policy (currently `handoff`); `overlap` is the old registry | `--association-mode handoff` |
| `--gallery-ttl-seconds N` | YAML retention (currently `120` seconds) | `--gallery-ttl-seconds 180` |
| `--record-dir PATH` | YAML setting (currently off); raw timestamped session frames | `--record-dir recordings/handoff` |
| `--output-dir PATH` | YAML output (currently `outputs/stage8/live_demo`) | `--output-dir outputs/stage8/run2` |
| `--max-runtime-seconds N` | YAML limit (currently unlimited) | `--max-runtime-seconds 60` |
| `--max-frames-per-camera N` | YAML limit (currently unlimited) | `--max-frames-per-camera 100` |
| `--no-display` | Disable GUI; use runtime/frame limits to stop headless runs | `--no-display --max-runtime-seconds 60` |
| `--dry-run` | Validate/print selected config without cameras or models | `--dry-run` |

Shared overlap arguments are listed in section 5. `-h` / `--help` prints help.

<!-- END AUTO-GENERATED -->

## 4. Replay your recorded live session

Use the actual session folder, not the parent `recordings/handoff/`:

```powershell
python main_handoff_replay.py `
  --recording recordings/handoff/session_YOUR_SESSION `
  --no-overlap --output-dir outputs/handoff/replay_run1 --show
```

For a recorded overlap setup, replace `--no-overlap` with e.g.
`--overlap "PHONE,LAPTOP"`. First check its inputs with `--dry-run` if needed.
Camera IDs come from the recording manifest. Replay uses recorded event times,
not elapsed inference time; its preview runs at processing speed.

<!-- AUTO-GENERATED: argument reference from pipeline/handoff_replay.py and association/topology.py -->

| Argument | Default / meaning | Example |
|---|---|---|
| `--recording PATH` | Required; session folder containing manifest/timeline/frames | `--recording recordings/handoff/session_YOUR_SESSION` |
| `--config PATH` | `configs/camera_demo.yaml`; model/association/display settings | `--config configs/camera_demo.yaml` |
| `--mode {solo,multi_ip}` | YAML active mode; loads config, not new live cameras | `--mode solo` |
| `--output-dir PATH` | `outputs/handoff/replay` | `--output-dir outputs/handoff/replay_run2` |
| `--show` | Off; display at processing speed; Q/Escape stops | `--show` |
| `--dry-run` | Validate manifest/config without inference | `--dry-run` |

Shared overlap arguments are listed in section 5. `-h` / `--help` prints help.

<!-- END AUTO-GENERATED -->

Replay and live runs do not have WiseNET's existing-directory protection: always
choose a new output folder to preserve evidence. Raw session replay does not
automatically calculate accuracy; that requires independent human labels.

## 5. Shared overlap and reconciliation arguments

<!-- AUTO-GENERATED: argument reference from association/topology.py -->

| Argument | Meaning | Example |
|---|---|---|
| `--overlap A,B ...` | Select hybrid policy and replace configured overlap permissions | WiseNET: `--overlap "2,3" "4,5"`; live/replay: `--overlap "PHONE,LAPTOP"` |
| `--no-overlap` | Clear configured pairs and force strict handoff | `--no-overlap` |
| `--no-reconcile` | Disable reconsidering already-issued IDs; arrival-time overlap matching stays enabled | `--overlap "2,3" "4,5" --no-reconcile` |

<!-- END AUTO-GENERATED -->

Do not combine `--overlap` and `--no-overlap`. If live `--association-mode` is
also supplied, it must agree: `hybrid` with pairs, `handoff` with `--no-overlap`.
Without either overlap option, the YAML policy/pairs remain in effect.

Use comma-separated pairs, not Python lists such as `[2,3]`. Permissions are
bidirectional but not transitive: three simultaneous views sharing an ID need
all three pairs, e.g. `--overlap "2,3" "2,4" "3,4"`. Only list cameras that are
selected/enabled and genuinely overlap. All other pairs retain strict handoff.

Reconciliation is enabled by default in hybrid. It requires five unambiguous,
mutual-best comparisons with fresh evidence on both sides over at least one
second, an acceptable distance, and no current/historical same-camera conflict.
The smaller global ID survives; old logs/screenshots are not rewritten.
Appearance alone cannot guarantee that similarly dressed people are distinct.

## 6. YAML-only settings and output files

The WiseNET and recording replay CLIs do **not** have live flags such as
`--device` or `--checkpoint`: edit their `--config` YAML instead. Settings without
CLI overrides include:

- `models`: confidence, image size, embedding interval/history, appearance
  thresholds, sample counts, ambiguity margin, grace/travel time, directed
  `allowed_transitions`, overlap/reconciliation confirmation settings.
- `display`: `tile_width`, `tile_height`, `info_panel_height`, `columns`,
  `control_bar_height`. Change these to resize the live/replay layout.
- Live camera entries: enabled state, URLs, webcam capture width/height/FPS,
  timeouts, reconnect delay. Camera IDs must match CLI overrides/pairs.
- Live `runtime`: startup timeout, minimum active cameras, cached-observation
  freshness, output/record directories, and stopping limits.

Keep credentials out of committed YAML and command examples. CLI URL overrides
are omitted from saved public session settings, but remain in shell history.

| Output | Purpose |
|---|---|
| `assignments.jsonl` | Camera-local tracks and assigned global IDs |
| `handoff_events.jsonl` | Accepted departure/arrival recoveries |
| `overlap_events.jsonl` | Accepted simultaneous-view arrival matches |
| `merge_events.jsonl` | Accepted reconciliation/merge events |
| `association_decisions.jsonl` | Distances, thresholds, rejection/waiting reasons |
| `settings.json`, `summary.json` | Run configuration, counts and provenance |
| `screenshots/` | Saved dashboard evidence; replay also saves event snapshots |
| `dashboard.mp4` | Offline replay dashboard video; not saved automatically by live mode |
| `processed_frames.jsonl` | Offline processed-frame coverage, including empty views |
| `evaluation.json` | WiseNET diagnostic scores and original/resolved ID audits |

The screenshot key/button is supported in **live mode**. Offline previews have
Q/Escape handling and automatic event screenshots; their drawn save button is
not wired to manual screenshot input.

## 7. Optional earlier-stage commands and checks

These are diagnostics or training tools, not steps required before each demo:

```powershell
# Full software regression suite; no live cameras needed.
python -m pytest
python -m ruff check .

# Test association independently of detector, model, and cameras.
python -m pytest tests/unit/test_reconciliation.py --no-cov -q
python -m pytest tests/unit/test_handoff.py tests/unit/test_hybrid.py --no-cov -q

# Verify a checkpoint on an optional real person crop.
python scripts/verify_reid_checkpoint.py --image path/to/person_crop.jpg --device auto

# Market-1501 crop retrieval evaluation, distinct from scene-level handoff scoring.
python scripts/evaluate_market1501.py --device auto
python scripts/evaluate_market1501.py --reuse-embeddings

# Only if you need to download/train again; not required for the existing checkpoint.
python scripts/download_market1501.py --output-dir data/reid/market1501
python scripts/train_market1501.py --dry-run
python scripts/train_market1501.py --smoke
python scripts/train_market1501.py --config configs/train_market1501.yaml
```

Training supports `--config`, `--smoke`, and `--dry-run`. Checkpoint verification
supports `--checkpoint`, `--image`, and `--device`. Market evaluation supports
`--dataset-root`, `--checkpoint`, `--output-dir`, `--device`, `--batch-size`,
`--max-rank`, `--retrieval-topk`, `--example-count`, `--query-chunk-size`, and
`--reuse-embeddings`. Use each script's `--help` for defaults and full syntax.

For isolated detector/embedding debugging:

```powershell
python scripts/run_single_camera.py --source path/to/input.mp4 --display
python scripts/run_ip_webcams.py --config configs/ip_webcams.local.yaml
python scripts/run_reid_diagnostic.py --source path/to/two_people.mp4 --display
python scripts/smoke_handoff.py --help
```

The IP-webcam diagnostic uses its own YAML and does not assign cross-camera IDs.
The handoff smoke needs existing smoke footage and uses an artificial schedule;
it verifies wiring, not accuracy. EPFL/WILDTRACK download/replay/evaluation commands
in the README are optional historical overlap diagnostics, not the current
separate-location demonstration workflow.

Get the exact current CLI help whenever unsure:

```powershell
python main_wisenet_test.py --help
python main_camera_demo.py --help
python main_handoff_replay.py --help
```
