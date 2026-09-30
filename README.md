# Multi-Camera Person Re-Identification

Course project for assigning consistent global person IDs across overlapping camera views.

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
- [ ] Cross-camera global association

## Environment

The verified local environment is Python 3.12. The package supports Python 3.10-3.12 so it can run on current Kaggle and Colab images.

- PyTorch 2.5.1 + CUDA 12.1
- torchvision 0.20.1
- OpenCV 4.10.0
- Ultralytics 8.4.64
- Torchreid 0.2.5

PyTorch should be installed separately using the build appropriate for the target GPU. On Kaggle and Colab, use the preinstalled PyTorch unless there is a demonstrated incompatibility.

Install this repository and its development tools:

~~~powershell
& python -m pip install ".[dev]"
~~~

This repository lives under a Windows path containing Vietnamese characters. Use a normal install as shown above: current setuptools editable installs (`-e`) cannot encode that path into their `.pth` file. Reinstall after changing package source locally; pytest reads directly from `src/`.

On Windows, installation also creates small `.exe` console launchers inside the
venv's `Scripts` directory. They are wrappers that invoke this package with the
venv's Python interpreter, not separately compiled applications. Every command
can alternatively be run through its corresponding script in `scripts/`.

## Verify the Repository

~~~powershell
& python -m pytest
& python -m ruff check .
~~~

## Download Market-1501

The downloader uses the public Kaggle mirror **pengcw1/market-1501**, extracts it into the Torchreid-compatible layout, and prints integrity statistics.

~~~powershell
& python scripts/download_market1501.py --output-dir data/reid/market1501
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
& python scripts/train_market1501.py --dry-run
~~~

Run the one-epoch smoke configuration:

~~~powershell
& python scripts/train_market1501.py --smoke
~~~

Run the full configuration:

~~~powershell
& python scripts/train_market1501.py
~~~

The default settings are in [configs/train_market1501.yaml](configs/train_market1501.yaml). Training outputs and model checkpoints are ignored by Git.

## Verify the Trained Re-ID Checkpoint

The epoch-60 checkpoint is the selected model: 64.6% mAP, 84.0% Rank-1, 94.1% Rank-5, and 96.6% Rank-10. Verify that it loads and produces a unit-normalized embedding:

~~~powershell
& python scripts/verify_reid_checkpoint.py
~~~

To encode a real OpenCV-compatible person crop instead of the synthetic smoke-test crop:

~~~powershell
& python scripts/verify_reid_checkpoint.py --image path/to/person_crop.jpg
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
& python scripts/evaluate_market1501.py --device cuda
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
& python scripts/evaluate_market1501.py --reuse-embeddings
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
& python scripts/run_single_camera.py `
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

The verified 60-frame diagnostic run maintained one local ID for all 60 tracked
frames, produced 60 valid crops, and processed 15.74 FPS on the GTX 1650. This is
a wiring smoke test using a simple generated video, not a claim about tracking
accuracy on difficult real footage.

## Cloud Training

Use [train_market1501_kaggle.ipynb](train_market1501_kaggle.ipynb) on Kaggle, or [train_market1501.ipynb](train_market1501.ipynb) when the repository is already available locally. The notebooks:

1. verifies GPU availability;
2. installs the repository package;
3. downloads and validates Market-1501;
4. resolves the training configuration;
5. exposes separate smoke and full-training switches.

Use a GPU runtime. Copy the best checkpoint out of the ephemeral runtime after training.

## Dataset Rules

- Do not commit or redistribute Market-1501 or WILDTRACK.
- Do not commit checkpoints, recordings, credentials, or RTSP URLs.
- Use the datasets for non-commercial educational work and cite their original papers.
