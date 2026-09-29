# Multi-Camera Person Re-Identification

Course project for assigning consistent global person IDs across overlapping camera views.

The frozen scope and design decisions are in [plan.md](plan.md).

## Current Status

- [x] Stage 0 repository and environment foundation
- [x] Market-1501 downloader and integrity inspection
- [x] Typed Torchreid training configuration
- [x] One-epoch smoke-training mode
- [x] Kaggle/Colab training notebook
- [ ] Full Market-1501 cloud training
- [ ] Standalone Re-ID evaluation
- [ ] YOLO + ByteTrack camera pipeline
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
& "C:\Users\ADMIN\ai_venv\Scripts\python.exe" -m pip install ".[dev]"
~~~

This repository lives under a Windows path containing Vietnamese characters. Use a normal install as shown above: current setuptools editable installs (`-e`) cannot encode that path into their `.pth` file. Reinstall after changing package source locally; pytest reads directly from `src/`.

## Verify the Repository

~~~powershell
& "C:\Users\ADMIN\ai_venv\Scripts\python.exe" -m pytest
& "C:\Users\ADMIN\ai_venv\Scripts\python.exe" -m ruff check .
~~~

## Download Market-1501

The downloader uses the public Kaggle mirror **pengcw1/market-1501**, extracts it into the Torchreid-compatible layout, and prints integrity statistics.

~~~powershell
& "C:\Users\ADMIN\ai_venv\Scripts\python.exe" scripts/download_market1501.py --output-dir data/reid/market1501
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
& "C:\Users\ADMIN\ai_venv\Scripts\python.exe" scripts/train_market1501.py --dry-run
~~~

Run the one-epoch smoke configuration:

~~~powershell
& "C:\Users\ADMIN\ai_venv\Scripts\python.exe" scripts/train_market1501.py --smoke
~~~

Run the full configuration:

~~~powershell
& "C:\Users\ADMIN\ai_venv\Scripts\python.exe" scripts/train_market1501.py
~~~

The default settings are in [configs/train_market1501.yaml](configs/train_market1501.yaml). Training outputs and model checkpoints are ignored by Git.

## Cloud Training

Open [notebooks/train_market1501.ipynb](notebooks/train_market1501.ipynb) in Kaggle or Colab. The notebook:

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
