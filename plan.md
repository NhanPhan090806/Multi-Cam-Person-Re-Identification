# Multi-Camera Person Tracking and Re-Identification

## 1. Objective

Build a system that detects and tracks people in multiple camera views and assigns the same **global identity** to the same person across those views.

The lecturer's requirement is:

> Person Re-identification: recognize the same person across multiple cameras.

This is an engineering group project, not a research paper. It does not need a novel model. It must demonstrate sound system design, model training, integration, evaluation, and a convincing multi-camera demo.

## 2. Required Deliverables

1. A reproducible Kaggle or Colab notebook that fine-tunes a person Re-ID model.
2. Standard Market-1501 Re-ID results: Rank-1, Rank-5, Rank-10, and mAP.
3. A local application that performs person detection, per-camera tracking, Re-ID embedding extraction, cross-camera association, and global ID assignment.
4. A reproducible multi-camera evaluation/demo using WILDTRACK.
5. A live or prerecorded demonstration using two real cameras.
6. A short report covering architecture, implementation, results, limitations, and team contributions.

An optional simulation demonstration may be included after all required work is complete.

## 3. Frozen Scope

### Required

- Two overlapping camera views.
- Pretrained YOLO detector.
- ByteTrack for local tracking.
- OSNet-x0.25 for person Re-ID.
- Market-1501 for Re-ID training and evaluation.
- WILDTRACK for multi-camera integration and evaluation.
- Tracklet-level appearance aggregation.
- Cosine distance and Hungarian assignment for cross-camera matching.
- Consistent displayed global IDs across views.

### Optional

- A third camera after the two-camera system is stable.
- BoT-SORT as a local-tracker comparison.
- A lightweight simulation or virtual-camera demonstration.
- WILDTRACK calibration and ground-plane geometry.

### Out of Scope

- Training YOLO or a tracker from scratch.
- Non-overlapping camera topology and travel-time learning.
- MSMT17, MTA, DukeMTMC, or another large benchmark during the baseline.
- Creating a synthetic training dataset.
- CNN-versus-Transformer or attention research.
- ROS2, Gazebo, Isaac Sim, complex physics, or distributed processing.
- Face recognition or identifying a person's real-world name.

The baseline must be completed before optional work starts.

## 4. Problem Definition

Each camera produces its own **local IDs**:

```text
Camera 1 -> Local ID 3
Camera 2 -> Local ID 8
```

Cross-camera association connects them to one **global ID**:

```text
Camera 1 / Local ID 3 --+
                         +--> Global ID 17
Camera 2 / Local ID 8 --+
```

A global ID means “the system believes these observations belong to the same person in this camera network.” It is not a legal or biometric identity.

## 5. Fixed Assumptions

1. The primary system uses overlapping cameras.
2. Frames are synchronized or approximately synchronized.
3. A person may appear in several cameras simultaneously.
4. Clothing does not change during a sequence.
5. Each camera owns a separate ByteTrack instance.
6. All cameras share one Re-ID model.
7. The first live demo uses two fixed cameras.
8. Cloud GPU is used for training; the GTX 1650 is used for local inference.

Overlapping views satisfy the assignment without introducing the separate problem of learning travel times between non-overlapping locations.

## 6. Verified Datasets

### 6.1 Market-1501: Re-ID Training and Evaluation

Use Market-1501 to fine-tune OSNet, evaluate image-level Re-ID, and select the association threshold using validation identities taken from the training split.

Verified source:

- Kaggle: <https://www.kaggle.com/datasets/pengcw1/market-1501>
- Kaggle handle: `pengcw1/market-1501`
- Approximate download: 146 MB
- Public and downloadable using KaggleHub

Locally verified contents:

| Split | Images | Identities |
|---|---:|---:|
| Training | 12,936 | 751 |
| Query | 3,368 | 750 |
| Gallery | 15,913 | 751 including distractors |

The downloaded archive was successfully parsed by Torchreid.

Kaggle/Colab download:

```python
!pip install -q kagglehub

import kagglehub

market_root = kagglehub.dataset_download(
    "pengcw1/market-1501",
    output_dir="/content/data/market1501",
)
print(market_root)
```

On Kaggle, use `/kaggle/working/data/market1501`.

Expected structure:

```text
Market-1501-v15.09.15/
├── bounding_box_train/
├── bounding_box_test/
└── query/
```

Rules:

- Use the original images, not background-modified derivatives.
- Keep the official query/gallery protocol for final metrics.
- Do not redistribute or commit the dataset.
- The Kaggle mirror has no clear license metadata; use it only for this non-commercial educational project and cite the original Market-1501 paper.

### 6.2 WILDTRACK: Multi-Camera Integration and Evaluation

Use WILDTRACK to replay real synchronized views, test independent trackers, evaluate cross-camera association against persistent person IDs, and optionally use calibration later.

References:

- Project page: <https://www.epfl.ch/labs/cvlab/data/data-wildtrack/>
- Official toolkit: <https://github.com/Chavdarova/WILDTRACK-toolkit>

Programmatic mirror:

- Hugging Face: <https://huggingface.co/datasets/disl/my_dataset>
- Repository ID: `disl/my_dataset`
- Ungated public download
- Approximate size: 7.67 GB

Locally verified:

- 1920x1080 images;
- seven synchronized views;
- JSON frame annotations;
- persistent `personID`;
- per-camera bounding boxes under `views`;
- camera calibration XML files.

One downloaded annotation contained 38 people with seven camera-view entries and parsed successfully.

Kaggle/Colab download:

```python
!pip install -q huggingface_hub

from huggingface_hub import snapshot_download

wildtrack_root = snapshot_download(
    repo_id="disl/my_dataset",
    repo_type="dataset",
    local_dir="/content/data/wildtrack",
    allow_patterns=[
        "Wildtrack_dataset_full/Wildtrack_dataset/Image_subsets/**",
        "Wildtrack_dataset_full/Wildtrack_dataset/annotations_positions/**",
        "Wildtrack_dataset_full/Wildtrack_dataset/calibrations/**",
    ],
)
print(wildtrack_root)
```

Start with cameras `C1` and `C2`. Add `C3` only after the two-camera pipeline works.

Rules:

- The Hugging Face repository is a mirror, not the authoritative publisher.
- Cite the original WILDTRACK paper and project.
- Do not commit or redistribute the images.
- Recheck the authoritative source terms before publishing dataset-derived media outside the class submission.

### 6.3 Real Camera Data

Use two USB webcams, phones acting as IP cameras, prerecorded files, or RTSP cameras with overlapping views.

- Obtain consent from everyone recorded.
- Do not record uninvolved people in public spaces.
- Delete recordings when they are no longer required.
- Treat team recordings as demonstration data, not a training benchmark.

## 7. Why Other Datasets Are Deferred

- **MSMT17:** much larger and harder than necessary for the baseline.
- **MTA:** relevant, but access requires contacting the maintainer, accepting research-only conditions, and owning GTA V; the full archive is also large.
- **DukeMTMC:** avoided because of access and redistribution concerns.
- **Custom synthetic data:** unnecessary because simulation is only a final visual input source.

## 8. Architecture

```text
                    Input adapters
       +----------------+----------------+
       |                |                |
       v                v                v
 WILDTRACK frames   Real cameras   Simulation (optional)
       |                |                |
       +----------------+----------------+
                        |
       +----------------+----------------+
       |                                 |
       v                                 v
 Camera 1: YOLO -> ByteTrack   Camera 2: YOLO -> ByteTrack
       |                                 |
       v                                 v
 Local tracklets + crops        Local tracklets + crops
       |                                 |
       +---------------+-----------------+
                       v
                 Shared OSNet
                       v
             Tracklet embeddings
                       v
       Cosine distance + Hungarian matching
                       v
               Global ID registry
                       v
          Visualization and evaluation
```

## 9. Fixed Technology Choices

| Component | Choice |
|---|---|
| Language | Python 3.12 |
| Deep learning | PyTorch |
| Detector | Ultralytics YOLO26n |
| Local tracker | ByteTrack |
| Re-ID framework | Torchreid |
| Re-ID model | OSNet-x0.25 |
| Re-ID input | 256x128 RGB crops |
| Similarity | Cosine distance |
| Assignment | Hungarian algorithm |
| Video input | OpenCV with FFmpeg |
| Re-ID metrics | Torchreid evaluator |
| MOT metrics | Motmetrics |
| Cloud training | Kaggle or Google Colab |
| Local GPU | NVIDIA GTX 1650 4 GB |

BoT-SORT may be tested later, but ByteTrack remains the baseline.

## 10. Common Input Interface

Every source produces the same object:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class FramePacket:
    camera_id: str
    frame_id: int
    timestamp: float
    frame: np.ndarray
```

Planned adapters:

- `WildtrackSource`
- `VideoFileSource`
- `RtspCameraSource`
- `WebcamSource`
- `SimulationSource` (optional)

Downstream components must not depend on where a frame originated.

## 11. Re-ID Training

### Model and Loss

- OSNet-x0.25 with ImageNet initialization
- 512-dimensional embedding
- 256x128 inputs
- Cross-entropy identity classification plus triplet loss

### Initial Configuration

| Setting | Initial value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Batch size | 32; reduce to 16 on memory error |
| Epochs | 60 maximum |
| Precision | Mixed precision when supported |
| Checkpoint | Best validation mAP |
| Random seed | Fixed and recorded |

First run a one-epoch smoke test. Adjust settings only after measuring it.

Create validation identities from the training split. Never tune thresholds or hyperparameters on the official query/gallery test result.

Save the final checkpoint as:

```text
checkpoints/osnet_x0_25_market1501_best.pth
```

Checkpoints must not be committed to Git.

## 12. Per-Camera Pipeline

Each camera owns:

```text
Camera state
├── input adapter
├── ByteTrack instance
├── frame counter
├── active local tracks
└── recent embeddings per local track
```

For every frame:

1. Read a `FramePacket`.
2. Run YOLO for the person class only.
3. Update that camera's ByteTrack instance.
4. Clip bounding boxes to the frame.
5. Reject invalid, tiny, or low-confidence crops.
6. Extract Re-ID embeddings periodically.
7. Update each local tracklet representation.

Never share one ByteTrack object between cameras.

## 13. Tracklet Embeddings

A single crop is noisy. For each local track:

1. L2-normalize every valid embedding.
2. Keep the latest 5-10 embeddings.
3. Average them.
4. L2-normalize the average.

Extract an embedding every few frames rather than every frame to reduce GPU cost.

## 14. Cross-Camera Association

At each association interval:

1. Collect active tracklets from each camera.
2. Build a cosine-distance matrix for every camera pair.
3. Reject pairs with excessive timestamp difference, too few valid crops, or appearance distance above the calibrated threshold.
4. Run Hungarian one-to-one assignment.
5. Merge accepted tracks into the same global identity.
6. Assign new global IDs to unmatched tracks.
7. Keep inactive identities for a short configurable TTL.

Because cameras overlap, one global identity may be active in several cameras simultaneously.

Thresholds must be selected using validation identities or a WILDTRACK development split, not the final test result.

```text
GlobalIdentity
├── global_id
├── member tracks: (camera_id, local_id)
├── aggregated embedding
├── first_seen
├── last_seen
└── active cameras
```

## 15. Evaluation

### Re-ID on Market-1501

- Rank-1
- Rank-5
- Rank-10
- mAP

Use the official query/gallery protocol.

### Local Tracking on WILDTRACK

Evaluate each selected camera:

- IDF1
- MOTA
- ID switches

### Cross-Camera Association

Use WILDTRACK `personID` to calculate:

- association precision;
- association recall;
- association F1;
- global ID switches;
- global ID fragmentation.

Distinguish detector/tracker errors from association errors.

### System Performance

Measure latency, FPS, GPU memory, CPU use, dropped frames, and two-camera throughput. Record image size, camera count, model, and hardware with every result.

## 16. Development Stages

### Stage 0: Repository and Environment

- Initialize Git and add `.gitignore`.
- Record dependencies.
- Verify CUDA, YOLO, ByteTrack, Torchreid, OpenCV, and FFmpeg.

Exit: all smoke tests pass.

### Stage 1: Market-1501 Training Notebook

- Download the dataset in code.
- Run one smoke-training epoch.
- Run full cloud training.
- Save the best checkpoint and training curves.

Exit: notebook runs from a clean Kaggle/Colab session.

### Stage 2: Re-ID Evaluation

- Extract query/gallery embeddings.
- Calculate Rank-k and mAP.
- Save example good and bad retrievals.

Exit: metrics and visual examples are saved.

### Stage 3: Single-Camera Pipeline

- YOLO26n detection.
- ByteTrack local IDs.
- Valid person crops.
- Video visualization.

Exit: one video runs without tracker-state or crop errors.

### Stage 4: Re-ID Integration

- Load the trained OSNet checkpoint.
- Extract embeddings from track crops.
- Maintain tracklet averages.

Exit: same-track embeddings are more similar than different-track embeddings in a small diagnostic.

### Stage 5: WILDTRACK Two-Camera Pipeline

- Download WILDTRACK in code.
- Start with C1 and C2.
- Read synchronized frames and annotations.
- Run independent trackers.

Exit: both views replay through the common pipeline.

### Stage 6: Global Association

- Build distance matrices.
- Add Hungarian assignment.
- Maintain the global registry.
- Tune the threshold on development data.

Exit: matched people show the same global ID and color across C1 and C2.

### Stage 7: Evaluation

- Generate Re-ID, local tracking, cross-camera association, and runtime results.

Exit: results are reproducible and exported for the report.

### Stage 8: Real Two-Camera Demo

- Connect two webcams, phones, files, or RTSP streams.
- Use overlapping views.
- Display local and global IDs.
- Test entry, exit, partial occlusion, and simultaneous visibility.

Exit: a stable demonstration is recorded.

### Stage 9: Optional Simulation Demo

Only begin after Stage 8 succeeds.

- Use Blender, prerecorded virtual-camera footage, or another lightweight scene.
- Provide two virtual camera streams through `SimulationSource`.
- Reuse the same detector, tracker, Re-ID, association, and visualization code.

Simulation is presentation material, not a training dataset, physics project, or separate AI pipeline.

## 17. Proposed Repository Structure

```text
Multi_Cam_ReID/
├── plan.md
├── README.md
├── requirements.txt
├── configs/
│   ├── train_market1501.yaml
│   └── system.yaml
├── notebooks/
│   ├── train_market1501.ipynb
│   └── evaluate_reid.ipynb
├── src/multicam_reid/
│   ├── inputs/
│   ├── detection/
│   ├── tracking/
│   ├── reid/
│   ├── association/
│   ├── evaluation/
│   └── visualization/
├── scripts/
│   ├── download_market1501.py
│   ├── download_wildtrack.py
│   ├── run_wildtrack.py
│   └── run_live.py
├── tests/
├── data/          # Git-ignored
├── checkpoints/   # Git-ignored
├── outputs/       # Git-ignored
└── runs/          # Git-ignored
```

## 18. Minimum Acceptable Submission

If time becomes limited, submit:

1. Market-1501 training notebook.
2. Rank-1 and mAP evaluation.
3. YOLO26n + ByteTrack on two WILDTRACK cameras.
4. OSNet tracklet embeddings.
5. Cosine distance + Hungarian global association.
6. A recorded two-camera real or prerecorded demo.
7. Results and limitations.

Simulation and the third camera are the first items to remove if the schedule slips.

## 19. Known Limitations

- Similar clothing can confuse appearance Re-ID.
- Market-1501 and WILDTRACK do not perfectly represent the team's cameras.
- Occlusion and small crops reduce embedding quality.
- Local ByteTrack errors propagate into global association.
- WILDTRACK does not test non-overlapping travel-time reasoning.
- The GTX 1650 may require lower resolution, less frequent embedding extraction, or sequential processing.
- The system cannot prove a person's real identity.

## 20. Data and Privacy Rules

- Do not commit datasets, weights, credentials, or RTSP URLs.
- Obtain consent from recorded participants.
- Avoid filming uninvolved people.
- Retain recordings only as long as needed.
- Document dataset citations and source URLs.
- Do not redistribute Market-1501 or WILDTRACK with the repository.

## 21. Acceptance Criteria

- [ ] Market-1501 downloads from code in a clean cloud notebook.
- [ ] OSNet-x0.25 training produces a reusable checkpoint.
- [ ] Rank-1, Rank-5, Rank-10, and mAP are reported.
- [ ] WILDTRACK downloads from code and annotations parse correctly.
- [ ] Two cameras run independent YOLO + ByteTrack pipelines.
- [ ] Tracklets produce aggregated embeddings.
- [ ] Cross-camera matching uses cosine distance and Hungarian assignment.
- [ ] The same person receives the same displayed global ID across views.
- [ ] WILDTRACK association results and runtime measurements are reported.
- [ ] A two-camera real or prerecorded demonstration is recorded.
- [ ] Setup and reproduction instructions are present.

Optional:

- [ ] A simulation source runs through the same pipeline.
- [ ] A third camera is demonstrated.
- [ ] BoT-SORT is compared with ByteTrack.

## 22. Final Summary

```text
Market-1501
    -> fine-tune and evaluate OSNet-x0.25

WILDTRACK C1 + C2
    -> YOLO26n
    -> independent ByteTrack instances
    -> person crops
    -> tracklet embeddings
    -> cosine distance
    -> Hungarian assignment
    -> global IDs
    -> quantitative evaluation

Two real cameras
    -> the same inference pipeline
    -> practical demonstration

Optional simulation
    -> SimulationSource adapter
    -> the same inference pipeline
    -> presentation-only demonstration
```

Guiding rule:

> Complete and measure a simple end-to-end system before adding another model, camera, dataset, or simulator.
