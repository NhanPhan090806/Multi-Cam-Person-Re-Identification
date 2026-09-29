# Person Multi-Camera Tracking & Re-Identification

## 1. Project Requirement

> **Person Re-identification: nhận diện lại một người qua nhiều camera.**

The project aims to build a system that can recognize the same person across different camera views.

The system should eventually demonstrate:

* Person detection
* Single-camera tracking
* Person Re-Identification (Re-ID)
* Cross-camera identity association
* Global identity assignment
* Multiple real camera inputs
* Optional simulation/demo input

The **main research and evaluation should use existing public datasets**.

Simulation is **not** intended to be a separate dataset-generation or research task. It is only an additional way to demonstrate the finished system.

---

# 2. Project Scope

The project is divided into two parts:

### Main research system

```text
Existing Dataset
      ↓
Detection / Tracking
      ↓
Person Re-ID
      ↓
Cross-Camera Association
      ↓
Evaluation
```

### Demonstration system

```text
Real IP Cameras
      │
      ├──→ Same inference pipeline
      │
Simulation / Virtual Cameras
      │
      └──→ Same inference pipeline
```

The research system provides quantitative results.

The demonstration system shows that the trained pipeline can operate on actual camera streams and, optionally, simulated camera views.

---

# 3. Overall Architecture

```text
                 ┌─────────────────────┐
                 │     Input Sources   │
                 └──────────┬──────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
        Dataset        Real IP Cam     Simulation
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │ Person Detector │
                  │      YOLO       │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Single-Camera   │
                  │    Tracking     │
                  │ ByteTrack /     │
                  │   BoT-SORT      │
                  └────────┬────────┘
                           │
                     Person Crops
                           │
                           ▼
                  ┌─────────────────┐
                  │   Re-ID Model   │
                  │ Appearance      │
                  │   Embedding     │
                  └────────┬────────┘
                           │
                      Embeddings
                           │
                           ▼
                  ┌─────────────────┐
                  │ Cross-Camera    │
                  │   Association   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Global Identity │
                  │    Assignment   │
                  └─────────────────┘
```

The key distinction is:

```text
Local ID
    =
identity maintained within one camera

Global ID
    =
identity maintained across multiple cameras
```

Example:

```text
Camera 1 → Local ID 3 → Global ID 17
Camera 2 → Local ID 8 → Global ID 17
Camera 3 → Local ID 2 → Global ID 17
```

---

# 4. Main Research Data

## 4.1 Use Existing Datasets

The project should **not require creating a synthetic dataset**.

Instead, use established public datasets for training and evaluation.

Possible datasets include:

### Market-1501

A standard person Re-ID benchmark containing 1,501 identities captured by six cameras, with 32,668 labeled images. It provides a standard training/testing protocol and is useful for establishing the initial Re-ID baseline.

### MSMT17

A larger and more varied Re-ID dataset containing 4,101 identities captured by 15 cameras, including indoor and outdoor cameras and different time/weather conditions. It can be used for a more challenging experiment.

### MTA

MTA is particularly relevant because it contains multi-camera tracking videos and annotations as well as a dedicated Re-ID dataset. Its short extracted version is much smaller than the full dataset and may be more practical for experimentation.

### Other datasets

Depending on availability and licensing:

* CUHK03
* DukeMTMC-reID where legally/technically available
* MARS
* Other established MTMC/Re-ID benchmarks

The exact dataset combination should be selected based on:

1. Dataset availability
2. Licensing
3. GPU/storage requirements
4. Whether camera IDs are available
5. Whether person identities are shared across cameras
6. Whether video/track information is available

---

# 5. Dataset Roles

Not every dataset needs to perform every role.

A clean setup is:

```text
Re-ID Dataset
     ↓
Train / fine-tune Re-ID model
```

and:

```text
Multi-camera Dataset
     ↓
Evaluate cross-camera identity association
```

and, if necessary:

```text
MOT Dataset
     ↓
Evaluate single-camera tracking
```

For example:

```text
Market-1501
    ↓
Re-ID baseline training

MTA / another MTMC dataset
    ↓
Multi-camera evaluation

Real IP cameras
    ↓
Final demonstration
```

This separation prevents the project from becoming unnecessarily complicated.

---

# 6. Person Detection

The detector answers:

> "Where are the people in this frame?"

Input:

```text
Image
```

Output:

```text
Person
├── bounding box
├── confidence
└── class = person
```

## Recommended approach

Use a pretrained YOLO model.

Do **not** train a detector from scratch initially.

The detector is infrastructure for the Re-ID system rather than the primary research contribution.

The exact YOLO model size should be selected according to:

* detection quality
* FPS
* GPU memory
* number of simultaneous cameras

Since the target machine is a GTX 1650, start with a relatively small model.

---

# 7. Single-Camera Tracking

The tracker answers:

> "Is this person in the current frame the same person I saw a few frames ago?"

Example:

```text
Frame 001 → Person ID 4
Frame 002 → Person ID 4
Frame 003 → Person ID 4
```

This is **local tracking**, not Re-ID.

## Candidate trackers

### ByteTrack

Use as the first baseline.

Advantages:

* relatively lightweight
* simple pipeline
* good baseline for static cameras
* does not require a separate appearance model

### BoT-SORT

Use as an optional comparison.

BoT-SORT can incorporate appearance/Re-ID information and camera-motion compensation. Ultralytics currently supports both ByteTrack and BoT-SORT for tracking.

Initial pipeline:

```text
YOLO
 ↓
ByteTrack
 ↓
Local Track IDs
```

Optional comparison:

```text
YOLO
 ↓
BoT-SORT
 ↓
Local Track IDs
```

The project does **not** need to train the tracker from scratch.

---

# 8. Person Re-Identification

This is the **main ML/research component**.

The Re-ID model answers:

> "Is this person the same person that appeared in another camera?"

A detected person crop is converted into an embedding:

```text
Person Crop
     ↓
Re-ID Network
     ↓
Embedding Vector
```

Example:

```text
Person A - Camera 1
        ↓
      Vector A

Person A - Camera 2
        ↓
      Vector B

distance(A, B) → small
```

For different people:

```text
Person A → Vector A
Person B → Vector B

distance(A, B) → larger
```

---

# 9. Re-ID Model Architecture

## Baseline

Start with a CNN-based architecture.

Example:

```text
Person Crop
    ↓
ResNet-style CNN
    ↓
Global Pooling
    ↓
Embedding Layer
    ↓
Feature Vector
```

CNN is sufficient for the initial project.

Attention is **optional**, not a requirement.

Possible research extension:

```text
CNN
vs
CNN + Attention
```

This gives a clear baseline-versus-improvement experiment without making the initial system unnecessarily complicated.

---

# 10. Re-ID Training

The Re-ID model is the main component that should be trained/fine-tuned.

A practical approach:

```text
ImageNet-pretrained CNN
        ↓
Re-ID architecture
        ↓
Train / fine-tune on Re-ID dataset
        ↓
Person embeddings
```

Possible objectives:

### Classification loss

Train the model to classify identities in the training set.

### Triplet loss

Encourage:

```text
same person
    ↓
closer embeddings

different person
    ↓
farther embeddings
```

A possible baseline is:

```text
Cross-Entropy Loss
+
Triplet Loss
```

The exact loss configuration should be finalized after the baseline is running.

---

# 11. Important Training Clarification

The project uses **staged development**, but it does not require staged training of every component.

### YOLO

```text
Pretrained
↓
Use directly
```

Fine-tune only if necessary.

### Tracker

```text
ByteTrack / BoT-SORT
↓
Use directly
```

No custom tracker training initially.

### Re-ID

```text
Pretrained backbone
↓
Fine-tune / train Re-ID head
```

This is the main training component.

### Cross-camera association

```text
Embeddings
↓
Similarity / association algorithm
```

Initially this does not require neural-network training.

---

# 12. Cross-Camera Association

After local tracking and Re-ID:

```text
Camera 1
Local ID 3
Embedding A

Camera 2
Local ID 8
Embedding B
```

Calculate similarity:

```text
cosine_similarity(A, B)
```

If the similarity is sufficiently high:

```text
Camera 1 / Local ID 3
          │
          ▼
      Global ID 17
          ▲
          │
Camera 2 / Local ID 8
```

Otherwise, the system can create a new global identity.

---

# 13. Baseline Association Algorithm

Start simple:

```text
Re-ID embedding
       ↓
Cosine similarity
       ↓
Threshold
       ↓
Same identity / New identity
```

Do not immediately build a sophisticated graph neural network or transformer-based association system.

The baseline must work first.

---

# 14. Possible Association Improvements

After the baseline works, additional information can be incorporated.

## 14.1 Tracklet-Level Embedding

Instead of relying on one frame:

```text
Frame 1 → embedding
Frame 2 → embedding
Frame 3 → embedding
Frame 4 → embedding
```

Aggregate:

```text
Tracklet
   ↓
Feature aggregation
   ↓
Stable embedding
```

Compare:

```text
Single-frame Re-ID
vs
Tracklet-level Re-ID
```

This is a potentially useful research experiment.

---

## 14.2 Temporal Constraints

If a person leaves Camera A:

```text
Camera A
    ↓
person disappears
    ↓
reasonable travel time
    ↓
Camera B
```

The association system can use this temporal information.

An identity appearing in Camera B immediately after leaving a physically distant Camera A may receive a lower association score.

---

## 14.3 Camera Topology

Represent possible camera transitions:

```text
Camera A ─── Camera B
    │
    └────── Camera C
```

This can reduce obviously impossible cross-camera matches.

---

# 15. Real IP Camera Demonstration

The final system should support multiple real camera streams.

Possible sources:

```text
IP Camera 1 → RTSP / HTTP
IP Camera 2 → RTSP / HTTP
IP Camera 3 → RTSP / HTTP
```

or:

```text
USB Webcam 1
USB Webcam 2
USB Webcam 3
```

OpenCV can initially handle the input streams.

Each stream should be normalized into a common representation:

```python
FramePacket(
    camera_id,
    frame_id,
    timestamp,
    frame
)
```

This keeps the downstream pipeline independent of the camera source.

---

# 16. Multi-Camera Processing

Each camera should maintain its own local tracking state:

```text
Camera 1 → Tracker 1
Camera 2 → Tracker 2
Camera 3 → Tracker 3
```

The global association system then connects the local identities:

```text
Tracker 1
   │
Tracker 2 ───→ Global Association
   │
Tracker 3
```

This is important because:

> **Local tracking and global Re-ID are two different problems.**

---

# 17. Simulation / Virtual Camera Demonstration

Simulation is **not a required research dataset**.

It is only a demonstration input source.

The goal is simply to show:

```text
Virtual Camera 1
        ↓
Person appears
        ↓
Virtual Camera 2
        ↓
Same Global ID
```

A lightweight virtual environment is sufficient.

Possible options:

* Blender
* BlenderProc
* lightweight 3D scene
* prerecorded multi-camera video

The project does **not** need:

* ROS2
* Gazebo
* Isaac Sim
* complex physics
* robotics simulation
* synthetic dataset generation

unless a later requirement specifically demands them.

---

# 18. Same Pipeline for Dataset, Camera, and Simulation

The system should be designed around a common input interface:

```text
                    Input Adapter
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    Dataset          Real IP Cam      Simulation
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                     YOLO
                         ↓
                     Tracker
                         ↓
                    Person Crop
                         ↓
                      Re-ID
                         ↓
                  Global Association
                         ↓
                    Global IDs
```

This means simulation does not require a separate AI pipeline.

It simply provides frames to the same system.

---

# 19. Development Stages

## Stage 0 — Environment

Install and verify:

```text
Python
PyTorch
OpenCV
Ultralytics
Re-ID framework/model
CUDA
```

Verify GPU compatibility with the GTX 1650.

---

## Stage 1 — Re-ID Dataset

Select one established Re-ID dataset.

Recommended initial candidate:

```text
Market-1501
```

because it is relatively manageable and provides a standard multi-camera Re-ID benchmark.

Train/fine-tune the first Re-ID model.

---

## Stage 2 — Re-ID Evaluation

Evaluate:

```text
Rank-1
Rank-5
Rank-10
mAP
```

Make sure the Re-ID model works independently before integrating the entire multi-camera system.

---

## Stage 3 — Person Detection

Run:

```text
YOLO
 ↓
Person bounding boxes
```

Test on:

* dataset frames
* recorded video
* real camera

---

## Stage 4 — Single-Camera Tracking

Build:

```text
YOLO
 ↓
ByteTrack
 ↓
Local Track IDs
```

Evaluate:

* tracking stability
* ID switches
* occlusion
* entering/leaving the scene

---

## Stage 5 — Integrate Re-ID

Build:

```text
YOLO
 ↓
ByteTrack
 ↓
Person Crop
 ↓
Re-ID
 ↓
Embedding
```

---

## Stage 6 — Cross-Camera Association

Build:

```text
Camera 1 Tracklets
        ↓
    Embeddings
        ↓
  Global Matcher
        ↑
    Embeddings
        ↑
Camera 2 Tracklets
```

Start with:

```text
Cosine similarity
+
threshold
```

---

## Stage 7 — Multi-Camera Dataset Evaluation

Use a dataset with actual multi-camera sequences/annotations.

For example, MTA provides multi-camera tracking annotations and Re-ID data, making it useful for testing the complete concept rather than only isolated image Re-ID.

---

## Stage 8 — Real Camera Demonstration

Connect:

```text
2 IP cameras
```

first.

Then increase to:

```text
3 cameras
```

if the hardware can maintain an acceptable FPS.

Demonstrate:

```text
Person enters Camera 1
        ↓
Global ID 17

Person leaves Camera 1
        ↓
Person appears Camera 2
        ↓
Global ID 17
```

---

## Stage 9 — Simulation Demonstration

Only after the real/dataset pipeline works.

Create or use a simple virtual scene:

```text
Camera A
Camera B
Several people
```

Demonstrate the same global identity association.

The simulation is **presentation/demo material**, not a new research dataset.

---

# 20. Evaluation

Evaluation should separate the different components.

## Re-ID

Use:

* Rank-1
* Rank-5
* Rank-10
* mAP

These are standard metrics for person Re-ID.

---

## Tracking

Possible metrics:

* IDF1
* MOTA
* HOTA
* ID switches

---

## Cross-Camera Association

Measure:

* correct identity matches
* false matches
* missed matches
* global ID consistency
* identity switches

---

## System Performance

Measure:

```text
FPS
Latency
GPU memory
CPU usage
Number of simultaneous cameras
```

The GTX 1650 should be treated as an actual system constraint.

---

# 21. Final Demonstration

The final demonstration should contain two or three parts.

## Demo 1 — Research Dataset

Show:

```text
Dataset
 ↓
Re-ID
 ↓
Cross-camera matching
 ↓
Metrics
```

This demonstrates quantitative performance.

---

## Demo 2 — Real IP Cameras

Show:

```text
Camera 1 ─┐
Camera 2 ─┼→ Detection → Tracking → Re-ID → Global IDs
Camera 3 ─┘
```

Example:

```text
Camera 1:
Person #17

Camera 2:
Person #17

Camera 3:
Person #17
```

This demonstrates practical multi-camera operation.

---

## Demo 3 — Simulation

Optional:

```text
Virtual Camera 1
       ↓
Virtual Camera 2
       ↓
Same global identity
```

The purpose is to make the multi-camera concept easy to visualize.

It is **not** intended to prove that the simulator is physically realistic.

---

# 22. Research Experiments

Once the baseline works, choose one or two focused experiments.

## Experiment A — Re-ID Architecture

```text
CNN baseline
vs
CNN + Attention
```

---

## Experiment B — Frame-Level vs Tracklet-Level

```text
Single-frame embedding
vs
Tracklet aggregation
```

---

## Experiment C — Association Strategy

```text
Appearance only
vs
Appearance + temporal constraints
```

---

## Experiment D — Dataset Generalization

For example:

```text
Train on Dataset A
        ↓
Test on Dataset B
```

This tests how well the Re-ID representation generalizes to a different camera environment.

---

# 23. Minimum Viable Project

If time becomes limited, implement:

```text
Existing Re-ID Dataset
        ↓
Re-ID Model
        ↓
Evaluation
```

then:

```text
YOLO
 ↓
ByteTrack
 ↓
Person Crop
 ↓
Re-ID
 ↓
Cosine Similarity
 ↓
Global ID
```

and finally:

```text
2 real IP cameras
```

for the live demonstration.

Simulation remains optional.

This is already enough to address the core requirement.

---

# 24. Out of Scope

The following should **not** be part of the initial project:

* Creating a custom synthetic Re-ID dataset
* Building a complete simulation environment
* ROS2
* Gazebo
* Isaac Sim
* Complex physics
* 3D reconstruction
* Full camera calibration
* Distributed multi-machine processing
* Detector training from scratch
* Tracker training from scratch
* Giant end-to-end MTMC Transformer
* Custom CUDA kernels
* Complicated camera synchronization

These can be considered only if the project requirements later demand them.

---

# 25. Final Architecture

The intended research system is:

```text
              EXISTING DATASET
                     │
                     ▼
              Re-ID Training
                     │
                     ▼
              ┌──────────────┐
              │  Re-ID Model │
              └──────┬───────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │ Multi-Camera Application │
        └────────────┬─────────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Camera 1   Camera 2   Camera 3
          │          │          │
          ▼          ▼          ▼
        YOLO       YOLO       YOLO
          │          │          │
          ▼          ▼          ▼
       Tracker    Tracker    Tracker
          │          │          │
          └──────────┼──────────┘
                     │
                     ▼
               Person Crops
                     │
                     ▼
                  Re-ID
                     │
                     ▼
              Embeddings
                     │
                     ▼
          Cross-Camera Association
                     │
                     ▼
                Global IDs
```

Additional inputs:

```text
Existing Dataset ────────────────┐
Real IP Cameras ─────────────────┼→ Same Pipeline
Simulation / Virtual Cameras ────┘
```

---

# 26. Project Philosophy

The project should follow:

```text
Use existing data
       ↓
Build a working baseline
       ↓
Measure it
       ↓
Find one weakness
       ↓
Improve one component
       ↓
Measure again
       ↓
Demonstrate on real cameras
```

The project is **not**:

```text
Build simulator
+
Generate dataset
+
Train detector
+
Train tracker
+
Train Re-ID
+
Build giant model
+
Build robotics stack
```

The core research question remains:

> **Can the system reliably recognize and maintain the identity of the same person across different camera views?**

Therefore:

* **Existing datasets** → research/training/evaluation
* **YOLO** → detection infrastructure
* **ByteTrack/BoT-SORT** → local tracking infrastructure
* **Re-ID model** → main learned component
* **Cross-camera association** → global identity logic
* **Real IP cameras** → practical demonstration
* **Simulation** → optional visual demonstration

This keeps the project focused on **Person Re-Identification**, while still producing a convincing multi-camera system demo.
