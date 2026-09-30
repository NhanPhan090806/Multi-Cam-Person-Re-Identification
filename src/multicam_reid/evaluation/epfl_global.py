"""Evaluate EPFL global identities using sparse calibrated ground positions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from multicam_reid.data.epfl_lab import (
    CALIBRATION_FILENAME,
    GROUND_TRUTH_FILENAME,
    EpflCameraCalibration,
    EpflPersonPosition,
    load_epfl_calibration,
    load_epfl_ground_truth,
    validate_epfl_cameras,
)

DEFAULT_STAGE6_RUN_DIR = Path("outputs/stage6/epfl_lab_full")
DEFAULT_OUTPUT_DIR = Path("outputs/stage7/epfl_lab_full")
DEFAULT_MARKET1501_METRICS = Path("outputs/evaluation/market1501/metrics.json")


@dataclass(frozen=True, slots=True)
class TopViewGeometry:
    """EPFL top-view rectangle used to convert grid cells into coordinates."""

    origin: tuple[float, float] = (0.0, 0.0)
    size: tuple[float, float] = (358.0, 360.0)

    def __post_init__(self) -> None:
        if len(self.origin) != 2 or not all(np.isfinite(self.origin)):
            raise ValueError("top-view origin must contain two finite values")
        if len(self.size) != 2 or not all(np.isfinite(self.size)):
            raise ValueError("top-view size must contain two finite values")
        if self.size[0] <= 0.0 or self.size[1] <= 0.0:
            raise ValueError("top-view dimensions must be positive")


@dataclass(frozen=True, slots=True)
class AssignmentObservation:
    """One online Stage 6 local-to-global assignment with track geometry."""

    frame_id: int
    timestamp: float
    camera_id: str
    local_id: int
    global_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    stored_samples: int
    total_samples: int

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.timestamp < 0.0 or not np.isfinite(self.timestamp):
            raise ValueError("assignment frame and timestamp must not be negative")
        if not self.camera_id.strip():
            raise ValueError("assignment camera_id must not be empty")
        if self.local_id <= 0 or self.global_id <= 0:
            raise ValueError("assignment local and global IDs must be positive")
        if len(self.xyxy) != 4 or not all(np.isfinite(self.xyxy)):
            raise ValueError("assignment xyxy must contain four finite values")
        if self.xyxy[2] <= self.xyxy[0] or self.xyxy[3] <= self.xyxy[1]:
            raise ValueError("assignment xyxy must have positive area")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("assignment confidence must be in [0, 1]")
        if self.stored_samples <= 0 or self.total_samples < self.stored_samples:
            raise ValueError("assignment sample counts are inconsistent")

    @classmethod
    def load_jsonl(cls, path: Path) -> tuple[AssignmentObservation, ...]:
        """Load and strictly validate the Stage 6 assignment schema."""
        assignment_path = path.expanduser().resolve()
        try:
            lines = assignment_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Stage 6 assignments do not exist: {assignment_path}"
            ) from error
        observations: list[AssignmentObservation] = []
        required = {
            "frame_id",
            "timestamp",
            "camera_id",
            "local_id",
            "global_id",
            "stored_samples",
            "total_samples",
            "xyxy",
            "confidence",
        }
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in assignments line {line_number}"
                ) from error
            missing = required - set(row)
            if missing:
                if {"xyxy", "confidence"} & missing:
                    raise ValueError(
                        "Stage 6 assignments lack track geometry; rerun Stage 6 "
                        "with assignment schema version 2"
                    )
                raise ValueError(
                    f"Assignments line {line_number} is missing {sorted(missing)}"
                )
            observations.append(
                cls(
                    frame_id=int(row["frame_id"]),
                    timestamp=float(row["timestamp"]),
                    camera_id=str(row["camera_id"]),
                    local_id=int(row["local_id"]),
                    global_id=int(row["global_id"]),
                    xyxy=tuple(float(value) for value in row["xyxy"]),
                    confidence=float(row["confidence"]),
                    stored_samples=int(row["stored_samples"]),
                    total_samples=int(row["total_samples"]),
                )
            )
        return tuple(observations)


@dataclass(frozen=True, slots=True)
class AttributedObservation:
    """A Stage 6 prediction spatially attributed to an EPFL person identity."""

    frame_id: int
    timestamp: float
    camera_id: str
    person_id: int
    local_id: int
    global_id: int
    confidence: float
    ground_distance_cells: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrameAttribution:
    """One camera/frame's accepted and rejected spatial attributions."""

    frame_id: int
    camera_id: str
    matches: tuple[AttributedObservation, ...]
    unmatched_predictions: tuple[AssignmentObservation, ...]
    unobserved_person_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EpflGlobalEvaluationConfig:
    """Frozen paths and geometry for reproducible Stage 7 evaluation."""

    dataset_root: Path = Path("data/epfl_lab")
    stage6_run_dir: Path = DEFAULT_STAGE6_RUN_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    cameras: tuple[str, ...] = ("C0", "C1")
    top_view: TopViewGeometry = TopViewGeometry()
    max_ground_distance_cells: float = 5.0
    market1501_metrics_path: Path | None = None

    def __post_init__(self) -> None:
        validate_epfl_cameras(self.cameras)
        if self.max_ground_distance_cells <= 0.0 or not np.isfinite(
            self.max_ground_distance_cells
        ):
            raise ValueError("max_ground_distance_cells must be finite and positive")


def image_point_to_top_view(
    point: tuple[float, float], homography: np.ndarray
) -> tuple[float, float]:
    """Project an image point through EPFL's image-to-ground homography."""
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("homography must be a finite 3x3 matrix")
    source = np.asarray((point[0], point[1], 1.0), dtype=np.float64)
    projected = matrix @ source
    if not np.all(np.isfinite(projected)) or abs(projected[2]) <= 1e-12:
        raise ValueError("image point projects to an invalid homogeneous coordinate")
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def top_view_point_to_grid(
    point: tuple[float, float],
    *,
    grid_size: tuple[int, int],
    geometry: TopViewGeometry,
) -> tuple[float, float]:
    """Convert top-view coordinates to continuous zero-based grid-cell centers."""
    grid_width, grid_height = grid_size
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError("grid dimensions must be positive")
    x = (point[0] - geometry.origin[0]) / (geometry.size[0] / grid_width) - 0.5
    y = (point[1] - geometry.origin[1]) / (geometry.size[1] / grid_height) - 0.5
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError("top-view point converts to invalid grid coordinates")
    return float(x), float(y)


def _position_to_grid(
    position: EpflPersonPosition, grid_size: tuple[int, int]
) -> tuple[float, float]:
    if position.state != "active" or position.position_id is None:
        raise ValueError("only active EPFL positions have grid coordinates")
    grid_width, grid_height = grid_size
    if not 0 <= position.position_id < grid_width * grid_height:
        raise ValueError("EPFL position_id is outside the declared grid")
    return (
        float(position.position_id % grid_width),
        float(position.position_id // grid_width),
    )


def attribute_frame(
    predictions: Sequence[AssignmentObservation],
    positions: Sequence[EpflPersonPosition],
    calibration: EpflCameraCalibration,
    *,
    grid_size: tuple[int, int],
    geometry: TopViewGeometry,
    max_ground_distance_cells: float,
) -> FrameAttribution:
    """Spatially attribute one camera frame using gated Hungarian assignment."""
    predicted = tuple(predictions)
    active = tuple(position for position in positions if position.state == "active")
    camera_id = calibration.camera_id
    frame_ids = {item.frame_id for item in predicted}
    camera_ids = {item.camera_id for item in predicted}
    if len(frame_ids) > 1 or (camera_ids and camera_ids != {camera_id}):
        raise ValueError("predictions must belong to one frame and the calibration camera")
    frame_id = next(iter(frame_ids), -1)
    if max_ground_distance_cells <= 0.0:
        raise ValueError("max_ground_distance_cells must be positive")

    if not predicted or not active:
        return FrameAttribution(
            frame_id=frame_id,
            camera_id=camera_id,
            matches=(),
            unmatched_predictions=predicted,
            unobserved_person_ids=tuple(position.person_id for position in active),
        )

    prediction_grid = []
    for item in predicted:
        footpoint = ((item.xyxy[0] + item.xyxy[2]) / 2.0, item.xyxy[3])
        top_view = image_point_to_top_view(footpoint, calibration.ground_homography)
        prediction_grid.append(
            top_view_point_to_grid(
                top_view,
                grid_size=grid_size,
                geometry=geometry,
            )
        )
    truth_grid = [_position_to_grid(position, grid_size) for position in active]
    distances = np.linalg.norm(
        np.asarray(prediction_grid)[:, None, :] - np.asarray(truth_grid)[None, :, :],
        axis=2,
    )
    prediction_indices, truth_indices = linear_sum_assignment(distances)
    accepted: list[AttributedObservation] = []
    accepted_predictions: set[int] = set()
    accepted_truth: set[int] = set()
    for prediction_index, truth_index in zip(
        prediction_indices, truth_indices, strict=True
    ):
        distance = float(distances[prediction_index, truth_index])
        if distance > max_ground_distance_cells:
            continue
        item = predicted[prediction_index]
        position = active[truth_index]
        accepted.append(
            AttributedObservation(
                frame_id=item.frame_id,
                timestamp=item.timestamp,
                camera_id=item.camera_id,
                person_id=position.person_id,
                local_id=item.local_id,
                global_id=item.global_id,
                confidence=item.confidence,
                ground_distance_cells=distance,
            )
        )
        accepted_predictions.add(prediction_index)
        accepted_truth.add(truth_index)
    return FrameAttribution(
        frame_id=frame_id,
        camera_id=camera_id,
        matches=tuple(accepted),
        unmatched_predictions=tuple(
            item for index, item in enumerate(predicted) if index not in accepted_predictions
        ),
        unobserved_person_ids=tuple(
            position.person_id
            for index, position in enumerate(active)
            if index not in accepted_truth
        ),
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _pair_metrics(tp: int, fp: int, fn: int, tn: int = 0) -> dict[str, int | float]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "true_positive_pairs": tp,
        "false_positive_pairs": fp,
        "false_negative_pairs": fn,
        "true_negative_pairs": tn,
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
    }


def _count_pair_outcomes(
    pairs: Iterable[tuple[AttributedObservation, AttributedObservation]],
) -> tuple[int, int, int, int]:
    counts = [0, 0, 0, 0]
    for left, right in pairs:
        same_truth = left.person_id == right.person_id
        same_prediction = left.global_id == right.global_id
        if same_truth and same_prediction:
            counts[0] += 1
        elif not same_truth and same_prediction:
            counts[1] += 1
        elif same_truth and not same_prediction:
            counts[2] += 1
        else:
            counts[3] += 1
    return tuple(counts)  # type: ignore[return-value]


def _switch_count(
    rows: Sequence[AttributedObservation], *, identity_field: str
) -> int:
    grouped: dict[tuple[str, int], list[AttributedObservation]] = defaultdict(list)
    for row in rows:
        grouped[(row.camera_id, row.person_id)].append(row)
    switches = 0
    for values in grouped.values():
        ordered = sorted(values, key=lambda item: item.frame_id)
        identities = [getattr(item, identity_field) for item in ordered]
        switches += sum(
            left != right
            for left, right in zip(identities, identities[1:], strict=False)
        )
    return switches


def _fragmentation(
    rows: Sequence[AttributedObservation],
    *,
    identity_field: str,
    include_camera: bool,
) -> int:
    grouped: dict[tuple[object, ...], set[int]] = defaultdict(set)
    for row in rows:
        key = (row.camera_id, row.person_id) if include_camera else (row.person_id,)
        grouped[key].add(int(getattr(row, identity_field)))
    return sum(max(0, len(identities) - 1) for identities in grouped.values())


def calculate_epfl_metrics(
    observations: Sequence[AttributedObservation],
    *,
    eligible_predictions: int,
    unmatched_predictions: int,
    scene_person_opportunities: Collection[tuple[int, int]],
) -> dict[str, Any]:
    """Calculate conditional association, coverage, switch, and purity metrics."""
    rows = tuple(observations)
    if eligible_predictions < len(rows) or unmatched_predictions < 0:
        raise ValueError("prediction counts are inconsistent with attributions")
    by_frame: dict[int, list[AttributedObservation]] = defaultdict(list)
    for row in rows:
        by_frame[row.frame_id].append(row)
    cross_camera_pairs = (
        pair
        for frame_rows in by_frame.values()
        for pair in combinations(frame_rows, 2)
        if pair[0].camera_id != pair[1].camera_id
    )
    cross_counts = _count_pair_outcomes(cross_camera_pairs)
    all_counts = _count_pair_outcomes(combinations(rows, 2))
    true_cross_pairs = cross_counts[0] + cross_counts[2]
    different_cross_pairs = cross_counts[1] + cross_counts[3]

    global_counts = Counter(row.global_id for row in rows)
    person_counts = Counter(row.person_id for row in rows)
    contingency = Counter((row.person_id, row.global_id) for row in rows)
    purity_correct = sum(
        max(
            (count for (person_id, candidate), count in contingency.items() if candidate == gid),
            default=0,
        )
        for gid in global_counts
    )
    covered = {(row.frame_id, row.person_id) for row in rows}
    distances = np.asarray([row.ground_distance_cells for row in rows], dtype=np.float64)

    per_camera: dict[str, dict[str, int | float]] = {}
    for camera_id in sorted({row.camera_id for row in rows}):
        camera_rows = tuple(row for row in rows if row.camera_id == camera_id)
        per_camera[camera_id] = {
            "attributed_observations": len(camera_rows),
            "unique_local_ids": len({row.local_id for row in camera_rows}),
            "unique_global_ids": len({row.global_id for row in camera_rows}),
            "local_id_switches": _switch_count(camera_rows, identity_field="local_id"),
            "global_id_switches": _switch_count(camera_rows, identity_field="global_id"),
        }
    per_person = {
        str(person_id): {
            "observations": count,
            "global_ids": sorted(
                {row.global_id for row in rows if row.person_id == person_id}
            ),
            "local_ids_by_camera": {
                camera_id: sorted(
                    {
                        row.local_id
                        for row in rows
                        if row.person_id == person_id and row.camera_id == camera_id
                    }
                )
                for camera_id in sorted({row.camera_id for row in rows})
            },
        }
        for person_id, count in sorted(person_counts.items())
    }

    separate_baseline = _pair_metrics(0, 0, true_cross_pairs, different_cross_pairs)
    single_baseline = _pair_metrics(
        true_cross_pairs,
        different_cross_pairs,
        0,
        0,
    )
    return {
        "eligible_predictions_at_annotated_frames": eligible_predictions,
        "attributed_predictions": len(rows),
        "unmatched_predictions": unmatched_predictions,
        "attribution_rate": _safe_ratio(len(rows), eligible_predictions),
        "scene_person_opportunities": len(scene_person_opportunities),
        "covered_scene_persons": len(covered & set(scene_person_opportunities)),
        "scene_person_coverage": _safe_ratio(
            len(covered & set(scene_person_opportunities)),
            len(scene_person_opportunities),
        ),
        "ground_distance_cells": {
            "mean": float(np.mean(distances)) if distances.size else None,
            "median": float(np.median(distances)) if distances.size else None,
            "p95": float(np.percentile(distances, 95)) if distances.size else None,
        },
        "cross_camera_association": _pair_metrics(*cross_counts),
        "global_cluster_pairwise": _pair_metrics(*all_counts),
        "global_tracking": {
            "id_switches": _switch_count(rows, identity_field="global_id"),
            "fragmentation": _fragmentation(
                rows, identity_field="global_id", include_camera=False
            ),
            "purity": _safe_ratio(purity_correct, len(rows)),
            "unique_global_ids": len(global_counts),
        },
        "local_tracking": {
            "id_switches": _switch_count(rows, identity_field="local_id"),
            "fragmentation": _fragmentation(
                rows, identity_field="local_id", include_camera=True
            ),
        },
        "baselines": {
            "separate_camera": separate_baseline,
            "single_global_identity": single_baseline,
        },
        "per_camera": per_camera,
        "per_person": per_person,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{context} does not exist: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a JSON object")
    return value


def _write_contingency(path: Path, rows: Sequence[AttributedObservation]) -> None:
    counts = Counter((row.person_id, row.global_id) for row in rows)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("person_id", "global_id", "observations"))
        for (person_id, global_id), count in sorted(counts.items()):
            writer.writerow((person_id, global_id, count))


def _write_report(path: Path, metrics: Mapping[str, Any]) -> None:
    association = metrics["cross_camera_association"]
    global_tracking = metrics["global_tracking"]
    local_tracking = metrics["local_tracking"]
    stage6 = metrics["stage6"]
    baselines = metrics["baselines"]
    market = metrics.get("market1501_reid")

    def stage6_float(name: str) -> str:
        value = stage6.get(name)
        return f"{float(value):.2f}" if value is not None else "not recorded"

    cross_camera_counts = (
        f"{association['true_positive_pairs']} / "
        f"{association['false_positive_pairs']} / "
        f"{association['false_negative_pairs']}"
    )
    market_rows = ""
    if market is not None:
        cmc = market.get("cmc") or {}
        market_rows = f"""
| Market-1501 mAP | {float(market['mAP']):.4f} |
| Market-1501 Rank-1 | {float(cmc['rank_1']):.4f} |
| Market-1501 Rank-5 | {float(cmc['rank_5']):.4f} |
| Market-1501 Rank-10 | {float(cmc['rank_10']):.4f} |"""
    report = f"""# EPFL Laboratory Stage 7 Evaluation

This report evaluates online global-ID output at the official sparse EPFL
annotation frames. Spatial attribution uses calibrated bounding-box footpoints;
association metrics are conditional on successful attribution in both cameras.

| Metric | Value |
|---|---:|
| Cross-camera association precision | {association['precision']:.4f} |
| Cross-camera association recall | {association['recall']:.4f} |
| Cross-camera association F1 | {association['f1']:.4f} |
| Cross-camera TP / FP / FN | {cross_camera_counts} |
| Attribution rate | {metrics['attribution_rate']:.4f} |
| Scene-person coverage | {metrics['scene_person_coverage']:.4f} |
| Global ID switches | {global_tracking['id_switches']} |
| Global fragmentation | {global_tracking['fragmentation']} |
| Global assignment purity | {global_tracking['purity']:.4f} |
| Local ID switches | {local_tracking['id_switches']} |
| Local fragmentation | {local_tracking['fragmentation']} |{market_rows}

## Runtime and baselines

| Measurement | Value |
|---|---:|
| Synchronized frames | {stage6.get('frames_processed', 'not recorded')} |
| Camera frames | {stage6.get('camera_frames_processed', 'not recorded')} |
| Elapsed seconds | {stage6_float('elapsed_seconds')} |
| Synchronized FPS | {stage6_float('synchronized_fps')} |
| Camera-processing FPS | {stage6_float('camera_processing_fps')} |
| Separate-camera baseline F1 | {baselines['separate_camera']['f1']:.4f} |
| Single-global-ID baseline F1 | {baselines['single_global_identity']['f1']:.4f} |

The Stage 6 cosine-distance threshold remained frozen at
`{stage6.get('threshold', 'not recorded')}` during this reporting run. The separate-camera
and single-global-ID baselines expose the two trivial failure modes: never merge
anyone, or merge everyone.

## Interpretation

Cross-camera F1 asks the narrow question that matters at one annotated instant:
when two attributed camera observations belong to the same person, did the
system give them the same global ID, without also merging different people?
Global/local switches and fragmentation use each person's sparse observed
sequence, so they diagnose continuity but are not dense-frame tracking scores.

The EPFL labels are scene-level ground positions, not per-camera bounding boxes
or visibility flags. These are calibrated sparse association diagnostics, not
official MOTChallenge MOTA/IDF1 scores.
"""
    path.write_text(report, encoding="utf-8")


def run_epfl_global_evaluation(config: EpflGlobalEvaluationConfig) -> dict[str, Any]:
    """Evaluate one completed Stage 6 run and write reproducible Stage 7 artifacts."""
    cameras = validate_epfl_cameras(config.cameras)
    dataset_root = config.dataset_root.expanduser().resolve()
    run_dir = config.stage6_run_dir.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = run_dir / "assignments.jsonl"
    summary_path = run_dir / "summary.json"
    ground_truth_path = dataset_root / GROUND_TRUTH_FILENAME
    calibration_path = dataset_root / CALIBRATION_FILENAME

    observations = AssignmentObservation.load_jsonl(assignments_path)
    summary = _load_json(summary_path, context="Stage 6 summary")
    summary_cameras = tuple(summary.get("cameras", ()))
    if summary_cameras != cameras:
        raise ValueError(
            f"Stage 6 cameras {summary_cameras} do not match evaluation cameras {cameras}"
        )
    start_frame = int(summary["start_frame"])
    frame_stride = int(summary["frame_stride"])
    frames_processed = int(summary["frames_processed"])
    if frame_stride <= 0 or frames_processed <= 0:
        raise ValueError("Stage 6 summary has invalid processed-frame metadata")
    processed_frames = {
        start_frame + index * frame_stride for index in range(frames_processed)
    }
    invalid_observations = [
        row
        for row in observations
        if row.camera_id not in cameras or row.frame_id not in processed_frames
    ]
    if invalid_observations:
        raise ValueError("Assignments contain cameras or frames outside the Stage 6 summary")

    ground_truth = load_epfl_ground_truth(ground_truth_path)
    calibrations = load_epfl_calibration(calibration_path)
    evaluated_frames = tuple(
        frame_id
        for frame_id in sorted(ground_truth.annotations)
        if frame_id in processed_frames
    )
    if not evaluated_frames:
        raise ValueError("Stage 6 run does not overlap any annotated EPFL frames")
    by_frame_camera: dict[tuple[int, str], list[AssignmentObservation]] = defaultdict(list)
    for row in observations:
        by_frame_camera[(row.frame_id, row.camera_id)].append(row)

    frame_results: list[FrameAttribution] = []
    opportunities: set[tuple[int, int]] = set()
    eligible_predictions = 0
    unmatched_predictions = 0
    for frame_id in evaluated_frames:
        positions = ground_truth.active_positions_at(frame_id)
        opportunities.update((frame_id, position.person_id) for position in positions)
        for camera_id in cameras:
            predictions = tuple(by_frame_camera[(frame_id, camera_id)])
            result = attribute_frame(
                predictions,
                positions,
                calibrations[camera_id],
                grid_size=ground_truth.grid_size,
                geometry=config.top_view,
                max_ground_distance_cells=config.max_ground_distance_cells,
            )
            frame_results.append(result)
            eligible_predictions += len(predictions)
            unmatched_predictions += len(result.unmatched_predictions)
    attributed = tuple(match for result in frame_results for match in result.matches)
    metrics = calculate_epfl_metrics(
        attributed,
        eligible_predictions=eligible_predictions,
        unmatched_predictions=unmatched_predictions,
        scene_person_opportunities=opportunities,
    )
    metrics.update(
        {
            "schema_version": 1,
            "dataset": "EPFL Laboratory six-person sequence",
            "dataset_root": str(dataset_root),
            "cameras": list(cameras),
            "evaluated_annotated_frames": len(evaluated_frames),
            "first_evaluated_frame": evaluated_frames[0],
            "last_evaluated_frame": evaluated_frames[-1],
            "evaluation_contract": {
                "label_grain": "scene person at one sparse annotated source frame",
                "prediction_grain": "Re-ID-eligible local track in one camera",
                "spatial_attribution": "bounding-box bottom center projected to ground grid",
                "assignment": "Hungarian one-to-one",
                "max_ground_distance_cells": config.max_ground_distance_cells,
                "primary_metric": "cross_camera_association.f1",
                "association_condition": (
                    "both camera observations were spatially attributed"
                ),
                "threshold_policy": "Stage 6 cosine threshold remains frozen",
                "mot_limitation": (
                    "scene-level sparse positions are not per-camera boxes or visibility labels"
                ),
            },
            "stage6": summary,
            "provenance": {
                "assignments_sha256": _sha256(assignments_path),
                "stage6_summary_sha256": _sha256(summary_path),
                "ground_truth_sha256": _sha256(ground_truth_path),
                "calibration_sha256": _sha256(calibration_path),
            },
        }
    )
    if config.market1501_metrics_path is not None:
        market_path = config.market1501_metrics_path.expanduser().resolve()
        market = _load_json(market_path, context="Market-1501 metrics")
        metrics["market1501_reid"] = {
            "mAP": market.get("mAP"),
            "cmc": market.get("cmc"),
            "checkpoint_sha256": market.get("checkpoint_sha256"),
            "metrics_path": str(market_path),
            "metrics_sha256": _sha256(market_path),
        }

    metrics_path = output_dir / "metrics.json"
    attributions_path = output_dir / "attributions.jsonl"
    contingency_path = output_dir / "contingency.csv"
    report_path = output_dir / "report.md"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with attributions_path.open("w", encoding="utf-8") as output:
        for row in attributed:
            output.write(json.dumps(row.to_dict()) + "\n")
        for result in frame_results:
            for row in result.unmatched_predictions:
                output.write(
                    json.dumps(
                        {
                            "frame_id": row.frame_id,
                            "timestamp": row.timestamp,
                            "camera_id": row.camera_id,
                            "local_id": row.local_id,
                            "global_id": row.global_id,
                            "status": "unmatched_prediction",
                        }
                    )
                    + "\n"
                )
    _write_contingency(contingency_path, attributed)
    _write_report(report_path, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 7 offline EPFL evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 6 EPFL global IDs against sparse ground positions."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/epfl_lab"))
    parser.add_argument("--stage6-run-dir", type=Path, default=DEFAULT_STAGE6_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cameras", nargs="+", default=("C0", "C1"))
    parser.add_argument("--max-ground-distance-cells", type=float, default=5.0)
    parser.add_argument(
        "--market1501-metrics", type=Path, default=DEFAULT_MARKET1501_METRICS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Stage 7 evaluation and print its stable metric artifact."""
    args = build_parser().parse_args(argv)
    metrics = run_epfl_global_evaluation(
        EpflGlobalEvaluationConfig(
            dataset_root=args.dataset_root,
            stage6_run_dir=args.stage6_run_dir,
            output_dir=args.output_dir,
            cameras=tuple(args.cameras),
            max_ground_distance_cells=args.max_ground_distance_cells,
            market1501_metrics_path=args.market1501_metrics,
        )
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
