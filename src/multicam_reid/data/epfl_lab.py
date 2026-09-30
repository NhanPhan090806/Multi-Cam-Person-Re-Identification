"""Download, parse, and validate the sparse EPFL Laboratory sequence."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlopen

import cv2
import numpy as np
from numpy.typing import NDArray

SEQUENCE_NAME = "6p"
CAMERA_IDS = tuple(f"C{index}" for index in range(4))
VIDEO_TEMPLATE = "6p-c{camera_index}.avi"
CALIBRATION_FILENAME = "calibration-6p.txt"
GROUND_TRUTH_FILENAME = "gt_lab_6p.txt"
VIDEO_BASE_URL = "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www"
METADATA_BASE_URL = "https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08"

FileDownloader = Callable[[str, Path], None]
PositionState = Literal["active", "undefined", "out_of_scene"]


class EpflLabLayoutError(ValueError):
    """Raised when EPFL Laboratory files violate the synchronized data contract."""


@dataclass(frozen=True, slots=True)
class EpflPersonPosition:
    """One persistent identity's ground-grid state at an annotated frame."""

    person_id: int
    position_id: int | None
    state: PositionState


@dataclass(frozen=True, slots=True)
class EpflGroundTruth:
    """Identity-labelled positions sampled at the dataset annotation interval."""

    number_of_frames: int
    person_ids: tuple[int, ...]
    grid_size: tuple[int, int]
    annotation_step: int
    first_frame: int
    last_frame: int
    annotations: dict[int, tuple[EpflPersonPosition, ...]]

    def positions_at(self, raw_frame_id: int) -> tuple[EpflPersonPosition, ...]:
        """Return all identity states when this frame is annotated."""
        return self.annotations.get(raw_frame_id, ())

    def active_positions_at(self, raw_frame_id: int) -> tuple[EpflPersonPosition, ...]:
        """Return only identities known to be inside the tracked area."""
        return tuple(
            position
            for position in self.positions_at(raw_frame_id)
            if position.state == "active"
        )


@dataclass(frozen=True, slots=True, eq=False)
class EpflCameraCalibration:
    """Image-to-ground homography and optional head-plane calibration."""

    camera_id: str
    ground_homography: NDArray[np.float64]
    head_homography: NDArray[np.float64] | None
    head_plane_height: float | None


@dataclass(frozen=True, slots=True)
class EpflLabSummary:
    """Integrity and sparsity statistics for selected synchronized videos."""

    dataset_root: Path
    cameras: tuple[str, ...]
    synchronized_frames: int
    video_fps: float
    image_size: tuple[int, int]
    annotated_frames: int
    annotation_step: int
    persistent_person_ids: int
    active_position_annotations: int
    maximum_people_in_scene: int
    bytes_on_disk: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        result = asdict(self)
        result["dataset_root"] = str(self.dataset_root)
        return result


def validate_epfl_cameras(cameras: Sequence[str]) -> tuple[str, ...]:
    """Validate and preserve an EPFL Laboratory C0-through-C3 selection."""
    selected = tuple(cameras)
    if not selected:
        raise ValueError("At least one EPFL camera is required")
    if len(set(selected)) != len(selected):
        raise ValueError("EPFL cameras must be unique")
    invalid = [camera for camera in selected if camera not in CAMERA_IDS]
    if invalid:
        raise ValueError(f"EPFL camera IDs must be C0 through C3; got {invalid}")
    return selected


def video_filename(camera_id: str) -> str:
    """Return the official six-person video filename for one camera."""
    validate_epfl_cameras((camera_id,))
    return VIDEO_TEMPLATE.format(camera_index=int(camera_id[1:]))


def _is_dataset_root(path: Path) -> bool:
    return (
        (path / CALIBRATION_FILENAME).is_file()
        and (path / GROUND_TRUTH_FILENAME).is_file()
        and any((path / video_filename(camera)).is_file() for camera in CAMERA_IDS)
    )


def find_epfl_lab_root(search_root: Path) -> Path:
    """Resolve a direct directory or a conventional ``epfl_lab`` child."""
    root = search_root.expanduser().resolve()
    for candidate in (root, root / "epfl_lab"):
        if _is_dataset_root(candidate):
            return candidate.resolve()
    raise EpflLabLayoutError(
        "Could not find EPFL Laboratory videos, calibration, and ground truth under "
        f"{root} or {root / 'epfl_lab'}"
    )


def load_epfl_ground_truth(path: Path) -> EpflGroundTruth:
    """Parse the official identity-column ground-grid annotation format."""
    annotation_path = path.expanduser().resolve()
    try:
        lines = [line.strip() for line in annotation_path.read_text().splitlines() if line.strip()]
    except FileNotFoundError as error:
        raise EpflLabLayoutError(
            f"EPFL ground truth does not exist: {annotation_path}"
        ) from error
    if len(lines) < 2:
        raise EpflLabLayoutError("EPFL ground truth must contain a version and header")
    try:
        version = int(lines[0])
        header = tuple(int(value) for value in lines[1].split())
    except ValueError as error:
        raise EpflLabLayoutError("EPFL ground-truth header contains non-integers") from error
    if version != 1:
        raise EpflLabLayoutError(f"Unsupported EPFL ground-truth version: {version}")
    if len(header) != 7:
        raise EpflLabLayoutError("EPFL ground-truth header must contain seven integers")
    (
        number_of_frames,
        number_of_people,
        grid_width,
        grid_height,
        annotation_step,
        first_frame,
        last_frame,
    ) = header
    if min(number_of_frames, number_of_people, grid_width, grid_height, annotation_step) <= 0:
        raise EpflLabLayoutError("EPFL ground-truth dimensions and step must be positive")
    if first_frame < 0 or last_frame < first_frame:
        raise EpflLabLayoutError("EPFL ground-truth frame range is invalid")
    if (last_frame - first_frame) % annotation_step:
        raise EpflLabLayoutError("EPFL ground-truth frame range is not divisible by its step")

    records = lines[2:]
    if len(records) != number_of_frames:
        raise EpflLabLayoutError(
            f"EPFL ground truth has {len(records)} records, expected {number_of_frames}"
        )
    if last_frame >= number_of_frames:
        raise EpflLabLayoutError(
            "EPFL ground-truth last annotated frame must be inside the dense frame rows"
        )

    annotations: dict[int, tuple[EpflPersonPosition, ...]] = {}
    for record_index, record in enumerate(records):
        try:
            values = tuple(int(value) for value in record.split())
        except ValueError as error:
            raise EpflLabLayoutError(
                f"EPFL ground-truth record {record_index} contains non-integers"
            ) from error
        if len(values) != number_of_people:
            raise EpflLabLayoutError(
                f"EPFL ground-truth record {record_index} has {len(values)} people, "
                f"expected {number_of_people}"
            )
        positions = []
        for person_id, value in enumerate(values):
            if value >= 0:
                positions.append(EpflPersonPosition(person_id, value, "active"))
            elif value == -1:
                positions.append(EpflPersonPosition(person_id, None, "undefined"))
            elif value == -2:
                positions.append(EpflPersonPosition(person_id, None, "out_of_scene"))
            else:
                raise EpflLabLayoutError(
                    f"EPFL ground-truth record {record_index} has invalid position {value}"
                )
        raw_frame_id = record_index
        is_scheduled = (
            first_frame <= raw_frame_id <= last_frame
            and (raw_frame_id - first_frame) % annotation_step == 0
        )
        if is_scheduled:
            annotations[raw_frame_id] = tuple(positions)

    return EpflGroundTruth(
        number_of_frames=number_of_frames,
        person_ids=tuple(range(number_of_people)),
        grid_size=(grid_width, grid_height),
        annotation_step=annotation_step,
        first_frame=first_frame,
        last_frame=last_frame,
        annotations=annotations,
    )


def _matrix_from_lines(lines: list[str], start: int, *, context: str) -> NDArray[np.float64]:
    rows: list[list[float]] = []
    index = start
    while index < len(lines) and len(rows) < 3:
        line = lines[index]
        index += 1
        if not line or line.startswith("#"):
            continue
        try:
            row = [float(value) for value in line.split()]
        except ValueError as error:
            raise EpflLabLayoutError(f"{context} contains non-numeric values") from error
        if len(row) != 3:
            raise EpflLabLayoutError(f"{context} must contain three values per row")
        rows.append(row)
    if len(rows) != 3:
        raise EpflLabLayoutError(f"{context} is incomplete")
    return np.asarray(rows, dtype=np.float64)


def _scalar_from_lines(lines: list[str], start: int, *, context: str) -> float:
    for line in lines[start:]:
        if not line or line.startswith("#"):
            continue
        try:
            return float(line)
        except ValueError as error:
            raise EpflLabLayoutError(f"{context} is not numeric") from error
    raise EpflLabLayoutError(f"{context} is incomplete")


def load_epfl_calibration(path: Path) -> dict[str, EpflCameraCalibration]:
    """Parse the official camera-section homography file."""
    calibration_path = path.expanduser().resolve()
    try:
        lines = calibration_path.read_text().splitlines()
    except FileNotFoundError as error:
        raise EpflLabLayoutError(
            f"EPFL calibration does not exist: {calibration_path}"
        ) from error

    camera_starts = [
        index for index, line in enumerate(lines) if line.strip().startswith("# Camera ")
    ]
    calibrations: dict[str, EpflCameraCalibration] = {}
    for camera_index, start in enumerate(camera_starts):
        section_end = (
            camera_starts[camera_index + 1]
            if camera_index + 1 < len(camera_starts)
            else len(lines)
        )
        section = lines[start:section_end]
        try:
            camera_number = int(section[0].strip().removeprefix("# Camera "))
        except ValueError as error:
            raise EpflLabLayoutError(f"Invalid EPFL camera heading: {section[0]}") from error
        camera_id = f"C{camera_number}"
        ground_marker = next(
            (index for index, line in enumerate(section) if "Ground plane homography" in line),
            None,
        )
        if ground_marker is None:
            raise EpflLabLayoutError(f"Camera {camera_id} lacks a ground homography")
        ground = _matrix_from_lines(
            section,
            ground_marker + 1,
            context=f"Camera {camera_id} ground homography",
        )
        head_marker = next(
            (index for index, line in enumerate(section) if "Head plane homography" in line),
            None,
        )
        height_marker = next(
            (index for index, line in enumerate(section) if "Head plane height" in line),
            None,
        )
        head = (
            _matrix_from_lines(
                section,
                head_marker + 1,
                context=f"Camera {camera_id} head homography",
            )
            if head_marker is not None
            else None
        )
        height = (
            _scalar_from_lines(
                section,
                height_marker + 1,
                context=f"Camera {camera_id} head-plane height",
            )
            if height_marker is not None
            else None
        )
        calibrations[camera_id] = EpflCameraCalibration(
            camera_id=camera_id,
            ground_homography=ground,
            head_homography=head,
            head_plane_height=height,
        )
    if not calibrations:
        raise EpflLabLayoutError("No EPFL camera sections were found in calibration")
    return calibrations


def _video_metadata(path: Path) -> tuple[int, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise EpflLabLayoutError(f"Could not open EPFL video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise EpflLabLayoutError(f"EPFL video metadata is invalid: {path}")
    return frame_count, fps, (width, height)


def inspect_epfl_lab(
    search_root: Path,
    *,
    cameras: Sequence[str] = ("C0", "C1"),
) -> EpflLabSummary:
    """Validate selected videos, identity positions, and calibration."""
    selected = validate_epfl_cameras(cameras)
    root = find_epfl_lab_root(search_root)
    ground_truth = load_epfl_ground_truth(root / GROUND_TRUTH_FILENAME)
    calibrations = load_epfl_calibration(root / CALIBRATION_FILENAME)
    missing_calibrations = [camera for camera in selected if camera not in calibrations]
    if missing_calibrations:
        raise EpflLabLayoutError(
            f"EPFL calibration is missing selected cameras: {missing_calibrations}"
        )

    metadata = {}
    for camera in selected:
        path = root / video_filename(camera)
        if not path.is_file():
            raise EpflLabLayoutError(f"Missing EPFL camera video: {path}")
        metadata[camera] = _video_metadata(path)
    frame_counts = {camera: values[0] for camera, values in metadata.items()}
    if len(set(frame_counts.values())) != 1:
        raise EpflLabLayoutError(f"EPFL camera frame counts differ: {frame_counts}")
    fps_values = {camera: values[1] for camera, values in metadata.items()}
    if max(fps_values.values()) - min(fps_values.values()) > 0.01:
        raise EpflLabLayoutError(f"EPFL camera frame rates differ: {fps_values}")
    sizes = {camera: values[2] for camera, values in metadata.items()}
    if len(set(sizes.values())) != 1:
        raise EpflLabLayoutError(f"EPFL camera image sizes differ: {sizes}")
    synchronized_frames = next(iter(frame_counts.values()))
    if ground_truth.last_frame >= synchronized_frames:
        raise EpflLabLayoutError(
            "EPFL ground truth references frame "
            f"{ground_truth.last_frame}, but videos contain {synchronized_frames} frames"
        )

    active_counts = [
        len(ground_truth.active_positions_at(frame_id))
        for frame_id in ground_truth.annotations
    ]
    bytes_on_disk = sum(
        (root / filename).stat().st_size
        for filename in (
            *(video_filename(camera) for camera in selected),
            CALIBRATION_FILENAME,
            GROUND_TRUTH_FILENAME,
        )
    )
    return EpflLabSummary(
        dataset_root=root,
        cameras=selected,
        synchronized_frames=synchronized_frames,
        video_fps=next(iter(fps_values.values())),
        image_size=next(iter(sizes.values())),
        annotated_frames=len(ground_truth.annotations),
        annotation_step=ground_truth.annotation_step,
        persistent_person_ids=len(ground_truth.person_ids),
        active_position_annotations=sum(active_counts),
        maximum_people_in_scene=max(active_counts, default=0),
        bytes_on_disk=bytes_on_disk,
    )


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url) as response, temporary.open("wb") as output:  # noqa: S310
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def download_epfl_lab(
    output_dir: Path,
    *,
    cameras: Sequence[str] = ("C0", "C1"),
    downloader: FileDownloader = _download,
) -> EpflLabSummary:
    """Download selected synchronized videos plus shared calibration and labels."""
    selected = validate_epfl_cameras(cameras)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = [
        *(
            (
                f"{VIDEO_BASE_URL}/{video_filename(camera)}",
                video_filename(camera),
            )
            for camera in selected
        ),
        (f"{METADATA_BASE_URL}/{CALIBRATION_FILENAME}", CALIBRATION_FILENAME),
        (f"{METADATA_BASE_URL}/{GROUND_TRUTH_FILENAME}", GROUND_TRUTH_FILENAME),
    ]
    for url, filename in files:
        path = destination / filename
        if path.is_file() and path.stat().st_size > 0:
            continue
        print(f"Downloading {filename} ...")
        downloader(url, path)
    return inspect_epfl_lab(destination, cameras=selected)


def build_parser() -> argparse.ArgumentParser:
    """Build the sparse EPFL Laboratory downloader CLI."""
    parser = argparse.ArgumentParser(
        description="Download and validate selected EPFL Laboratory six-person views."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/epfl_lab"))
    parser.add_argument("--cameras", nargs="+", default=("C0", "C1"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the EPFL Laboratory downloader CLI."""
    args = build_parser().parse_args(argv)
    summary = download_epfl_lab(args.output_dir, cameras=args.cameras)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
