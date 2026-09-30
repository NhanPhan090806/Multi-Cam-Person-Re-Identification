"""Download and validate the synchronized WILDTRACK camera subset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HF_REPO_ID = "disl/my_dataset"
DATASET_RELATIVE_ROOT = Path("Wildtrack_dataset_full") / "Wildtrack_dataset"
IMAGE_DIRECTORY = "Image_subsets"
ANNOTATION_DIRECTORY = "annotations_positions"
CALIBRATION_DIRECTORY = "calibrations"
ANNOTATION_FPS = 2.0

DatasetDownloader = Callable[..., str]


class WildtrackLayoutError(ValueError):
    """Raised when WILDTRACK files or annotations violate the data contract."""


@dataclass(frozen=True, slots=True)
class WildtrackViewAnnotation:
    """One visible ground-truth person box in one WILDTRACK camera."""

    camera_id: str
    view_number: int
    xyxy: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class WildtrackPersonAnnotation:
    """One persistent WILDTRACK person ID and its selected visible views."""

    person_id: int
    position_id: int
    views: tuple[WildtrackViewAnnotation, ...]

    def box_for(self, camera_id: str) -> WildtrackViewAnnotation | None:
        """Return this person's visible box for one camera, if present."""
        return next((view for view in self.views if view.camera_id == camera_id), None)


@dataclass(frozen=True, slots=True)
class WildtrackSummary:
    """Integrity statistics for an annotation-aligned camera subset."""

    dataset_root: Path
    cameras: tuple[str, ...]
    synchronized_frames: int
    images_per_camera: dict[str, int]
    image_size: tuple[int, int]
    person_annotations: int
    visible_boxes_by_camera: dict[str, int]
    annotation_fps: float = ANNOTATION_FPS
    calibrations_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        result = asdict(self)
        result["dataset_root"] = str(self.dataset_root)
        return result


def validate_wildtrack_cameras(cameras: Sequence[str]) -> tuple[str, ...]:
    """Validate and preserve a WILDTRACK C1-through-C7 camera selection."""
    selected = tuple(cameras)
    if not selected:
        raise ValueError("At least one WILDTRACK camera is required")
    if len(set(selected)) != len(selected):
        raise ValueError("WILDTRACK cameras must be unique")
    allowed = {f"C{number}" for number in range(1, 8)}
    invalid = [camera for camera in selected if camera not in allowed]
    if invalid:
        raise ValueError(f"WILDTRACK camera IDs must be C1 through C7; got {invalid}")
    return selected


def _is_dataset_root(path: Path) -> bool:
    return (path / IMAGE_DIRECTORY).is_dir() and (path / ANNOTATION_DIRECTORY).is_dir()


def find_wildtrack_root(search_root: Path) -> Path:
    """Resolve the dataset under direct, official-archive, or mirror layouts."""
    root = search_root.expanduser().resolve()
    candidates = (
        root,
        root / "Wildtrack_dataset",
        root / DATASET_RELATIVE_ROOT,
        root / "Wildtrack_dataset_full",
    )
    for candidate in candidates:
        if _is_dataset_root(candidate):
            return candidate.resolve()
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise WildtrackLayoutError(
        "Could not find WILDTRACK Image_subsets and annotations_positions under "
        f"any expected root: {expected}"
    )


def _required_int(record: dict[str, Any], key: str, *, context: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise WildtrackLayoutError(f"{context} field {key!r} must be an integer")
    return value


def load_wildtrack_annotations(
    path: Path,
    *,
    cameras: Sequence[str] = ("C1", "C2"),
) -> tuple[WildtrackPersonAnnotation, ...]:
    """Parse one official WILDTRACK JSON file for selected camera views."""
    selected = validate_wildtrack_cameras(cameras)
    annotation_path = path.expanduser().resolve()
    try:
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"WILDTRACK annotation does not exist: {annotation_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise WildtrackLayoutError(
            f"Invalid WILDTRACK JSON annotation: {annotation_path}"
        ) from error
    if not isinstance(raw, list):
        raise WildtrackLayoutError(f"WILDTRACK annotation must contain a list: {annotation_path}")

    people: list[WildtrackPersonAnnotation] = []
    seen_people: set[int] = set()
    for person_index, person in enumerate(raw):
        context = f"{annotation_path.name} person[{person_index}]"
        if not isinstance(person, dict):
            raise WildtrackLayoutError(f"{context} must be an object")
        person_id = _required_int(person, "personID", context=context)
        position_id = _required_int(person, "positionID", context=context)
        if person_id in seen_people:
            raise WildtrackLayoutError(
                f"{annotation_path.name} contains duplicate personID {person_id}"
            )
        seen_people.add(person_id)
        raw_views = person.get("views")
        if not isinstance(raw_views, list):
            raise WildtrackLayoutError(f"{context} field 'views' must be a list")

        by_view_number: dict[int, dict[str, Any]] = {}
        for fallback_view_number, view in enumerate(raw_views):
            if not isinstance(view, dict):
                raise WildtrackLayoutError(f"{context} view must be an object")
            raw_view_number = view.get("viewNum", fallback_view_number)
            if isinstance(raw_view_number, bool) or not isinstance(raw_view_number, int):
                raise WildtrackLayoutError(f"{context} viewNum must be an integer")
            if raw_view_number in by_view_number:
                raise WildtrackLayoutError(
                    f"{context} contains duplicate viewNum {raw_view_number}"
                )
            by_view_number[raw_view_number] = view

        parsed_views: list[WildtrackViewAnnotation] = []
        for camera_id in selected:
            view_number = int(camera_id[1:]) - 1
            view = by_view_number.get(view_number)
            if view is None:
                raise WildtrackLayoutError(f"{context} is missing viewNum {view_number}")
            coordinates = tuple(
                _required_int(view, key, context=f"{context} viewNum {view_number}")
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
            if any(value == -1 for value in coordinates):
                continue
            xmin, ymin, xmax, ymax = coordinates
            if xmin >= xmax or ymin >= ymax:
                raise WildtrackLayoutError(
                    f"{context} viewNum {view_number} has invalid box {coordinates}"
                )
            parsed_views.append(
                WildtrackViewAnnotation(
                    camera_id=camera_id,
                    view_number=view_number,
                    xyxy=(xmin, ymin, xmax, ymax),
                )
            )
        people.append(
            WildtrackPersonAnnotation(
                person_id=person_id,
                position_id=position_id,
                views=tuple(parsed_views),
            )
        )
    return tuple(people)


def _decode_image(path: Path) -> np.ndarray:
    try:
        encoded = path.read_bytes()
    except FileNotFoundError as error:
        raise WildtrackLayoutError(f"WILDTRACK image does not exist: {path}") from error
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise WildtrackLayoutError(f"Could not decode WILDTRACK image: {path}")
    return image


def inspect_wildtrack(
    search_root: Path,
    *,
    cameras: Sequence[str] = ("C1", "C2"),
) -> WildtrackSummary:
    """Validate synchronized images and annotations for selected cameras."""
    selected = validate_wildtrack_cameras(cameras)
    dataset_root = find_wildtrack_root(search_root)
    annotation_paths = sorted((dataset_root / ANNOTATION_DIRECTORY).glob("*.json"))
    if not annotation_paths:
        raise WildtrackLayoutError("No WILDTRACK JSON annotations were found")

    images_per_camera: dict[str, int] = {}
    for camera_id in selected:
        camera_dir = dataset_root / IMAGE_DIRECTORY / camera_id
        if not camera_dir.is_dir():
            raise WildtrackLayoutError(f"Missing WILDTRACK camera directory: {camera_dir}")
        images_per_camera[camera_id] = len(tuple(camera_dir.glob("*.png")))
        for annotation_path in annotation_paths:
            image_path = camera_dir / f"{annotation_path.stem}.png"
            if not image_path.is_file():
                raise WildtrackLayoutError(
                    f"Camera {camera_id} is missing synchronized frame {annotation_path.stem}"
                )

    first_sizes = []
    for camera_id in selected:
        first_image = _decode_image(
            dataset_root / IMAGE_DIRECTORY / camera_id / f"{annotation_paths[0].stem}.png"
        )
        first_sizes.append((first_image.shape[1], first_image.shape[0]))
    if len(set(first_sizes)) != 1:
        raise WildtrackLayoutError(f"Selected WILDTRACK cameras have unequal sizes: {first_sizes}")

    person_annotations = 0
    visible_boxes = {camera_id: 0 for camera_id in selected}
    for annotation_path in annotation_paths:
        people = load_wildtrack_annotations(annotation_path, cameras=selected)
        person_annotations += len(people)
        for person in people:
            for view in person.views:
                visible_boxes[view.camera_id] += 1

    return WildtrackSummary(
        dataset_root=dataset_root,
        cameras=selected,
        synchronized_frames=len(annotation_paths),
        images_per_camera=images_per_camera,
        image_size=first_sizes[0],
        person_annotations=person_annotations,
        visible_boxes_by_camera=visible_boxes,
        calibrations_available=(dataset_root / CALIBRATION_DIRECTORY).is_dir(),
    )


def download_wildtrack(
    output_dir: Path,
    *,
    cameras: Sequence[str] = ("C1", "C2"),
    downloader: DatasetDownloader | None = None,
) -> WildtrackSummary:
    """Download annotations, calibrations, and only the selected camera folders."""
    selected = validate_wildtrack_cameras(cameras)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    prefix = DATASET_RELATIVE_ROOT.as_posix()
    allow_patterns = [
        *(f"{prefix}/{IMAGE_DIRECTORY}/{camera_id}/**" for camera_id in selected),
        f"{prefix}/{ANNOTATION_DIRECTORY}/**",
        f"{prefix}/{CALIBRATION_DIRECTORY}/**",
    ]
    downloaded_path = Path(
        downloader(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(destination),
            allow_patterns=allow_patterns,
        )
    ).expanduser()
    return inspect_wildtrack(downloaded_path, cameras=selected)


def build_parser() -> argparse.ArgumentParser:
    """Build the WILDTRACK subset downloader parser."""
    parser = argparse.ArgumentParser(
        description="Download and validate an annotation-aligned WILDTRACK camera subset."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/wildtrack"))
    parser.add_argument("--cameras", nargs="+", default=("C1", "C2"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the WILDTRACK subset downloader CLI."""
    args = build_parser().parse_args(argv)
    summary = download_wildtrack(args.output_dir, cameras=args.cameras)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
