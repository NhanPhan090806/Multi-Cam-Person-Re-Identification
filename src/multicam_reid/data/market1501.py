"""Download and validate the Market-1501 person Re-ID dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

KAGGLE_HANDLE = "pengcw1/market-1501"
DATASET_DIRECTORY = "Market-1501-v15.09.15"
REQUIRED_SPLITS: Mapping[str, str] = {
    "train": "bounding_box_train",
    "query": "query",
    "gallery": "bounding_box_test",
}
FILENAME_PATTERN = re.compile(
    # The selected Kaggle mirror contains a small number of otherwise valid
    # images with a duplicated ".jpg.jpg" suffix.
    r"^(?P<person_id>-?\d+)_c(?P<camera_id>\d+)s\d+_\d+_\d+(?:\.jpg){1,2}$",
    flags=re.IGNORECASE,
)

DatasetDownloader = Callable[..., str]


class DatasetLayoutError(ValueError):
    """Raised when Market-1501 does not have its expected directory layout."""


@dataclass(frozen=True, slots=True)
class Market1501Sample:
    """Identity metadata encoded in a Market-1501 image filename."""

    path: Path
    person_id: int
    camera_id: int

    @property
    def is_junk(self) -> bool:
        """Return whether the official protocol marks this image as junk."""
        return self.person_id == -1


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """Basic integrity statistics for one Market-1501 split."""

    num_images: int
    num_identities: int
    camera_ids: tuple[int, ...]
    num_junk_images: int


@dataclass(frozen=True, slots=True)
class Market1501Summary:
    """Validated paths and statistics for all official splits."""

    dataset_root: Path
    train: SplitSummary
    query: SplitSummary
    gallery: SplitSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result = asdict(self)
        result["dataset_root"] = str(self.dataset_root)
        return result


def _has_required_splits(path: Path) -> bool:
    return all((path / directory).is_dir() for directory in REQUIRED_SPLITS.values())


def find_market1501_root(search_root: Path) -> Path:
    """Find the official Market-1501 directory under common download layouts."""
    root = search_root.expanduser().resolve()
    candidates = (
        root,
        root / DATASET_DIRECTORY,
        root / "market1501" / DATASET_DIRECTORY,
    )
    for candidate in candidates:
        if _has_required_splits(candidate):
            return candidate.resolve()

    expected = ", ".join(str(candidate) for candidate in candidates)
    raise DatasetLayoutError(
        "Could not find a complete Market-1501 dataset. "
        f"Expected train/query/gallery directories under one of: {expected}"
    )


def parse_market1501_filename(path: Path) -> Market1501Sample:
    """Parse person and camera IDs encoded in a Market-1501 filename."""
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid Market-1501 filename: {path.name}")
    return Market1501Sample(
        path=path,
        person_id=int(match.group("person_id")),
        camera_id=int(match.group("camera_id")),
    )


def _inspect_split(split_dir: Path) -> SplitSummary:
    samples = [parse_market1501_filename(path) for path in sorted(split_dir.glob("*.jpg"))]
    if not samples:
        raise DatasetLayoutError(f"No JPG images found in Market-1501 split: {split_dir}")

    valid_samples = [sample for sample in samples if not sample.is_junk]
    return SplitSummary(
        num_images=len(samples),
        num_identities=len({sample.person_id for sample in valid_samples}),
        camera_ids=tuple(sorted({sample.camera_id for sample in samples})),
        num_junk_images=len(samples) - len(valid_samples),
    )


def inspect_market1501(search_root: Path) -> Market1501Summary:
    """Validate Market-1501 and calculate split statistics."""
    dataset_root = find_market1501_root(search_root)
    summaries = {
        split: _inspect_split(dataset_root / directory)
        for split, directory in REQUIRED_SPLITS.items()
    }
    return Market1501Summary(
        dataset_root=dataset_root,
        train=summaries["train"],
        query=summaries["query"],
        gallery=summaries["gallery"],
    )


def download_market1501(
    output_dir: Path,
    *,
    downloader: DatasetDownloader | None = None,
) -> Market1501Summary:
    """Download the public Kaggle mirror, then validate its extracted contents."""
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if downloader is None:
        import kagglehub

        downloader = kagglehub.dataset_download

    downloaded_path = Path(
        downloader(KAGGLE_HANDLE, output_dir=str(destination))
    ).expanduser()
    return inspect_market1501(downloaded_path)


def build_parser() -> argparse.ArgumentParser:
    """Create the Market-1501 downloader command-line parser."""
    parser = argparse.ArgumentParser(
        description="Download and validate the Market-1501 Kaggle mirror."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reid/market1501"),
        help="Destination that will contain Market-1501-v15.09.15.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the downloader CLI."""
    args = build_parser().parse_args(argv)
    summary = download_market1501(args.output_dir)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
