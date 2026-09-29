from pathlib import Path

import pytest

from multicam_reid.data import market1501
from multicam_reid.data.market1501 import (
    DatasetLayoutError,
    download_market1501,
    find_market1501_root,
    inspect_market1501,
    parse_market1501_filename,
)


def _make_market1501(root: Path) -> Path:
    dataset_root = root / "Market-1501-v15.09.15"
    filenames = {
        "bounding_box_train": [
            "0001_c1s1_000001_00.jpg",
            "0001_c2s1_000002_00.jpg",
            "0002_c1s1_000003_00.jpg",
        ],
        "query": ["0003_c1s1_000004_00.jpg"],
        "bounding_box_test": [
            "0003_c2s1_000005_00.jpg",
            "-1_c3s1_000006_00.jpg",
        ],
    }
    for directory, names in filenames.items():
        split_dir = dataset_root / directory
        split_dir.mkdir(parents=True)
        for name in names:
            (split_dir / name).touch()
    return dataset_root


@pytest.mark.parametrize(
    "relative_input",
    [
        ".",
        "market1501",
        "market1501/Market-1501-v15.09.15",
    ],
)
def test_find_market1501_root_accepts_common_layouts(
    tmp_path: Path, relative_input: str
) -> None:
    expected = _make_market1501(tmp_path / "market1501")
    search_root = (tmp_path / relative_input).resolve()

    assert find_market1501_root(search_root) == expected.resolve()


def test_find_market1501_root_rejects_missing_splits(tmp_path: Path) -> None:
    incomplete = tmp_path / "Market-1501-v15.09.15"
    incomplete.mkdir()

    with pytest.raises(DatasetLayoutError, match="Could not find"):
        find_market1501_root(incomplete)


@pytest.mark.parametrize(
    ("filename", "person_id", "camera_id", "is_junk"),
    [
        ("0001_c1s1_000001_00.jpg", 1, 1, False),
        ("1501_c6s3_001234_02.jpg", 1501, 6, False),
        ("-1_c3s1_000006_00.jpg", -1, 3, True),
        ("1488_c1s6_023021_00.jpg.jpg", 1488, 1, False),
    ],
)
def test_parse_market1501_filename(
    filename: str, person_id: int, camera_id: int, is_junk: bool
) -> None:
    sample = parse_market1501_filename(Path(filename))

    assert sample.person_id == person_id
    assert sample.camera_id == camera_id
    assert sample.is_junk is is_junk


def test_parse_market1501_filename_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="Invalid Market-1501 filename"):
        parse_market1501_filename(Path("person-one.jpg"))


def test_inspect_market1501_counts_images_identities_and_junk(tmp_path: Path) -> None:
    dataset_root = _make_market1501(tmp_path)

    summary = inspect_market1501(dataset_root)

    assert summary.dataset_root == dataset_root.resolve()
    assert summary.train.num_images == 3
    assert summary.train.num_identities == 2
    assert summary.train.camera_ids == (1, 2)
    assert summary.query.num_images == 1
    assert summary.query.num_identities == 1
    assert summary.gallery.num_images == 2
    assert summary.gallery.num_identities == 1
    assert summary.gallery.num_junk_images == 1
    assert summary.to_dict()["dataset_root"] == str(dataset_root.resolve())


def test_download_market1501_delegates_and_validates(tmp_path: Path) -> None:
    output_dir = tmp_path / "reid" / "market1501"

    def fake_downloader(handle: str, *, output_dir: str) -> str:
        assert handle == "pengcw1/market-1501"
        dataset_root = _make_market1501(Path(output_dir))
        return str(dataset_root.parent)

    summary = download_market1501(output_dir, downloader=fake_downloader)

    assert summary.dataset_root.name == "Market-1501-v15.09.15"
    assert summary.train.num_images == 3


def test_market1501_cli_prints_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_root = _make_market1501(tmp_path)
    summary = inspect_market1501(dataset_root)
    monkeypatch.setattr(market1501, "download_market1501", lambda output_dir: summary)

    assert market1501.main(["--output-dir", str(tmp_path / "download")]) == 0
    assert '"train"' in capsys.readouterr().out
