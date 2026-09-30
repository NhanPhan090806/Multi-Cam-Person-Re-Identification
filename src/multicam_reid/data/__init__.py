"""Dataset download, validation, and parsing helpers."""

from multicam_reid.data.market1501 import (
    DatasetLayoutError,
    Market1501Summary,
    download_market1501,
    find_market1501_root,
    inspect_market1501,
)
from multicam_reid.data.wildtrack import (
    WildtrackLayoutError,
    WildtrackSummary,
    download_wildtrack,
    inspect_wildtrack,
)

__all__ = [
    "DatasetLayoutError",
    "Market1501Summary",
    "download_market1501",
    "find_market1501_root",
    "inspect_market1501",
    "WildtrackLayoutError",
    "WildtrackSummary",
    "download_wildtrack",
    "inspect_wildtrack",
]
