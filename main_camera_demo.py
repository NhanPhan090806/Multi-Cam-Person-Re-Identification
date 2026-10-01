"""Repository-surface entry point for the Stage 8 live camera demonstration."""

import sys
from importlib import import_module
from pathlib import Path


def main() -> int:
    """Run the source-tree module even before the project is installed."""
    source_root = Path(__file__).resolve().parent / "src"
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = import_module("multicam_reid.pipeline.camera_demo")
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
