"""Surface entry point: replay a recorded walk between separate camera locations."""

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from multicam_reid.pipeline.handoff_replay import main as run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
