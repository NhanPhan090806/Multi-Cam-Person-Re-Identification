"""Model-backed wiring check, using an artificial schedule of existing smoke footage.

This is explicitly NOT a real non-overlapping camera dataset or accuracy test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multicam_reid.inputs.handoff_recordings import SessionRecorder
from multicam_reid.inputs.video import VideoFileSource
from multicam_reid.pipeline.handoff_replay import main as replay
from multicam_reid.types import FramePacket


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("outputs/stage3/smoke_input.mp4"))
    parser.add_argument("--record-dir", type=Path, default=Path("recordings/handoff_smoke"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/handoff/smoke"))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--gap-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.frames < 10 or args.gap_seconds < 2:
        parser.error("use at least ten frames and a two-second blind gap")
    frames = []
    with VideoFileSource(args.video, "A") as source:
        for packet in source:
            frames.append(packet.frame)
            if len(frames) >= args.frames:
                break
    if len(frames) < 10:
        raise ValueError("not enough source frames")
    step = 1.0 / 12.0
    with SessionRecorder(args.record_dir, ("A", "B")) as recorder:
        for frame_id, image in enumerate(frames):
            timestamp = frame_id * step
            recorder.write(FramePacket("A", frame_id, timestamp, image), elapsed=timestamp)
        end = len(frames) * step
        blank = np.full_like(frames[0], 32)
        recorder.write(FramePacket("A", len(frames), end, blank), elapsed=end)
        recorder.write(FramePacket("B", 0, end, blank), elapsed=end)
        for frame_id, image in enumerate(frames, start=1):
            timestamp = end + args.gap_seconds + (frame_id - 1) * step
            recorder.write(FramePacket("B", frame_id, timestamp, image), elapsed=timestamp)
        recording = recorder.directory
    print("Artificial smoke schedule only; reused source images are not independent cameras.")
    return replay(["--recording", str(recording), "--output-dir", str(args.output_dir)])


if __name__ == "__main__":
    raise SystemExit(main())
