"""Stage 4 wrapper connecting Stage 3 person crops to rolling Re-ID tracklets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from multicam_reid.pipeline.single_camera import ProcessedFrame
from multicam_reid.reid.tracklets import TrackletAppearance, TrackletEmbeddingStore
from multicam_reid.types import FramePacket


class Stage3CameraPipeline(Protocol):
    def process(self, packet: FramePacket) -> ProcessedFrame: ...


@dataclass(frozen=True, slots=True)
class ReIDProcessedFrame:
    """One Stage 3 result plus the updated appearances visible in that frame."""

    stage3: ProcessedFrame
    appearances: tuple[TrackletAppearance, ...]


class ReIDCameraPipeline:
    """Compose a Stage 3 camera pipeline with the shared Stage 4 Re-ID store."""

    def __init__(
        self,
        *,
        stage3: Stage3CameraPipeline,
        tracklets: TrackletEmbeddingStore,
    ) -> None:
        self.stage3 = stage3
        self.tracklets = tracklets

    def process(self, packet: FramePacket) -> ReIDProcessedFrame:
        """Detect, track, crop, and update rolling appearance representations."""
        stage3_result = self.stage3.process(packet)
        appearances = self.tracklets.update(
            stage3_result.crops,
            current_frame_id=packet.frame_id,
        )
        return ReIDProcessedFrame(stage3=stage3_result, appearances=appearances)
