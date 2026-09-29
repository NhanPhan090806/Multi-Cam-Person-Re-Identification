import numpy as np
import pytest

from multicam_reid.types import FramePacket


def test_frame_packet_accepts_valid_bgr_frame() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    packet = FramePacket(camera_id="C1", frame_id=7, timestamp=1.5, frame=frame)

    assert packet.camera_id == "C1"
    assert packet.frame_id == 7
    assert packet.frame is frame


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"camera_id": ""}, "camera_id"),
        ({"frame_id": -1}, "frame_id"),
        ({"timestamp": -0.1}, "timestamp"),
        ({"frame": np.zeros((10, 10), dtype=np.uint8)}, "shape"),
        ({"frame": np.zeros((0, 10, 3), dtype=np.uint8)}, "non-empty"),
    ],
)
def test_frame_packet_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "camera_id": "C1",
        "frame_id": 0,
        "timestamp": 0.0,
        "frame": np.zeros((10, 10, 3), dtype=np.uint8),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        FramePacket(**values)  # type: ignore[arg-type]
