import time
from pathlib import Path

import cv2
import pytest

MODEL = Path(__file__).resolve().parents[1] / "models" / "coldsight_v2_int8.onnx"


@pytest.mark.skipif(not MODEL.exists(), reason="model file not present")
def test_service_end_to_end(tmp_path):
    from fastapi.testclient import TestClient

    from coldsight.service import create_app
    from coldsight.synth import H, W, Sequence

    clip = tmp_path / "dock.mp4"
    seq = Sequence(203, 60)
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    for t in range(60):
        vw.write(seq.frame(t)[0])
    vw.release()
    app = create_app(source=str(clip), model=str(MODEL), line_x=640)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    for _ in range(60):
        if client.get("/stats").json()["processed"] >= 20:
            break
        time.sleep(0.5)
    stats = client.get("/stats").json()
    assert stats["processed"] >= 20 and stats["monitor"]["mean_conf"] > 0.5
    assert isinstance(client.get("/events").json(), list)
    assert "coldsight_frames_total" in client.get("/metrics").text
    app.state.runner.stop()
