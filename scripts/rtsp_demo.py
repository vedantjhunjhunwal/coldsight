"""Live RTSP run: an ffmpeg camera simulator publishes H.264 over RTSP/TCP to the service.

The service runs in push mode (it is the RTSP listener) so no extra media server is needed
for the test; production pulls from the camera or a media server with listen=False. The camera
is cut for 4 s mid-run to check that the reader recovers.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coldsight.service import Runner  # noqa: E402

URL = "rtsp://127.0.0.1:8554/dock1"


def camera(clip):
    return subprocess.Popen(["ffmpeg", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", str(clip),
                             "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-g", "15",
                             "-f", "rtsp", "-rtsp_transport", "tcp", URL])


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "models/coldsight_v2_int8.onnx"
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    clip = sorted(Path("data/clips/test").glob("*.mp4"))[0]
    run = Runner(URL, model, line_x=640, listen=True)
    run.start()
    time.sleep(1.5)
    cam = camera(clip)
    t0 = time.time()
    cut_done = False
    while time.time() - t0 < secs:
        time.sleep(1)
        if not cut_done and time.time() - t0 > secs / 2:
            cam.terminate(); cam.wait()  # camera drops off the network
            time.sleep(4)
            cut_at = round(time.time() - t0, 1)
            cam = camera(clip)
            cut_done = True
    stats = run.stats()
    run.stop()
    cam.terminate()
    stats.update(model=Path(model).name, seconds=secs, source_fps=15, camera_cut_seconds=4,
                 events_sample=[e.__dict__ for e in run.pipe.events[:6]])
    Path("results").mkdir(exist_ok=True)
    Path("results/rtsp_demo.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str))
