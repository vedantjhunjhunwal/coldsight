"""FastAPI service: reads an RTSP (or file) source, runs detect -> track -> read -> count, exposes
/health, /stats, /events and Prometheus /metrics."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from .detector import OnnxDetector
from .monitor import BINS, Monitor, luma_sample
from .stream import LatestFrameReader
from .tracking import DockPipeline

FRAMES = Counter("coldsight_frames_total", "Frames processed")
EVENTS = Counter("coldsight_pallet_events_total", "Pallets crossing the dock line", ["direction", "id_read"])
LAT = Histogram("coldsight_frame_latency_seconds", "Capture-to-result latency",
                buckets=(0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6))
PSI = Gauge("coldsight_input_luma_psi", "Luminance PSI vs training data")
CONF = Gauge("coldsight_mean_confidence", "Rolling mean detection confidence")
DROPPED = Gauge("coldsight_dropped_frames", "Frames skipped to stay real time")
RECON = Gauge("coldsight_stream_reconnects", "Stream reconnects")


class Runner(threading.Thread):
    def __init__(self, source, model, line_x, ref_lumas=None, listen=False):
        super().__init__(daemon=True)
        self.reader = LatestFrameReader(source, listen=listen)
        self.det = OnnxDetector(model, conf=0.3)
        self.pipe = DockPipeline(line_x)
        ref = Path(model).with_name("reference_luma_hist.npy")
        if ref_lumas is None and ref.exists():  # expand the stored histogram back into samples
            counts = np.load(ref)
            ref_lumas = np.repeat((BINS[:-1] + BINS[1:]) / 2, np.maximum(counts // 50, 1))
        cfg = Path(model).with_name("monitor_config.json")
        thr = json.loads(cfg.read_text())["psi_alert"] if cfg.exists() else 2.5
        self.mon = Monitor(ref_lumas if ref_lumas is not None else np.full(100, 120.0), psi_alert=thr)
        self.latencies: list[float] = []
        self.processed = 0
        self.running = True
        self.started = time.time()

    def run(self):
        self.reader.start()
        while self.running:
            got = self.reader.read(timeout=1.0)
            if got is None:
                continue
            idx, ts, frame = got
            xyxy, conf, cls = self.det(frame)
            self.pipe.update(idx, frame, xyxy, conf, cls)
            n_before = len(self.pipe.events)
            self.mon.observe("live", idx, luma_sample(frame), conf)
            lat = time.time() - ts
            self.latencies.append(lat)
            LAT.observe(lat)
            FRAMES.inc()
            self.processed += 1
            for e in self.pipe.events[n_before:]:
                EVENTS.labels(e.direction, str(e.pallet_id is not None)).inc()
            s = self.mon.stats()
            if s["luma_psi"] is not None:
                PSI.set(s["luma_psi"])
            if s["mean_conf"] is not None:
                CONF.set(s["mean_conf"])
            DROPPED.set(self.reader.dropped)
            RECON.set(self.reader.reconnects)

    def stop(self):
        self.running = False
        self.reader.stop()

    def stats(self):
        lat = np.array(self.latencies[-500:]) if self.latencies else np.zeros(1)
        el = max(time.time() - self.started, 1e-6)
        return dict(processed=self.processed, fps=round(self.processed / el, 2),
                    grabbed=self.reader.grabbed, dropped=self.reader.dropped, reconnects=self.reader.reconnects,
                    latency_ms_p50=round(1000 * float(np.percentile(lat, 50)), 1),
                    latency_ms_p95=round(1000 * float(np.percentile(lat, 95)), 1),
                    events=len(self.pipe.events), monitor=self.mon.stats())


def create_app(source=None, model=None, line_x=None, start=True) -> FastAPI:
    source = source or os.environ.get("COLDSIGHT_SOURCE", "rtsp://127.0.0.1:8554/dock1")
    model = model or os.environ.get("COLDSIGHT_MODEL", str(Path(__file__).resolve().parents[1] /
                                                           "models/coldsight_v2_int8.onnx"))
    line_x = float(line_x or os.environ.get("COLDSIGHT_LINE_X", 640))
    app = FastAPI(title="ColdSight")
    runner = Runner(source, model, line_x, listen=os.environ.get("COLDSIGHT_LISTEN") == "1")
    if start:
        runner.start()
    app.state.runner = runner

    @app.get("/health")
    def health():
        return {"status": "ok", "model": Path(model).name, "source": source.split("@")[-1]}

    @app.get("/stats")
    def stats():
        return runner.stats()

    @app.get("/events")
    def events(limit: int = 50):
        return [e.__dict__ for e in runner.pipe.events[-limit:]]

    @app.get("/metrics")
    def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
