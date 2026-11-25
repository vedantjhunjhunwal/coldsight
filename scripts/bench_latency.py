"""CPU latency and size of FP32 vs INT8 at 1 and 2 threads (a proxy for a small edge box)."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coldsight.detector import OnnxDetector  # noqa: E402

if __name__ == "__main__":
    out = {}
    for m in sys.argv[1:]:
        for th in (1, 2):
            d = OnnxDetector(m, threads=th)
            x = np.random.default_rng(0).random((1, 3, d.imgsz, d.imgsz), dtype=np.float32)
            for _ in range(10):
                d.raw(x)
            ts = []
            for _ in range(150):
                t = time.perf_counter()
                d.raw(x)
                ts.append(1000 * (time.perf_counter() - t))
            out[f"{Path(m).stem}|threads={th}"] = dict(p50_ms=round(float(np.percentile(ts, 50)), 1),
                                                     p95_ms=round(float(np.percentile(ts, 95)), 1),
                                                     size_mb=round(Path(m).stat().st_size / 2**20, 2))
            print(out[f"{Path(m).stem}|threads={th}"], m, th, flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/latency.json").write_text(json.dumps(out, indent=1))
