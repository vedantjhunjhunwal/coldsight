"""mAP of ONNX models on test (normal) and shift (low light), with/without low-light preprocessing."""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coldsight.detector import OnnxDetector  # noqa: E402
from coldsight.evaluate import load_yolo_split, mean_ap  # noqa: E402

ROOT = Path("data/yolo")


def evaluate(model, split, low_light=True):
    det = OnnxDetector(model, conf=0.001, iou=0.6, low_light=low_light)
    imgs, gts = load_yolo_split(ROOT, split, 1280, 768)
    preds = [det(cv2.imread(str(p))) for p in imgs]
    return mean_ap(preds, gts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["test", "shift"])
    ap.add_argument("--ablate-preprocess", action="store_true")
    ap.add_argument("--out", default="results/detection.json")
    a = ap.parse_args()
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for m in a.models:
        for s in a.splits:
            key = f"{Path(m).stem}|{s}"
            out[key] = evaluate(m, s, True)
            print(key, out[key]["mAP50"], out[key]["mAP50_95"], flush=True)
            if a.ablate_preprocess:
                k2 = key + "|no_low_light_fix"
                out[k2] = evaluate(m, s, False)
                print(k2, out[k2]["mAP50"], out[k2]["mAP50_95"], flush=True)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
