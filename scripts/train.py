"""Train the YOLOv8n detector and record the run in the model registry (+ MLflow if installed)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.01)
    a = ap.parse_args()
    from ultralytics import YOLO

    t = time.time()
    model = YOLO(a.weights)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, lr0=a.lr0, device="cpu",
                workers=2, cache="ram", project=str(Path("runs").resolve()), name=a.name, exist_ok=True, seed=0, deterministic=True,
                plots=False, patience=100, verbose=False)
    best = Path("runs") / a.name / "weights" / "best.pt"
    info = dict(name=a.name, base=a.weights, data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                lr0=a.lr0, train_minutes=round((time.time() - t) / 60, 1), weights=str(best))
    Path("runs", a.name, "train_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
