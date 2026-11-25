"""Log training runs, evaluation results and model files to MLflow (local file store: ./mlruns)."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow  # noqa: E402

mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
mlflow.set_experiment("coldsight-dock-detector")
det = json.loads(Path("results/detection.json").read_text())
reg = {d["version"]: d for d in json.loads(Path("models/registry.json").read_text())}
for ver in ("v1", "v2"):
    info_p = Path("runs", ver, "train_info.json")
    if not info_p.exists():
        continue
    info = json.loads(info_p.read_text())
    with mlflow.start_run(run_name=ver):
        mlflow.log_params({k: v for k, v in info.items() if k != "weights"})
        for row in csv.DictReader(open(Path("runs", ver, "results.csv"))):
            ep = int(float(row["epoch"]))
            for k, v in row.items():
                if k.strip().startswith(("metrics/", "train/", "val/")):
                    mlflow.log_metric(k.strip().replace("(B)", "").replace("/", "_"), float(v), step=ep)
        for key, m in det.items():
            stem, split, *rest = key.split("|")
            if stem.startswith(f"coldsight_{ver}_") and not rest:
                prec = stem.split(f"coldsight_{ver}_")[1]
                mlflow.log_metric(f"{split}_{prec}_mAP50_95", m["mAP50_95"])
                mlflow.log_metric(f"{split}_{prec}_mAP50", m["mAP50"])
        for v, d in reg.items():
            if v.startswith(ver + "-"):
                mlflow.set_tag(f"sha256_{v}", d["sha256"])
                mlflow.log_artifact(str(Path("models", d["file"])), artifact_path="onnx")
print("logged to", mlflow.get_tracking_uri())
