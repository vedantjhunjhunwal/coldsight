"""Export a trained model to ONNX (FP32), quantize to INT8 (QDQ, static calibration), register both."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coldsight.preprocess import enhance_low_light, letterbox  # noqa: E402


class Calib:
    def __init__(self, files, imgsz, name):
        self.files, self.imgsz, self.name, self.i = files, imgsz, name, 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        img = enhance_low_light(cv2.imread(str(self.files[self.i])))
        self.i += 1
        lb, *_ = letterbox(img, self.imgsz)
        x = lb[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return {self.name: np.ascontiguousarray(x)}

    def rewind(self):
        self.i = 0


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def register(entry, reg=Path("models/registry.json")):
    data = json.loads(reg.read_text()) if reg.exists() else []
    data = [d for d in data if d["version"] != entry["version"]] + [entry]
    reg.write_text(json.dumps(data, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--calib", type=int, default=120)
    ap.add_argument("--head", choices=["fp32", "decode-fp32", "int8"], default="fp32",
                    help="fp32: whole Detect head in FP32 (default); decode-fp32: only decode ops in FP32; "
                         "int8: everything quantized (breaks the scores - kept for the ablation)")
    ap.add_argument("--suffix", default="int8")
    a = ap.parse_args()
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import onnx
    from ultralytics import YOLO

    out = Path("models")
    out.mkdir(exist_ok=True)
    fp32 = out / f"coldsight_{a.version}_fp32.onnx"
    exported = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, opset=17, simplify=True, dynamic=False)
    shutil.move(exported, fp32)
    pre = out / f"_pre_{a.version}.onnx"
    quant_pre_process(str(fp32), str(pre))
    m = onnx.load(str(pre))
    # The Detect head concatenates box coords (0-416) and class scores (0-1) into one tensor; one INT8
    # scale for both wipes out the scores. Keep the head's decode ops (everything but its convs) in FP32.
    in_head = [n for n in m.graph.node if n.name.startswith("/model.22/")]
    head = {"fp32": [n.name for n in in_head],
            "decode-fp32": [n.name for n in in_head if n.op_type != "Conv"], "int8": []}[a.head]
    rng = np.random.default_rng(0)
    files = sorted(Path("data/yolo/images/train").glob("*.jpg"))
    files = [files[i] for i in rng.choice(len(files), size=min(a.calib, len(files)), replace=False)]
    int8 = out / f"coldsight_{a.version}_{a.suffix}.onnx"
    t = time.time()
    quantize_static(str(pre), str(int8), Calib(files, a.imgsz, m.graph.input[0].name), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=head)
    pre.unlink()
    for p, prec in ((fp32, "fp32"), (int8, a.suffix)):
        register(dict(version=f"{a.version}-{prec}", file=p.name, sha256=sha256(p),
                      size_mb=round(p.stat().st_size / 2**20, 2), source_weights=a.weights, imgsz=a.imgsz,
                      calibration_images=a.calib if p == int8 else None,
                      head=a.head if p == int8 else None,
                      created=time.strftime("%Y-%m-%d %H:%M")))
    print(json.dumps(json.loads(Path("models/registry.json").read_text())[-2:], indent=1))
    print("quantize seconds", round(time.time() - t, 1))


if __name__ == "__main__":
    main()
