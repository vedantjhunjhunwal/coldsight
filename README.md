# ColdSight — live pallet tracking and label reading for cold-store dock cameras

ColdSight watches a dock-door camera over RTSP, detects and tracks **pallets, pallet labels,
forklifts and people**, reads each pallet's 8-digit ID from its label, and logs every pallet that
crosses the dock line (in/out, with its ID). It runs on CPU as an INT8 ONNX model, exposes
Prometheus metrics, watches its own input for drift, and turns uncertain frames into a
relabelling queue that feeds the next model version.

> **Data note.** All images and video are **synthetic**: a procedural cold-store dock renderer
> (`coldsight/synth.py`) with exact ground truth, normal lighting for training and a held-out
> low-light / frosted-lens condition for testing. Numbers below are on that synthetic data;
> real camera footage would need its own labelled set.

```
RTSP camera ─► LatestFrameReader ─► OpenCV preprocessing ─► YOLOv8n (ONNX INT8, 416 px)
 (reconnects,     (drops stale        (letterbox, low-light       │ pallet · label · forklift · person
  backoff)         frames)             CLAHE+gamma)               ▼
                                                    ByteTrack per class (Kalman + 2-stage IoU)
                                                                  │
               label → front-most pallet ◄────────────────────────┤
               OCR crop cleanup → Tesseract → GS1 check digit     │
               → confirm on ≥2 agreeing reads (≥60% share)        ▼
                                              dock-line crossing events {in/out, pallet ID}
Monitor: luminance PSI vs training data, confidence health, relabel queue → Label Studio tasks
Service: FastAPI /health /stats /events /metrics (Prometheus) · Docker · Azure Container Apps
```

## Results

**Detection** (mAP, COCO-style 101-point, own evaluator in `coldsight/evaluate.py`; 240 test and
240 low-light images, 1280×768, held-out seeds):

| Model | Size | CPU p50, 1 thread | mAP50 normal | mAP50-95 normal | mAP50 low light | mAP50-95 low light |
|---|---|---|---|---|---|---|
| v1 FP32 | 12.1 MB | — | 0.983 | 0.911 | 0.987 | 0.837 |
| v1 INT8 (head FP32) | 5.8 MB | — | 0.983 | 0.908 | 0.987 | 0.832 |
| v2 FP32 | 12.1 MB | 38.0 ms | 0.981 | 0.898 | 0.990 | 0.855 |
| **v2 INT8 (deployed)** | **5.8 MB** | **29.3 ms** | 0.980 | 0.890 | **0.990** | **0.852** |

**Quantization ablation** (v1, static QDQ INT8, 120 calibration frames):

| What stays FP32 | Size | p50, 1 thread | mAP50-95 normal |
|---|---|---|---|
| nothing | 3.4 MB | — | **0.000** — box coords (0-416) and class scores (0-1) share one INT8 scale after the head's Concat, so scores collapse |
| head decode ops only | 3.4 MB | 20.0 ms | 0.883 (−2.8 pts) |
| whole Detect head (shipped) | 5.8 MB | 29.3 ms | 0.908 (−0.3 pts) |

**End to end on full clips** (v2 INT8, 8 normal + 8 low-light clips, 150 frames each, 15 fps):

| | Normal light | Low light |
|---|---|---|
| Pallet tracking MOTA / IDF1 | 0.936 / 0.939 | 0.946 / 0.972 |
| ID switches | 6 | 2 |
| Dock-line counts | 21/21, all 8 clips exact | 16/16, all 8 clips exact |
| Pallet IDs read correctly | 13 / 21 | 11 / 16 |
| Wrong pallet IDs booked | **0** | 1 |
| Flagged for manual scan | 8 | 4 |
| Pipeline time per frame (2 vCPU) | 68 ms | 78 ms |

Unread pallets are mostly labels that are physically covered (e.g. by the forklift mast) or never
fully in view; the reader refuses rather than guessing — a wrong ID is costlier than a manual scan.

**Label OCR alone** (on ground-truth label boxes): crop cleanup (inset, upscale to ~64 px, CLAHE,
adaptive threshold with Otsu fallback, border-blob removal) lifts exact single reads from **3% → 68%**
in normal light and **5% → 52%** in low light; the GS1 check digit rejects most misreads
(0.3% / 7.8% of accepted reads wrong before multi-frame voting).

**Live RTSP run** (`scripts/rtsp_demo.py`, 60 s, ffmpeg camera simulator publishing H.264 over
RTSP/TCP, camera cut for 4 s mid-run): 11.1 fps processed on 2 vCPUs shared with the camera's
encoder, capture-to-result latency p50 38 ms / p95 186 ms, 166 stale frames dropped to stay
real time, 1 automatic reconnect, all 6 pallet crossings logged with the correct ID.

**Deployment feedback loop** (`scripts/feedback_loop.py`):
1. v1 (trained on normal light only) watches 12 unlabelled **low-light** camera streams.
2. Drift: luminance PSI vs the training set, threshold calibrated on validation cameras
   (1.25 × worst normal camera = 1.99). **12/12 low-light cameras flagged within 30 frames (2 s);
   0/8 normal test cameras flagged.** (Margin is tight: the worst normal test camera reached 1.97.)
3. Frames with uncertain detections (0.25–0.5 confidence), spaced ≥ 8 frames apart, go to a
   relabel queue: **127 frames**, exported as **Label Studio tasks with v1 pre-annotations**.
4. Corrections come back as a Label Studio export (simulated here from ground truth; 154 false
   boxes deleted, 4 added, 823 kept), are QA-checked and converted to YOLO.
5. v2 = v1 fine-tuned 8 epochs on train + feedback frames: **low-light mAP50-95 +2.0 pts
   (INT8 0.832 → 0.852)**; normal-light validation flat (0.921 → 0.920), normal test −1.8 pts.

## Annotation workflow

- Bulk labels are **CVAT for video 1.1 XML** (tracks with keyframes, `outside`, `occluded`, and a
  `pallet_id` attribute). The YOLO training set is built *from* the CVAT export
  (`read_cvat_video` → `yolo_lines`), exactly as with hand-labelled data.
- The relabel queue goes out as **Label Studio** JSON (`predictions` from the model) and comes back
  as a Label Studio export (`annotations`); `read_labelstudio` ignores unreviewed predictions.
- `qa_frame` checks every frame: out-of-frame and tiny boxes, duplicates, unknown classes, and
  labels not sitting on any pallet (it flagged 2 genuine edge cases in validation and 1 in feedback).

## Run it

```bash
pip install -r requirements-train.txt            # + apt install tesseract-ocr ffmpeg
python scripts/make_data.py --out data           # render clips + CVAT XML, build YOLO set (≈20 min)
python scripts/train.py --name v1 --epochs 20    # YOLOv8n, CPU ≈ 55 min
python scripts/export_quantize.py --weights runs/v1/weights/best.pt --version v1
python scripts/feedback_loop.py                  # drift + relabel queue -> data/feedback
python scripts/train.py --name v2 --weights runs/v1/weights/best.pt \
       --data data/yolo/feedback.yaml --epochs 8 --lr0 0.002
python scripts/export_quantize.py --weights runs/v2/weights/best.pt --version v2
python scripts/eval_detection.py models/coldsight_v2_int8.onnx
python scripts/eval_tracking.py models/coldsight_v2_int8.onnx
python scripts/eval_ocr.py && python scripts/bench_latency.py models/*.onnx
python scripts/rtsp_demo.py models/coldsight_v2_int8.onnx 60
python scripts/log_mlflow.py                     # runs, metrics, ONNX files + hashes -> ./mlruns
python -m pytest -q                              # 25 tests
```

Serve: `COLDSIGHT_SOURCE=rtsp://user:pass@camera/stream python -m coldsight.service`
(or `docker build -t coldsight . && docker run -e COLDSIGHT_SOURCE=... -p 8000:8000 coldsight`).
Azure: `deploy/azure/deploy.sh` (Container Registry build + Container Apps, camera URL as a secret).
Edge: `deploy/edge/Dockerfile.arm64`.

## Model versioning

`models/registry.json` records every exported model: version, file, SHA-256, size, source weights,
quantization settings and date. `scripts/log_mlflow.py` logs both training runs (per-epoch losses
and mAP), evaluation metrics and the ONNX files to MLflow. `models/monitor_config.json` and
`models/reference_luma_hist.npy` ship the drift threshold and reference histogram with the model.

## Limitations (read before quoting numbers)

- Synthetic data only; the renderer is simple (flat-shaded sprites, side-on camera).
- The Dockerfile and Azure script are provided but were **not built or run** here (no Docker
  daemon or Azure subscription in the build environment). The service itself is tested end to
  end through FastAPI's test client and the RTSP run.
- The RTSP test runs the service as the RTSP listener with the camera publishing into it; pulling
  from a real camera or media server uses the same reader with `listen=False` but was not tested here.
- Annotator corrections in the feedback loop are simulated from ground truth.
- ARM64 edge image not built; latency is measured on x86 CPU threads as a proxy.
- v2 trades a little normal-light accuracy for low-light accuracy (see table); a production
  rollout would A/B the two per camera.
