# Model card — ColdSight detector v2 (INT8)

- **Architecture:** YOLOv8n (3.1 M params), 416×416 letterbox input, 4 classes: pallet, label, forklift, person.
- **Training:** COCO-pretrained YOLOv8n → v1: 20 epochs on 800 synthetic normal-light frames
  (4,336 boxes) → v2: 8 epochs fine-tune on those + 127 relabelled low-light frames from the feedback loop.
- **Export:** ONNX opset 17; static INT8 (QDQ, per-channel weights, MinMax calibration on 120 frames),
  Detect head kept in FP32. 5.8 MB, SHA-256 prefix in `models/registry.json`.
- **Evaluation:** held-out synthetic seeds; normal light mAP50-95 0.890, low light 0.852 (see README).
- **Intended use:** demo / prototype of a dock-door pallet counter. Not validated on real footage.
- **Known failure modes:** small far-lane labels (< ~120 px wide) and labels covered by the forklift
  mast are not read (flagged for manual scan); localisation (mAP50-95) drops in low light more than
  detection (mAP50); drift threshold margin is small on the synthetic normal cameras.
