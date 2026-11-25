"""ONNX Runtime detector (FP32 or INT8 QDQ) with letterbox, per-class NMS and box un-mapping."""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from .preprocess import enhance_low_light, letterbox


class OnnxDetector:
    def __init__(self, path, conf=0.25, iou=0.5, threads: int | None = None, low_light=True):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name, self.imgsz = inp.name, int(inp.shape[2])
        self.conf, self.iou, self.low_light = conf, iou, low_light

    def prepare(self, img):
        if self.low_light:
            img = enhance_low_light(img)
        lb, r, left, top = letterbox(img, self.imgsz)
        x = lb[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), (r, left, top)

    def raw(self, x):
        return self.sess.run(None, {self.input_name: x})[0]

    def __call__(self, img):
        x, (r, left, top) = self.prepare(img)
        pred = self.raw(x)[0].T  # (N, 4 + nc)
        scores = pred[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        pred, cls, conf = pred[keep], cls[keep], conf[keep]
        if len(pred) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        xywh = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1).astype(int)
        b = xywh[idx]
        xyxy = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r
        h0, w0 = img.shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0)
        return xyxy.astype(np.float32), conf[idx].astype(np.float32), cls[idx].astype(int)
