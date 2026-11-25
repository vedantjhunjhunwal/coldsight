"""Metrics: COCO-style mAP (101-point, IoU .50:.95), OCR read accuracy, MOT and counting."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import CLASSES

IOUS = np.linspace(0.5, 0.95, 10)


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)
    ab = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def _ap(rec, prec):
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def mean_ap(preds, gts):
    """preds/gts: list per image of (xyxy[N,4], conf[N] (preds only), cls[N]).
    Returns dict with per-class AP50, AP50-95 and overall mAP50 / mAP50-95."""
    stats = {c: {"tp": [], "conf": [], "n_gt": 0} for c in range(len(CLASSES))}
    for (pb, pc, pk), (gb, gk) in zip(preds, gts):
        for c in range(len(CLASSES)):
            pm, gm = pk == c, gk == c
            p, conf, g = pb[pm], pc[pm], gb[gm]
            stats[c]["n_gt"] += len(g)
            if len(p) == 0:
                continue
            order = np.argsort(-conf)
            p, conf = p[order], conf[order]
            iou = box_iou(p, g)
            tp = np.zeros((len(p), len(IOUS)), bool)
            for ti, thr in enumerate(IOUS):
                used = np.zeros(len(g), bool)
                for i in range(len(p)):
                    if len(g) == 0:
                        break
                    cand = np.where((iou[i] >= thr) & ~used)[0]
                    if len(cand):
                        j = cand[np.argmax(iou[i, cand])]
                        used[j] = True
                        tp[i, ti] = True
            stats[c]["tp"].append(tp)
            stats[c]["conf"].append(conf)
    out = {}
    for c, s in stats.items():
        if s["n_gt"] == 0:
            continue
        if not s["tp"]:
            out[CLASSES[c]] = (0.0, 0.0)
            continue
        tp = np.concatenate(s["tp"])
        conf = np.concatenate(s["conf"])
        o = np.argsort(-conf)
        tp = tp[o]
        aps = []
        for ti in range(len(IOUS)):
            ctp = np.cumsum(tp[:, ti])
            cfp = np.cumsum(~tp[:, ti])
            aps.append(_ap(ctp / s["n_gt"], ctp / np.maximum(ctp + cfp, 1e-9)))
        out[CLASSES[c]] = (aps[0], float(np.mean(aps)))
    return {"per_class": {k: {"AP50": round(v[0], 4), "AP50_95": round(v[1], 4)} for k, v in out.items()},
            "mAP50": round(float(np.mean([v[0] for v in out.values()])), 4),
            "mAP50_95": round(float(np.mean([v[1] for v in out.values()])), 4)}


def load_yolo_split(root: Path, split: str, W=640, H=384):
    imgs = sorted((root / "images" / split).glob("*.jpg"))
    gts = []
    for p in imgs:
        lab = root / "labels" / split / (p.stem + ".txt")
        rows = [list(map(float, l.split())) for l in lab.read_text().splitlines() if l.strip()]
        a = np.array(rows).reshape(-1, 5)
        cx, cy, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
        gts.append((np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), a[:, 0].astype(int)))
    return imgs, gts
