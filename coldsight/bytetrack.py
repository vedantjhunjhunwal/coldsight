"""Compact ByteTrack (Zhang et al., 2022): Kalman motion model + two-stage association.

Stage 1 matches high-score detections to all live tracks by IoU; stage 2 matches the
leftover low-score detections (often occluded objects) to still-unmatched tracks, which
is what keeps IDs alive through partial occlusion. Unmatched high-score detections start
tentative tracks that are confirmed on their next match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)
    ab = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


class _KF:
    """Constant-velocity Kalman filter on (cx, cy, w, h)."""
    F = np.eye(8); F[:4, 4:] = np.eye(4)
    Hm = np.eye(4, 8)

    def __init__(self, box):
        cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], float)
        s = max(w, h)
        self.P = np.diag([s, s, s, s, 10 * s, 10 * s, 10 * s, 10 * s]) ** 2 * 0.01

    def predict(self):
        s = max(self.x[2], self.x[3])
        Q = np.diag([s / 20] * 4 + [s / 160] * 4) ** 2
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, box):
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])
        s = max(self.x[2], self.x[3])
        R = np.diag([s / 20] * 4) ** 2
        S = self.Hm @ self.P @ self.Hm.T + R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.Hm @ self.x)
        self.P = (np.eye(8) - K @ self.Hm) @ self.P

    def box(self):
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class _Track:
    tid: int
    kf: _KF
    score: float
    confirmed: bool = False
    lost: int = 0
    last_box: np.ndarray | None = None


@dataclass
class Tracked:
    xyxy: np.ndarray
    tracker_id: np.ndarray
    confidence: np.ndarray


class ByteTrack:
    def __init__(self, high=0.5, low=0.1, new_track=0.6, match_iou=0.2, low_match_iou=0.5, buffer=30):
        self.high, self.low, self.new_track = high, low, new_track
        self.match_iou, self.low_match_iou, self.buffer = match_iou, low_match_iou, buffer
        self.tracks: list[_Track] = []
        self._next = 1

    @staticmethod
    def _match(tracks, boxes, thr):
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        iou = _iou(np.array([t.kf.box() for t in tracks]), boxes)
        r, c = linear_sum_assignment(-iou)
        pairs = [(i, j) for i, j in zip(r, c) if iou[i, j] >= thr]
        mi = {i for i, _ in pairs}
        mj = {j for _, j in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mi], [j for j in range(len(boxes)) if j not in mj]

    def update(self, xyxy, conf) -> Tracked:
        xyxy, conf = np.asarray(xyxy, float).reshape(-1, 4), np.asarray(conf, float)
        for t in self.tracks:
            t.kf.predict()
        hi, lo = conf >= self.high, (conf >= self.low) & (conf < self.high)
        hb, hs, lb, ls = xyxy[hi], conf[hi], xyxy[lo], conf[lo]
        live = [t for t in self.tracks if t.confirmed]
        tent = [t for t in self.tracks if not t.confirmed]
        updated = []
        # stage 1: high-score dets vs confirmed (incl. lost) tracks
        pairs, um_t, um_d = self._match(live, hb, self.match_iou)
        for i, j in pairs:
            self._hit(live[i], hb[j], hs[j]); updated.append(live[i])
        # stage 2: low-score dets vs confirmed tracks that were visible last frame
        rest = [live[i] for i in um_t if live[i].lost == 0]
        pairs2, _, _ = self._match(rest, lb, self.low_match_iou)
        for i, j in pairs2:
            self._hit(rest[i], lb[j], ls[j]); updated.append(rest[i])
        # stage 3: tentative tracks vs leftover high-score dets
        hb2, hs2 = hb[um_d], hs[um_d]
        pairs3, um_tent, um_d2 = self._match(tent, hb2, self.match_iou)
        for i, j in pairs3:
            tent[i].confirmed = True
            self._hit(tent[i], hb2[j], hs2[j]); updated.append(tent[i])
        upd = {id(t) for t in updated}
        keep = []
        for t in self.tracks:
            if id(t) in upd:
                keep.append(t)
            elif t.confirmed:
                t.lost += 1
                if t.lost <= self.buffer:
                    keep.append(t)
            # unmatched tentative tracks are dropped
        for j in um_d2:
            if hs2[j] >= self.new_track:
                t = _Track(self._next, _KF(hb2[j]), float(hs2[j]), last_box=hb2[j])
                self._next += 1
                keep.append(t)
        self.tracks = keep
        out = [t for t in self.tracks if t.confirmed and t.lost == 0]
        return Tracked(np.array([t.last_box for t in out]).reshape(-1, 4),
                       np.array([t.tid for t in out], int), np.array([t.score for t in out]))

    @staticmethod
    def _hit(t, box, score):
        t.kf.update(box)
        t.lost, t.score, t.last_box = 0, float(score), np.asarray(box, float)
