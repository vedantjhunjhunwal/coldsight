"""Live model monitoring: input drift (PSI on pixel luminance), confidence health, and a relabel queue.

Drift compares the luminance histogram of the last `window` frames (64x40 thumbnails) with the
training set's histogram, so a whole-camera lighting change shows up while scene content moving
around does not.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

BINS = np.linspace(0, 255, 17)


def luma_hist(values):
    h, _ = np.histogram(values, bins=BINS)
    h = h.astype(np.float64) + 1e-3
    return h / h.sum()


def luma_sample(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 40), interpolation=cv2.INTER_AREA).ravel()


def psi(ref, cur):
    return float(np.sum((cur - ref) * np.log(cur / ref)))


class Monitor:
    def __init__(self, ref_pixels, window=150, psi_alert=0.25, uncertain=(0.25, 0.5), queue_budget=200,
                 min_gap=8):
        self.ref = luma_hist(ref_pixels)
        self.hists = deque(maxlen=window)
        self.confs = deque(maxlen=window)
        self.psi_alert, self.uncertain = psi_alert, uncertain
        self.queue: list[tuple] = []  # (key, frame_idx, uncertainty)
        self.budget, self.min_gap = queue_budget, min_gap
        self._last_q = -10**9
        self.alerts = 0
        self.frames = 0

    def observe(self, key, frame_idx, pixels, confs):
        self.frames += 1
        h, _ = np.histogram(pixels, bins=BINS)
        self.hists.append(h)
        self.confs.extend(confs.tolist() if len(confs) else [])
        lo, hi = self.uncertain
        unc = int(((confs >= lo) & (confs < hi)).sum()) if len(confs) else 0
        if unc and frame_idx - self._last_q >= self.min_gap and len(self.queue) < self.budget:
            self.queue.append((key, frame_idx, unc))
            self._last_q = frame_idx
        drift = self.drift()
        if drift is not None and drift > self.psi_alert:
            self.alerts += 1
        return drift

    def drift(self):
        if len(self.hists) < 30:
            return None
        cur = np.sum(self.hists, axis=0).astype(np.float64) + 1e-3
        return psi(self.ref, cur / cur.sum())

    def stats(self):
        c = np.array(self.confs) if self.confs else np.zeros(0)
        lo, hi = self.uncertain
        return dict(frames=self.frames, luma_psi=self.drift(), drift_alert_frames=self.alerts,
                    mean_conf=float(c.mean()) if len(c) else None,
                    uncertain_frac=float(((c >= lo) & (c < hi)).mean()) if len(c) else None,
                    relabel_queue=len(self.queue))
