"""Per-class ByteTrack, label-to-pallet association, dock-line counting and pallet events."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import CLASSES
from .bytetrack import ByteTrack
from .ocr import TrackReader

TRACKED = ("pallet", "forklift", "person")


@dataclass
class Event:
    frame: int
    track: int
    direction: str  # "in" (left->right) or "out"
    pallet_id: str | None


class DockPipeline:
    """detections -> tracks -> labels read per pallet -> line-crossing events."""

    def __init__(self, line_x: float, fps: int = 15, ocr: bool = True, ocr_every: int = 3):
        self.trackers = {c: ByteTrack(buffer=fps * 2) for c in TRACKED}
        self.line_x = line_x
        self.side: dict[int, int] = {}
        self.counted: set[int] = set()
        self.reader = TrackReader(every=ocr_every) if ocr else None
        self.events: list[Event] = []
        self.pending: list[Event] = []  # crossed before the label was confirmed

    def update(self, frame_idx, frame, xyxy, conf, cls):
        out = {}
        for c in TRACKED:
            m = cls == CLASSES.index(c)
            out[c] = self.trackers[c].update(xyxy[m], conf[m])
        pallets = out["pallet"]
        labels = xyxy[(cls == CLASSES.index("label")) & (conf >= 0.3)]
        owner = assign_labels(pallets.xyxy, labels)
        for k, (box, tid) in enumerate(zip(pallets.xyxy, pallets.tracker_id)):
            tid = int(tid)
            if self.reader is not None:
                self.reader.touch(tid, frame_idx)
            lab = owner.get(k)
            if self.reader is not None and lab is not None:
                self.reader.offer(tid, frame_idx, frame, lab)
            cx = (box[0] + box[2]) / 2
            s = 1 if cx >= self.line_x else -1
            prev = self.side.get(tid)
            if prev is not None and s != prev and tid not in self.counted:
                self.counted.add(tid)
                self.pending.append(Event(frame_idx, tid, "in" if s > 0 else "out", None))
            self.side[tid] = s
        # attach IDs once confirmed (or give up 3 s after the crossing)
        still = []
        for e in self.pending:
            pid = self.reader.result(e.track) if self.reader else None
            if (self.reader and self.reader.done(e.track)) or frame_idx - e.frame > 45 or self.reader is None:
                e.pallet_id = pid
                self.events.append(e)
            else:
                still.append(e)
        self.pending = still
        return out

    def flush(self):
        for e in self.pending:
            e.pallet_id = self.reader.result(e.track) if self.reader else None
            self.events.append(e)
        self.pending = []
        return self.events


def assign_labels(pallet_boxes, labels):
    """Give each label to ONE pallet: of the pallets containing its centre, the front-most one
    (largest bottom y = closest to the camera), since an overlapping pallet behind it would also
    contain the label. Returns {pallet index: label box}; a pallet keeps its largest label."""
    owner = {}
    if len(labels) == 0 or len(pallet_boxes) == 0:
        return owner
    P = np.asarray(pallet_boxes)
    for lb in labels:
        cx, cy = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
        inside = np.where((cx > P[:, 0]) & (cx < P[:, 2]) & (cy > P[:, 1]) & (cy < P[:, 3]))[0]
        if len(inside) == 0:
            continue
        k = int(inside[np.argmax(P[inside, 3])])
        area = (lb[2] - lb[0]) * (lb[3] - lb[1])
        if k not in owner or area > (owner[k][2] - owner[k][0]) * (owner[k][3] - owner[k][1]):
            owner[k] = lb
    return owner
