"""Pallet-label reading: Tesseract on cleaned crops, GS1 check digit, per-track voting."""
import re
from collections import Counter

import pytesseract

from .gs1 import is_valid
from .preprocess import ocr_crop, ocr_crop_raw

CFG = "--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789"


def read_digits(img) -> str:
    if img is None:
        return ""
    txt = pytesseract.image_to_string(img, config=CFG)
    return re.sub(r"\D", "", txt)


def read_label(frame, box, clean=True) -> str:
    """Adaptive threshold first; fall back to Otsu if that read fails the check digit."""
    if not clean:
        return read_digits(ocr_crop_raw(frame, box))
    code = read_digits(ocr_crop(frame, box, method="adaptive"))
    if is_valid(code):
        return code
    return read_digits(ocr_crop(frame, box, method="otsu"))


class TrackReader:
    """Reads each pallet track until two valid reads agree, then stops spending OCR on it."""

    def __init__(self, every=3, confirm=2, max_attempts=40, min_share=0.6, min_width=90, edge=16):
        self.every, self.confirm, self.max_attempts = every, confirm, max_attempts
        self.min_share, self.min_width, self.edge = min_share, min_width, edge
        self.votes: dict[int, Counter] = {}
        self.attempts: dict[int, int] = {}
        self.last: dict[int, int] = {}
        self.confirmed_by: dict[str, int] = {}  # pallet ID -> track that confirmed it
        self.last_seen: dict[int, int] = {}
        self.release_after = 60  # frames: an ID is free again once its track has been gone this long
        self.ocr_calls = 0

    def _top(self, tid):
        c = self.votes.get(tid)
        if not c:
            return None, 0, 0
        code, n = c.most_common(1)[0]
        return code, n, sum(c.values())

    def done(self, tid):
        code, n, total = self._top(tid)
        return n >= self.confirm and n / total >= self.min_share

    def readable(self, frame, box):
        """Only spend OCR on labels that are fully in view and big enough to resolve."""
        if frame is None:
            return True
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = box
        e = self.edge  # detector boxes on labels cut by the frame edge stop a few px short of it
        return x0 > e and y0 > e and x1 < w - e and y1 < h - e and x1 - x0 >= self.min_width

    def touch(self, tid, frame_idx):
        self.last_seen[tid] = frame_idx

    def _owned_by_other(self, code, tid, frame_idx):
        owner = self.confirmed_by.get(code)
        return owner is not None and owner != tid and \
            frame_idx - self.last_seen.get(owner, -10**9) <= self.release_after

    def offer(self, tid, frame_idx, frame, label_box):
        if self.done(tid) or self.attempts.get(tid, 0) >= self.max_attempts:
            return
        if frame_idx - self.last.get(tid, -10**9) < self.every or not self.readable(frame, label_box):
            return
        self.last[tid] = frame_idx
        self.attempts[tid] = self.attempts.get(tid, 0) + 1
        self.ocr_calls += 1
        code = read_label(frame, label_box)
        if is_valid(code) and not self._owned_by_other(code, tid, frame_idx):  # one live pallet per ID
            self.votes.setdefault(tid, Counter())[code] += 1
            if self.done(tid) and self._top(tid)[0] == code:
                self.confirmed_by[code] = tid

    def result(self, tid):
        """Only a confirmed ID (>= `confirm` agreeing valid reads
        that also hold >= `min_share` of all valid reads); otherwise None, so the pallet is
        flagged for a manual scan rather than booked under a wrong ID."""
        code, n, total = self._top(tid)
        return code if code and self.done(tid) else None
