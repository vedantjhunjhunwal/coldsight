import numpy as np

from coldsight.ocr import read_label
from coldsight.preprocess import enhance_low_light, letterbox, mean_luma
from coldsight.synth import Sequence


def test_letterbox_geometry():
    img = np.zeros((768, 1280, 3), np.uint8)
    out, r, left, top = letterbox(img, 416)
    assert out.shape == (416, 416, 3) and left == 0 and abs(r - 0.325) < 1e-6 and top == (416 - 250) // 2


def test_low_light_fix_only_touches_dark_frames():
    bright = np.full((64, 64, 3), 150, np.uint8)
    assert enhance_low_light(bright) is bright
    dark = np.full((64, 64, 3), 25, np.uint8)
    assert mean_luma(enhance_low_light(dark)) > 60


def test_reads_clean_label_exactly():
    seq = Sequence(8, 150)
    hits = total = 0
    for t in (40, 60, 80):
        img, gts = seq.frame(t, degrade=False)
        for g in gts:
            if g["cls"] == "label" and g["visible"] > 0.95 and g["box"][2] - g["box"][0] > 120:
                total += 1
                hits += read_label(img, g["box"]) == g["attrs"]["pallet_id"]
    assert total > 0 and hits / total >= 0.8
