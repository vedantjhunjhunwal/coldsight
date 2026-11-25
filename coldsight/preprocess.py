"""OpenCV preprocessing: letterbox for the detector, low-light enhancement, OCR crop cleanup."""
import cv2
import numpy as np


def letterbox(img, size, color=114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((size, size, 3), color, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, left, top


def mean_luma(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def enhance_low_light(img, dark_threshold=80.0, clip=2.5):
    """Gamma lift + CLAHE on the L channel, applied only when the frame is dark."""
    m = mean_luma(img)
    if m >= dark_threshold:
        return img
    gamma = np.clip(np.log(118 / 255) / np.log(max(m, 1) / 255), 0.35, 1.0)
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    img = cv2.LUT(img, lut)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def ocr_crop(frame, box, inset=0.04, method="adaptive"):
    """Crop the text band of a pallet label and binarise it for Tesseract.

    The crop is inset slightly so the card edge and background do not survive
    thresholding, and any blob still touching the crop border is removed.
    """
    x0, y0, x1, y1 = [float(v) for v in box]
    bw_, bh_ = x1 - x0, y1 - y0
    x0, x1 = x0 + inset * bw_, x1 - inset * bw_
    y0, y1 = y0 + inset * bh_, y0 + 0.62 * bh_  # digits sit in the upper part, barcode below
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), w), min(int(y1), h)
    if x1 - x0 < 16 or y1 - y0 < 8:
        return None
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    target_h = 64  # Tesseract likes ~30-40 px glyphs
    f = max(1.0, target_h / g.shape[0])
    g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 8)).apply(g)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    if method == "adaptive":  # copes with uneven light across the card
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    else:
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # text -> white
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    H_, W_ = bw.shape
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        touches = x == 0 or y == 0 or x + ww >= W_ or y + hh >= H_
        if touches and (hh < 0.5 * H_ or ww > 0.5 * W_) or area < 6:
            bw[lab == i] = 0
    bw = 255 - bw
    return cv2.copyMakeBorder(bw, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def ocr_crop_raw(frame, box):
    """Baseline without cleanup: plain grayscale crop (used only for the ablation)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    crop = frame[max(y0, 0):y1, max(x0, 0):x1]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
