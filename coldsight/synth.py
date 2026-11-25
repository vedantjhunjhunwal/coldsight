"""Procedural cold-store dock video with exact ground truth.

Every sequence is fully determined by its seed. Objects are drawn back to front,
and an id-map records which object owns each pixel, so visibility and occlusion
in the ground truth are exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .gs1 import make_id

W, H = 1280, 768  # HD-class dock camera; the detector sees a 416 px letterbox
S = W / 640  # size scale relative to the 640 px design
CLS = {"pallet": 0, "label": 1, "forklift": 2, "person": 3}


@dataclass
class Sprite:
    img: np.ndarray
    mask: np.ndarray
    children: list = field(default_factory=list)  # (cls, x, y, w, h, attrs) relative boxes


@dataclass
class Actor:
    oid: int
    cls: str
    sprite: Sprite
    x: float
    y: float  # bottom of sprite
    vx: float
    t0: int
    phase: float = 0.0
    attrs: dict = field(default_factory=dict)
    rider: "Actor | None" = None  # pallet carried on forks


# ---------------------------------------------------------------- sprites
def _noise(rng, h, w, amp):
    n = rng.normal(0, amp, (h // 4 + 1, w // 4 + 1)).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]


def pallet_sprite(rng, w: int, code: str) -> Sprite:
    h = int(w * rng.uniform(0.85, 1.25))
    img = np.zeros((h, w, 3), np.uint8)
    base_h = max(8, int(h * 0.14))
    load_h = h - base_h
    # wooden base with gaps
    wood = np.array([60, 105, 150]) + rng.integers(-15, 15, 3)
    img[load_h:, :] = wood
    n_gap = 3
    for i in range(n_gap):
        gx = int((i + 1) * w / (n_gap + 1))
        img[load_h + base_h // 3:, gx - w // 20: gx + w // 20] = (25, 30, 35)
    # cartons
    rows, cols = rng.integers(2, 4), rng.integers(2, 4)
    carton = np.array([95, 150, 190]) + rng.integers(-25, 25, 3)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r * load_h / rows), int((r + 1) * load_h / rows)
            x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
            col = np.clip(carton + rng.integers(-12, 12, 3), 0, 255)
            img[y0:y1, x0:x1] = col
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (60, 90, 120), 1)
            cv2.line(img, ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1), (170, 190, 200), max(2, w // 80))
    # stretch wrap
    wrap = img.astype(np.float32) * 0.75 + 255 * 0.25
    img[:load_h] = wrap[:load_h].astype(np.uint8)
    for k in range(3):
        x = int(rng.uniform(0, w))
        cv2.line(img, (x, 0), (x + w // 3, load_h), (235, 235, 235), 1)
    # label: white card, ID digits, barcode stripes
    lw = int(w * rng.uniform(0.62, 0.78))
    lh = int(lw * 0.42)
    lx = int(rng.uniform(0.05, 0.95) * (w - lw))
    ly = int(rng.uniform(0.35, 0.6) * (load_h - lh))
    card = np.full((lh, lw, 3), 245, np.uint8)
    th_px = max(1, int(round(lw / 70)))
    scale = 2.0
    while cv2.getTextSize(code, cv2.FONT_HERSHEY_SIMPLEX, scale, th_px)[0][0] > lw * 0.9:
        scale -= 0.05
    (tw, th), _ = cv2.getTextSize(code, cv2.FONT_HERSHEY_SIMPLEX, scale, th_px)
    cv2.putText(card, code, ((lw - tw) // 2, int(lh * 0.08) + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (10, 10, 10), th_px, cv2.LINE_AA)
    bx = int(lw * 0.08)
    by0, by1 = int(lh * 0.62), int(lh * 0.92)
    while bx < lw * 0.92:
        bw = int(rng.integers(1, 4))
        card[by0:by1, bx:bx + bw] = 15
        bx += bw + int(rng.integers(1, 4))
    img[ly:ly + lh, lx:lx + lw] = card
    mask = np.full((h, w), 255, np.uint8)
    return Sprite(img, mask, [("label", lx, ly, lw, lh, {"pallet_id": code})])


def forklift_sprite(rng, w: int, facing: int) -> Sprite:
    h = int(w * 0.95)
    img = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    body = [(0, 140, 240), (20, 200, 240), (40, 60, 200)][rng.integers(0, 3)]
    bx0, bx1 = int(w * 0.30), w - 1
    by0 = int(h * 0.45)
    cv2.rectangle(img, (bx0, by0), (bx1, int(h * 0.85)), body, -1)
    cv2.rectangle(mask, (bx0, by0), (bx1, int(h * 0.85)), 255, -1)
    # overhead guard / cab
    cv2.rectangle(img, (bx0 + w // 10, int(h * 0.08)), (bx1 - w // 8, by0), (40, 40, 40), max(3, w // 60))
    cv2.rectangle(mask, (bx0 + w // 10, int(h * 0.08)), (bx1 - w // 8, by0), 255, max(3, w // 60))
    cv2.line(img, (bx0 + w // 10, int(h * 0.08)), (bx1 - w // 8, int(h * 0.08)), (40, 40, 40), max(5, w // 40))
    # mast and forks
    mx = int(w * 0.24)
    cv2.rectangle(img, (mx, 0), (mx + w // 14, int(h * 0.92)), (70, 70, 75), -1)
    cv2.rectangle(mask, (mx, 0), (mx + w // 14, int(h * 0.92)), 255, -1)
    cv2.rectangle(img, (0, int(h * 0.88)), (mx + w // 14, int(h * 0.92)), (80, 80, 85), -1)
    cv2.rectangle(mask, (0, int(h * 0.88)), (mx + w // 14, int(h * 0.92)), 255, -1)
    for cx in (int(w * 0.42), int(w * 0.86)):
        cv2.circle(img, (cx, int(h * 0.86)), int(h * 0.12), (20, 20, 20), -1)
        cv2.circle(mask, (cx, int(h * 0.86)), int(h * 0.12), 255, -1)
    img = img * (mask[..., None] > 0)
    if facing > 0:  # base drawing has forks on the left; flip so forks lead the motion
        img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
    return Sprite(img.astype(np.uint8), mask)


def person_sprite(rng, h: int) -> Sprite:
    w = int(h * 0.42)
    img = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    vest = [(0, 165, 255), (60, 255, 200)][rng.integers(0, 2)]
    hr = max(4, h // 11)
    cv2.circle(img, (w // 2, hr + 1), hr, (140, 170, 210), -1)
    cv2.circle(mask, (w // 2, hr + 1), hr, 255, -1)
    cv2.ellipse(img, (w // 2, hr), (hr, hr // 2 + 1), 0, 180, 360, (240, 240, 240), -1)
    cv2.rectangle(img, (w // 5, 2 * hr + 2), (4 * w // 5, int(h * 0.58)), vest, -1)
    cv2.rectangle(mask, (w // 5, 2 * hr + 2), (4 * w // 5, int(h * 0.58)), 255, -1)
    cv2.line(img, (w // 5, int(h * 0.42)), (4 * w // 5, int(h * 0.42)), (210, 210, 210), max(2, h // 60))
    cv2.rectangle(img, (w // 4, int(h * 0.58)), (3 * w // 4, h - 1), (60, 50, 45), -1)
    cv2.rectangle(mask, (w // 4, int(h * 0.58)), (3 * w // 4, h - 1), 255, -1)
    img[int(h * 0.75):, w // 2 - 1: w // 2 + 1] = 0
    return Sprite(img, mask)


# ---------------------------------------------------------------- scene
def background(rng) -> np.ndarray:
    bg = np.zeros((H, W, 3), np.float32)
    wall_y = int(H * rng.uniform(0.38, 0.48))
    bg[:wall_y] = np.array([150, 140, 125]) + rng.integers(-15, 15, 3)
    floor = np.linspace(105, 145, H - wall_y)[:, None, None] * np.ones((1, W, 3))
    bg[wall_y:] = floor + rng.integers(-10, 10, 3)
    bg += _noise(rng, H, W, 6)
    # dock doors with rubber seals
    for k in range(rng.integers(1, 3)):
        dx = int(rng.uniform(0.05, 0.7) * W)
        dw = int(W * rng.uniform(0.18, 0.26))
        cv2.rectangle(bg, (dx, int(wall_y * 0.15)), (dx + dw, wall_y), (70, 70, 72), -1)
        cv2.rectangle(bg, (dx, int(wall_y * 0.15)), (dx + dw, wall_y), (25, 25, 25), int(6 * S))
        for yy in range(int(wall_y * 0.15) + 10, wall_y, int(12 * S)):
            cv2.line(bg, (dx + 6, yy), (dx + dw - 6, yy), (90, 90, 92), 1)
    # floor safety lines
    for ly in (wall_y + int((H - wall_y) * 0.25), H - int(12 * S)):
        cv2.line(bg, (0, ly), (W, ly), (0, 200, 230), int(3 * S))
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    # wall signs: white cards with text, hard negatives for the label class
    signs = ["DOCK %d" % rng.integers(1, 9), "COLD ZONE -18C", "FORKLIFT ROUTE", "BAY %02d" % rng.integers(1, 40)]
    for k in range(rng.integers(1, 3)):
        txt = signs[int(rng.integers(0, len(signs)))]
        sw = int(rng.uniform(70, 120) * S); sh = int(sw * 0.4)
        sx = int(rng.uniform(0, W - sw)); sy = int(rng.uniform(4, max(6, wall_y * 0.6)))
        cv2.rectangle(bg, (sx, sy), (sx + sw, sy + sh), (240, 240, 240), -1)
        sc = 0.9
        sc = 0.9 * S
        while cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, int(2 * S))[0][0] > sw * 0.9:
            sc -= 0.05
        cv2.putText(bg, txt, (sx + 4, sy + sh // 2 + int(6 * S)), cv2.FONT_HERSHEY_SIMPLEX, sc, (20, 20, 160), int(2 * S))
    return bg


@dataclass
class Degrade:
    gamma: float = 1.0
    haze: float = 0.0
    blur: float = 0.0
    noise: float = 2.0
    jpeg: int = 90
    tint: tuple = (1.0, 1.0, 1.0)
    jitter: float = 0.0


def sample_degrade(rng, mode: str) -> Degrade:
    if mode == "normal":
        return Degrade(gamma=rng.uniform(0.85, 1.25), haze=rng.uniform(0, 0.12), blur=rng.uniform(0, 0.7),
                       noise=rng.uniform(1, 4), jpeg=int(rng.integers(70, 95)),
                       tint=tuple(rng.uniform(0.92, 1.08, 3)), jitter=rng.uniform(0, 1.0))
    if mode == "lowlight":  # night shift, failing lights, frosted lens
        return Degrade(gamma=rng.uniform(2.2, 3.0), haze=rng.uniform(0.10, 0.22), blur=rng.uniform(0.8, 1.6),
                       noise=rng.uniform(6, 11), jpeg=int(rng.integers(40, 65)),
                       tint=(rng.uniform(1.05, 1.25), rng.uniform(0.95, 1.05), rng.uniform(0.75, 0.9)),
                       jitter=rng.uniform(0.5, 2.0))
    raise ValueError(mode)


def apply_degrade(frame: np.ndarray, d: Degrade, rng) -> np.ndarray:
    f = frame.astype(np.float32) / 255.0
    f = f * np.array(d.tint, np.float32)
    f = np.clip(f, 0, 1) ** d.gamma
    if d.haze > 0:
        f = f * (1 - d.haze) + d.haze * (0.78 if d.gamma < 1.6 else 0.35)
    out = np.clip(f * 255, 0, 255).astype(np.uint8)
    if d.blur > 0.05:
        out = cv2.GaussianBlur(out, (0, 0), d.blur * S)
    if d.jitter > 0:
        M = np.float32([[1, 0, rng.normal(0, d.jitter * S)], [0, 1, rng.normal(0, d.jitter * S)]])
        out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    out = np.clip(out.astype(np.float32) + rng.normal(0, d.noise, out.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, d.jpeg])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


class Sequence:
    """One camera clip: actors enter, cross the dock and leave."""

    def __init__(self, seed: int, n_frames: int = 150, mode: str = "normal"):
        self.rng = np.random.default_rng(seed)
        self.seed, self.n, self.mode = seed, n_frames, mode
        self.bg = background(self.rng)
        self.deg = sample_degrade(self.rng, mode)
        self.actors: list[Actor] = []
        self.pallet_ids: dict[int, str] = {}
        self._spawn()

    def _new_code(self) -> str:
        return make_id(f"{int(self.rng.integers(0, 10**7)):07d}")

    def _spawn(self):
        rng, oid = self.rng, 0
        lanes = [H - int(10 * S), H - int(55 * S), H - int(95 * S)]
        n_events = int(rng.integers(4, 8))
        for e in range(n_events):
            t0 = int(rng.uniform(-self.n * 0.25, self.n * 0.7))
            lane = lanes[int(rng.integers(0, len(lanes)))]
            depth = 1.0 - 0.18 * lanes.index(lane)
            d = 1 if rng.random() < 0.5 else -1
            speed = rng.uniform(3.0, 6.5) * depth * S
            kind = rng.choice(["forklift", "conveyor", "person"], p=[0.45, 0.35, 0.2])
            if kind == "person":
                sp = person_sprite(rng, int(rng.uniform(125, 165) * depth * S))
                x0 = -sp.img.shape[1] if d > 0 else W
                self.actors.append(Actor(oid, "person", sp, x0, lane, d * speed * 0.5, t0)); oid += 1
                continue
            pw = int(rng.uniform(120, 165) * depth * S)
            code = self._new_code()
            ps = pallet_sprite(rng, pw, code)
            if kind == "forklift":
                fs = forklift_sprite(rng, int(pw * 1.25), d)
                x0 = -fs.img.shape[1] - pw * 0.6 if d > 0 else W + pw * 0.6
                fk = Actor(oid, "forklift", fs, x0, lane, d * speed, t0); oid += 1
                pal = Actor(oid, "pallet", ps, 0, 0, 0, t0, attrs={"pallet_id": code}); oid += 1
                fk.rider = pal
                self.pallet_ids[pal.oid] = code
                self.actors += [fk, pal]
            else:
                x0 = -pw if d > 0 else W
                pal = Actor(oid, "pallet", ps, x0, lane, d * speed, t0, attrs={"pallet_id": code}); oid += 1
                self.pallet_ids[pal.oid] = code
                self.actors.append(pal)

    def _place(self, a: Actor, t: int):
        """Return top-left (x, y) of actor a at frame t, or None if not started."""
        if t < a.t0:
            return None
        for fk in self.actors:
            if fk.rider is a:
                fx, fy = self._place(fk, t)
                fw = fk.sprite.img.shape[1]
                ph, pw = a.sprite.img.shape[:2]
                fork_y = fy + int(fk.sprite.img.shape[0] * 0.90)
                px = fx + fw - int(pw * 0.55) if fk.vx > 0 else fx - int(pw * 0.45)
                return px, fork_y - ph
        dt = t - a.t0
        x = a.x + a.vx * dt
        bob = int(2 * np.sin(dt * 0.6)) if a.cls == "person" else 0
        return int(x), int(a.y - a.sprite.img.shape[0] + bob)

    def frame(self, t: int, degrade: bool = True, render: bool = True):
        canvas = self.bg.copy()
        idmap = np.full((H, W), -1, np.int32)
        placed = []
        for a in self.actors:
            p = self._place(a, t)
            if p is None:
                continue
            placed.append((a, p))
        # paint back to front: smaller bottom-y (farther) first; rider after its forklift
        order = sorted(placed, key=lambda ap: (ap[1][1] + ap[0].sprite.img.shape[0], ap[0].cls == "pallet"))
        full_area = {}
        for a, (x, y) in order:
            img, m = a.sprite.img, a.sprite.mask
            h, w = m.shape
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
            full_area[a.oid] = int((m > 0).sum())
            if x1 <= x0 or y1 <= y0:
                continue
            sub_m = m[y0 - y:y1 - y, x0 - x:x1 - x] > 0
            canvas[y0:y1, x0:x1][sub_m] = img[y0 - y:y1 - y, x0 - x:x1 - x][sub_m]
            idmap[y0:y1, x0:x1][sub_m] = a.oid
        gts = []
        for a, (x, y) in order:
            h, w = a.sprite.mask.shape
            vis = int((idmap == a.oid).sum())
            frac = vis / max(full_area[a.oid], 1)
            box = _clip(x, y, w, h)
            if box and frac >= 0.25:
                gts.append(dict(track=a.oid, cls=a.cls, box=box, visible=round(frac, 3), attrs=dict(a.attrs)))
            for (ccls, cx, cy, cw, ch, cattrs) in a.sprite.children:
                lx, ly = x + cx, y + cy
                lbox = _clip(lx, ly, cw, ch)
                if not lbox:
                    continue
                region = idmap[max(ly, 0):min(ly + ch, H), max(lx, 0):min(lx + cw, W)]
                lfrac = float((region == a.oid).sum()) / (cw * ch)
                if lfrac >= 0.6:
                    gts.append(dict(track=10000 + a.oid, cls=ccls, box=lbox, visible=round(lfrac, 3),
                                    attrs=dict(cattrs)))
        if not render:
            return None, gts
        img = apply_degrade(canvas, self.deg, self.rng) if degrade else canvas
        return img, gts


def _clip(x, y, w, h, min_side=8):
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None
    return [int(x0), int(y0), int(x1), int(y1)]
