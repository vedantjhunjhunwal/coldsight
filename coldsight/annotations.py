"""Annotation I/O: CVAT for video 1.1 XML, Label Studio JSON, YOLO txt, plus QA checks.

Bulk ground truth lives in CVAT video XML (tracks with keyframes, `outside`,
`occluded` and a `pallet_id` attribute). The relabelling queue built from live
deployment feedback goes out as Label Studio tasks with model pre-annotations,
and corrected tasks come back through `read_labelstudio`.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from . import CLASSES

# ------------------------------------------------------------------ CVAT
def write_cvat_video(path, task_name, n_frames, width, height, frames_gt):
    """frames_gt: list (per frame) of dicts {track, cls, box, visible, attrs}."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(meta, "name").text = task_name
    ET.SubElement(meta, "size").text = str(n_frames)
    ET.SubElement(meta, "mode").text = "interpolation"
    labels = ET.SubElement(meta, "labels")
    for c in CLASSES:
        lab = ET.SubElement(labels, "label")
        ET.SubElement(lab, "name").text = c
        attrs = ET.SubElement(lab, "attributes")
        if c in ("pallet", "label"):
            a = ET.SubElement(attrs, "attribute")
            ET.SubElement(a, "name").text = "pallet_id"
            ET.SubElement(a, "input_type").text = "text"
    osz = ET.SubElement(meta, "original_size")
    ET.SubElement(osz, "width").text = str(width)
    ET.SubElement(osz, "height").text = str(height)

    tracks = defaultdict(list)
    for f, gts in enumerate(frames_gt):
        for g in gts:
            tracks[(g["track"], g["cls"])].append((f, g))
    for (tid, cls), items in sorted(tracks.items()):
        tr = ET.SubElement(root, "track", id=str(tid), label=cls, source="manual")
        prev = None
        for f, g in items:
            x0, y0, x1, y1 = g["box"]
            b = ET.SubElement(tr, "box", frame=str(f), outside="0", occluded=str(int(g["visible"] < 0.6)),
                              keyframe="1", xtl=f"{x0:.2f}", ytl=f"{y0:.2f}", xbr=f"{x1:.2f}", ybr=f"{y1:.2f}",
                              z_order="0")
            for k, v in g.get("attrs", {}).items():
                ET.SubElement(b, "attribute", name=k).text = str(v)
            if prev is not None and f != prev + 1:  # gap: close then reopen
                pass
            prev = f
        last = items[-1][0]
        if last + 1 < n_frames:
            x0, y0, x1, y1 = items[-1][1]["box"]
            ET.SubElement(tr, "box", frame=str(last + 1), outside="1", occluded="0", keyframe="1",
                          xtl=f"{x0:.2f}", ytl=f"{y0:.2f}", xbr=f"{x1:.2f}", ybr=f"{y1:.2f}", z_order="0")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read_cvat_video(path):
    """Return (width, height, n_frames, {frame: [ {track, cls, box, occluded, attrs} ]})."""
    root = ET.parse(path).getroot()
    task = root.find("meta/task")
    w = int(task.find("original_size/width").text)
    h = int(task.find("original_size/height").text)
    n = int(task.find("size").text)
    frames = defaultdict(list)
    for tr in root.iter("track"):
        tid, cls = int(tr.get("id")), tr.get("label")
        for b in tr.iter("box"):
            if b.get("outside") == "1":
                continue
            box = [float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            attrs = {a.get("name"): a.text for a in b.iter("attribute")}
            frames[int(b.get("frame"))].append(dict(track=tid, cls=cls, box=box,
                                                    occluded=b.get("occluded") == "1", attrs=attrs))
    return w, h, n, frames


# ------------------------------------------------------------ Label Studio
def to_labelstudio_task(image_url, width, height, boxes, as_prediction=True, model_version="v1"):
    """boxes: list of (cls, [x0,y0,x1,y1], score|None). Percent coords as Label Studio expects."""
    res = []
    for i, (cls, (x0, y0, x1, y1), score) in enumerate(boxes):
        r = {"id": f"r{i}", "type": "rectanglelabels", "from_name": "label", "to_name": "image",
             "original_width": width, "original_height": height, "image_rotation": 0,
             "value": {"x": 100 * x0 / width, "y": 100 * y0 / height, "width": 100 * (x1 - x0) / width,
                       "height": 100 * (y1 - y0) / height, "rotation": 0, "rectanglelabels": [cls]}}
        if score is not None:
            r["score"] = float(score)
        res.append(r)
    key = "predictions" if as_prediction else "annotations"
    entry = {"result": res}
    if as_prediction:
        entry["model_version"] = model_version
    return {"data": {"image": image_url}, key: [entry]}


def read_labelstudio(tasks):
    """Return list of (image_url, [(cls, [x0,y0,x1,y1])]) from completed annotations."""
    out = []
    for t in tasks:
        anns = t.get("annotations") or []
        if not anns:
            continue
        boxes = []
        for r in anns[-1]["result"]:
            if r.get("type") != "rectanglelabels":
                continue
            v, W, H = r["value"], r["original_width"], r["original_height"]
            x0, y0 = v["x"] * W / 100, v["y"] * H / 100
            boxes.append((v["rectanglelabels"][0], [x0, y0, x0 + v["width"] * W / 100, y0 + v["height"] * H / 100]))
        out.append((t["data"]["image"], boxes))
    return out


# ------------------------------------------------------------------ YOLO
def yolo_lines(boxes, width, height):
    lines = []
    for cls, (x0, y0, x1, y1) in boxes:
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        lines.append(f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")
    return lines


# ------------------------------------------------------------------- QA
def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def qa_frame(boxes, width, height, min_side=6):
    """Return list of issue strings for one frame's (cls, box) list."""
    issues = []
    for i, (cls, b) in enumerate(boxes):
        if cls not in CLASSES:
            issues.append(f"unknown class {cls}")
        if b[0] < -1 or b[1] < -1 or b[2] > width + 1 or b[3] > height + 1:
            issues.append(f"{cls} box out of frame")
        if b[2] - b[0] < min_side or b[3] - b[1] < min_side:
            issues.append(f"{cls} box too small")
        for cls2, b2 in boxes[i + 1:]:
            if cls2 == cls and _iou(b, b2) > 0.95:
                issues.append(f"duplicate {cls}")
    pallets = [b for c, b in boxes if c == "pallet"]
    for cls, b in boxes:
        if cls == "label":
            inside = any(b[0] >= p[0] - 2 and b[1] >= p[1] - 2 and b[2] <= p[2] + 2 and b[3] <= p[3] + 2
                         for p in pallets)
            if pallets and not inside:
                issues.append("label not on any pallet")
    return issues
