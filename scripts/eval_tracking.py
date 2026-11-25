"""End-to-end on full clips: ByteTrack MOT metrics for pallets, dock-line counts, and pallet-ID events."""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coldsight.annotations import read_cvat_video  # noqa: E402
from coldsight.detector import OnnxDetector  # noqa: E402
from coldsight.evaluate import box_iou  # noqa: E402
from coldsight.tracking import DockPipeline  # noqa: E402

W, H = 1280, 768
LINE_X = W / 2


def gt_crossings(ann, n):
    side, out = {}, []
    for t in range(n):
        for a in ann.get(t, []):
            if a["cls"] != "pallet":
                continue
            cx = (a["box"][0] + a["box"][2]) / 2
            s = 1 if cx >= LINE_X else -1
            if a["track"] in side and side[a["track"]] != s and a["track"] not in [o[1] for o in out]:
                out.append((t, a["track"], "in" if s > 0 else "out", a["attrs"].get("pallet_id")))
            side[a["track"]] = s
    return out


def run_clip(det, mp4, xml):
    _, _, n, ann = read_cvat_video(xml)
    pipe = DockPipeline(LINE_X, fps=15)
    acc = mm.MOTAccumulator(auto_id=True)
    cap = cv2.VideoCapture(str(mp4))
    hyp_boxes = defaultdict(dict)
    t, t_det, t_all = 0, 0.0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        xyxy, conf, cls = det(frame)
        t1 = time.perf_counter()
        tracks = pipe.update(t, frame, xyxy, conf, cls)
        t_all += time.perf_counter() - t0
        t_det += t1 - t0
        p = tracks["pallet"]
        g = [a for a in ann.get(t, []) if a["cls"] == "pallet"]
        gb = np.array([a["box"] for a in g]).reshape(-1, 4)
        hb = p.xyxy.reshape(-1, 4)
        for b, tid in zip(hb, p.tracker_id):
            hyp_boxes[int(tid)][t] = b
        dist = 1 - box_iou(gb, hb)
        dist[dist > 0.5] = np.nan
        acc.update([a["track"] for a in g], [int(i) for i in p.tracker_id], dist)
        t += 1
    events = pipe.flush()
    gtc = gt_crossings(ann, n)
    # match each event to the GT pallet under the hypothesis box at that frame
    id_ok = id_wrong = id_none = 0
    for e in events:
        hb = hyp_boxes[e.track].get(e.frame)
        g = [a for a in ann.get(e.frame, []) if a["cls"] == "pallet"]
        truth = None
        if hb is not None and g:
            ious = box_iou(hb[None], np.array([a["box"] for a in g]))[0]
            if ious.max() > 0.3:
                truth = g[int(ious.argmax())]["attrs"].get("pallet_id")
        if e.pallet_id is None:
            id_none += 1
        elif e.pallet_id == truth:
            id_ok += 1
        else:
            id_wrong += 1
    return acc, dict(frames=t, gt_crossings=len(gtc), pred_crossings=len(events),
                     gt_in=sum(c[2] == "in" for c in gtc), pred_in=sum(e.direction == "in" for e in events),
                     id_correct=id_ok, id_wrong=id_wrong, id_unread=id_none, ocr_calls=pipe.reader.ocr_calls,
                     ms_detector=1000 * t_det / max(t, 1), ms_pipeline=1000 * t_all / max(t, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--splits", nargs="+", default=["test", "shift"])
    ap.add_argument("--out", default="results/tracking.json")
    a = ap.parse_args()
    det = OnnxDetector(a.model, conf=0.3, iou=0.5)
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for split in a.splits:
        accs, names, clips = [], [], []
        for mp4 in sorted(Path("data/clips", split).glob("*.mp4")):
            acc, info = run_clip(det, mp4, mp4.with_suffix(".xml"))
            accs.append(acc)
            names.append(mp4.stem)
            clips.append(dict(clip=mp4.stem, **info))
            print(mp4.stem, info, flush=True)
        mh = mm.metrics.create()
        summ = mh.compute_many(accs, metrics=["mota", "idf1", "num_switches", "mostly_tracked", "num_unique_objects",
                                              "precision", "recall"], names=names, generate_overall=True)
        ov = summ.loc["OVERALL"].to_dict()
        tot = {k: sum(c[k] for c in clips) for k in ("gt_crossings", "pred_crossings", "id_correct", "id_wrong",
                                                       "id_unread", "frames", "ocr_calls")}
        out[f"{Path(a.model).stem}|{split}"] = dict(
            mot={k: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in ov.items()},
            counting=dict(clips=len(clips), clips_exact=sum(c["gt_crossings"] == c["pred_crossings"] for c in clips),
                          gt_total=tot["gt_crossings"], pred_total=tot["pred_crossings"],
                          abs_error=sum(abs(c["gt_crossings"] - c["pred_crossings"]) for c in clips)),
            pallet_ids=dict(correct=tot["id_correct"], wrong=tot["id_wrong"], unread=tot["id_unread"]),
            ocr_calls_per_frame=round(tot["ocr_calls"] / max(tot["frames"], 1), 3),
            ms_per_frame=dict(detector=round(float(np.mean([c["ms_detector"] for c in clips])), 1),
                              pipeline=round(float(np.mean([c["ms_pipeline"] for c in clips])), 1)),
            clips=clips)
        print(json.dumps({k: v for k, v in out[f"{Path(a.model).stem}|{split}"].items() if k != "clips"}, indent=1))
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
