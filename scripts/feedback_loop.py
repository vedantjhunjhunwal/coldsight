"""Live-feedback loop: watch a shifted stream with v1, queue uncertain frames, relabel, fine-tune v2.

1. Run the v1 model over unlabelled low-light stream clips with the Monitor attached.
   Luminance PSI against the training set flags the shift; frames with uncertain
   detections go to a relabel queue (budgeted, spaced out in time).
2. Export the queue as Label Studio tasks with v1 pre-annotations.
3. Annotators correct them. Here the correction is simulated with the clips' ground truth
   and written back as a Label Studio export, so the import path is the real one.
4. Import -> YOLO, then fine-tune v1 into v2 on train + feedback frames.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coldsight import CLASSES  # noqa: E402
from coldsight.annotations import (qa_frame, read_cvat_video, read_labelstudio, to_labelstudio_task,  # noqa: E402
                                   yolo_lines)
from coldsight.detector import OnnxDetector  # noqa: E402
from coldsight.monitor import Monitor, luma_hist, luma_sample, psi  # noqa: E402

W, H = 1280, 768
FB = Path("data/feedback")


def reference_lumas():
    files = sorted(Path("data/yolo/images/train").glob("*.jpg"))[::2]
    return np.concatenate([luma_sample(cv2.imread(str(p))) for p in files])


def calibrate_threshold(ref, margin=1.25):
    """Alert threshold = margin x the worst PSI seen on held-out normal-light validation cameras."""
    worst = 0.0
    by_clip = {}
    for p in sorted(Path("data/yolo/images/val").glob("*.jpg")):
        by_clip.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)
    for files in by_clip.values():
        px = np.concatenate([luma_sample(cv2.imread(str(f))) for f in files])
        worst = max(worst, psi(luma_hist(ref), luma_hist(px)))
    return round(margin * worst, 3), round(worst, 3)


def monitor_clips(det, split, ref, budget, min_gap, thr):
    """Return monitor stats, PSI trace and queue for one split's clips (each clip a camera)."""
    queue, trace, alerts, frames, first = [], [], 0, 0, []
    per_clip = max(1, budget // len(list(Path("data/clips", split).glob("*.mp4"))))
    for mp4 in sorted(Path("data/clips", split).glob("*.mp4")):
        mon = Monitor(ref, window=90, psi_alert=thr, queue_budget=per_clip, min_gap=min_gap)
        first_alert = None
        cap = cv2.VideoCapture(str(mp4))
        t = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            xyxy, conf, cls = det(frame)
            d = mon.observe(mp4.stem, t, luma_sample(frame), conf)
            if d is not None:
                trace.append(d)
                if first_alert is None and d > thr:
                    first_alert = t
            if mon.queue and mon.queue[-1][1] == t and mon.queue[-1][0] == mp4.stem:
                queue.append((mp4.stem, t, frame, [(CLASSES[c], b.tolist(), float(s))
                                                   for b, s, c in zip(xyxy, conf, cls)]))
            t += 1
        alerts += mon.alerts
        frames += mon.frames
        first.append(first_alert)
    flagged = [f for f in first if f is not None]
    return dict(cameras=len(first), cameras_flagged=len(flagged),
                first_alert_frame_max=max(flagged) if flagged else None,
                frames=frames, alert_frames=alerts, alert_rate=round(alerts / max(frames, 1), 3),
                psi_median=round(float(np.median(trace)), 3), psi_max=round(float(np.max(trace)), 3)), queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/coldsight_v1_fp32.onnx")
    ap.add_argument("--budget", type=int, default=180)
    a = ap.parse_args()
    det = OnnxDetector(a.model, conf=0.25, iou=0.5)
    ref = reference_lumas()
    report = {}
    # false-alarm check on normal-light clips, then the shifted live stream
    np.save("models/reference_luma_hist.npy", np.histogram(ref, bins=np.linspace(0, 255, 17))[0])
    thr, worst = calibrate_threshold(ref)
    report["psi_threshold"] = dict(value=thr, worst_normal_val=worst, rule="1.25 x worst normal-light val camera")
    json.dump(dict(psi_alert=thr), open("models/monitor_config.json", "w"))
    report["monitor_normal_test"], _ = monitor_clips(det, "test", ref, 0, 8, thr)
    report["monitor_lowlight_stream"], queue = monitor_clips(det, "stream", ref, a.budget, 8, thr)
    print(json.dumps(report, indent=1), flush=True)

    img_dir, lab_dir = FB / "images", FB / "labels"
    for d in (img_dir, lab_dir):
        d.mkdir(parents=True, exist_ok=True)
    tasks = []
    for stem, t, frame, boxes in queue:
        name = f"{stem}_{t:04d}.jpg"
        cv2.imwrite(str(img_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tasks.append(to_labelstudio_task(f"/data/local-files/?d=feedback/images/{name}", W, H,
                                         [(c, b, s) for c, b, s in boxes if s >= 0.25], model_version="v1"))
    json.dump(tasks, open(FB / "labelstudio_tasks.json", "w"), indent=1)

    # simulated annotators: correct each task to ground truth, as a Label Studio export
    gt_cache, edits = {}, dict(added=0, removed=0, kept=0)
    for task in tasks:
        name = task["data"]["image"].rsplit("/", 1)[-1]
        stem, t = name[:-9], int(name[-8:-4])
        if stem not in gt_cache:
            gt_cache[stem] = read_cvat_video(Path("data/clips/stream") / f"{stem}.xml")[3]
        gt = [(g["cls"], g["box"]) for g in gt_cache[stem].get(t, [])]
        n_pred = len(task["predictions"][0]["result"])
        edits["kept"] += min(n_pred, len(gt))
        edits["added"] += max(0, len(gt) - n_pred)
        edits["removed"] += max(0, n_pred - len(gt))
        task["annotations"] = [to_labelstudio_task("", W, H, [(c, b, None) for c, b in gt],
                                                   as_prediction=False)["annotations"][0]]
    json.dump(tasks, open(FB / "labelstudio_export.json", "w"), indent=1)

    issues = 0
    for url, boxes in read_labelstudio(json.load(open(FB / "labelstudio_export.json"))):
        name = url.rsplit("/", 1)[-1]
        issues += len(qa_frame(boxes, W, H))
        (lab_dir / name.replace(".jpg", ".txt")).write_text("\n".join(yolo_lines(boxes, W, H)))
    report["relabel"] = dict(frames=len(tasks), qa_issues=issues, box_count_edits=edits)
    (Path("data/yolo") / "feedback.yaml").write_text(
        f"path: {Path('data').resolve()}\ntrain: [yolo/images/train, feedback/images]\nval: yolo/images/val\n"
        "names: {0: pallet, 1: label, 2: forklift, 3: person}\n")
    Path("results").mkdir(exist_ok=True)
    json.dump(report, open("results/feedback_loop.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
