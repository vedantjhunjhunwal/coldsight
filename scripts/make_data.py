"""Render sequences, write CVAT video XML per sequence, convert through CVAT -> YOLO with QA.

Splits (disjoint seeds):
  train   normal lighting               - detector training (v1)
  val     normal lighting               - model selection
  test    normal lighting, full clips   - detection, tracking, OCR, counting
  shift   low light + frosted lens      - held-out deployment shift
  stream  low light, unlabelled clips   - simulated live feed for the feedback loop
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coldsight.annotations import qa_frame, read_cvat_video, write_cvat_video, yolo_lines  # noqa: E402
from coldsight.synth import H, W, Sequence  # noqa: E402

SPLITS = {  # name: (seeds, mode, frames per clip, keep every k-th frame for the image dataset)
    "train": (range(0, 40), "normal", 120, 6),
    "val": (range(100, 108), "normal", 120, 8),
    "test": (range(200, 208), "normal", 150, 5),
    "shift": (range(300, 308), "lowlight", 150, 5),
    "stream": (range(400, 412), "lowlight", 120, 1),
}


def main(out: Path, only=None):
    summary = {}
    for split, (seeds, mode, n, k) in SPLITS.items():
        if only and split not in only:
            continue
        clip_dir = out / "clips" / split
        img_dir, lab_dir = out / "yolo" / "images" / split, out / "yolo" / "labels" / split
        for d in (clip_dir, img_dir, lab_dir):
            d.mkdir(parents=True, exist_ok=True)
        n_img, issues, n_boxes = 0, [], 0
        for seed in seeds:
            seq = Sequence(seed, n, mode)
            need_video = split in ("test", "shift", "stream")
            frames, gts = {}, []
            for t in range(n):
                render = need_video or t % k == 0
                img, gt = seq.frame(t, render=render)
                if render:
                    frames[t] = img
                gts.append(gt)
            name = f"{split}_{seed:03d}"
            if need_video:
                vw = cv2.VideoWriter(str(clip_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
                for t in range(n):
                    vw.write(frames[t])
                vw.release()
            write_cvat_video(clip_dir / f"{name}.xml", name, n, W, H, gts)
            json.dump(seq.pallet_ids, open(clip_dir / f"{name}.ids.json", "w"))
            if split == "stream":
                continue
            # the image dataset is built FROM the CVAT export, as with a real annotation workflow
            _, _, _, ann = read_cvat_video(clip_dir / f"{name}.xml")
            for t in range(0, n, k):
                boxes = [(a["cls"], a["box"]) for a in ann.get(t, [])]
                issues += [f"{name}:{t}: {m}" for m in qa_frame(boxes, W, H)]
                cv2.imwrite(str(img_dir / f"{name}_{t:04d}.jpg"), frames[t], [cv2.IMWRITE_JPEG_QUALITY, 92])
                (lab_dir / f"{name}_{t:04d}.txt").write_text("\n".join(yolo_lines(boxes, W, H)))
                n_img += 1
                n_boxes += len(boxes)
        summary[split] = dict(clips=len(seeds), images=n_img, boxes=n_boxes, qa_issues=len(issues),
                              qa_examples=issues[:5])
        print(split, summary[split], flush=True)
    (out / "yolo" / "data.yaml").write_text(
        f"path: {(out / 'yolo').resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names: {0: pallet, 1: label, 2: forklift, 3: person}\n")
    (out / "yolo" / "shift.yaml").write_text(
        f"path: {(out / 'yolo').resolve()}\ntrain: images/train\nval: images/shift\n"
        "names: {0: pallet, 1: label, 2: forklift, 3: person}\n")
    json.dump(summary, open(out / f"dataset_summary_{'_'.join(summary)}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    main(Path(a.out), a.only)
