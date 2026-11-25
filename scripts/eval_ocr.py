"""OCR on ground-truth label boxes (isolates the reader from the detector).

Reports single-read exact accuracy with and without crop cleanup, how often the GS1
check digit rejects a wrong read, and per-pallet accuracy after voting across frames.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coldsight.annotations import read_cvat_video  # noqa: E402
from coldsight.gs1 import is_valid  # noqa: E402
from coldsight.ocr import read_label  # noqa: E402


def run(split, every=5, clips=None):
    res = {"clean": Counter(), "raw": Counter()}
    votes = defaultdict(Counter)
    truth = {}
    for xml in sorted(Path("data/clips", split).glob("*.xml"))[:clips]:
        _, _, _, ann = read_cvat_video(xml)
        stem = xml.stem
        for t in sorted(ann):
            if t % every:
                continue
            img = cv2.imread(f"data/yolo/images/{split}/{stem}_{t:04d}.jpg")
            if img is None:
                continue
            for a in ann[t]:
                if a["cls"] != "label":
                    continue
                gt = a["attrs"]["pallet_id"]
                key = (stem, a["track"])
                truth[key] = gt
                for mode in ("clean", "raw"):
                    code = read_label(img, a["box"], clean=(mode == "clean"))
                    r = res[mode]
                    r["n"] += 1
                    r["exact"] += code == gt
                    if is_valid(code):
                        r["accepted"] += 1
                        r["accepted_wrong"] += code != gt
                        if mode == "clean":
                            votes[key][code] += 1
    out = {}
    for mode, r in res.items():
        n = max(r["n"], 1)
        out[mode] = dict(reads=r["n"], exact=round(r["exact"] / n, 3), accepted=round(r["accepted"] / n, 3),
                         wrong_among_accepted=round(r["accepted_wrong"] / max(r["accepted"], 1), 4))
    def final(k):  # same rule as TrackReader: >= 2 agreeing valid reads
        if not votes[k]:
            return None
        code, n = votes[k].most_common(1)[0]
        return code if n >= 2 else None
    voted_ok = sum(1 for k, g in truth.items() if final(k) == g)
    voted_wrong = sum(1 for k, g in truth.items() if final(k) not in (None, g))
    out["per_pallet_voted"] = dict(pallets=len(truth), correct=voted_ok, wrong=voted_wrong,
                                   unread=len(truth) - voted_ok - voted_wrong)
    return out


if __name__ == "__main__":
    report = {s: run(s) for s in ("test", "shift")}
    Path("results").mkdir(exist_ok=True)
    json.dump(report, open("results/ocr_gt_boxes.json", "w"), indent=1)
    print(json.dumps(report, indent=1))
