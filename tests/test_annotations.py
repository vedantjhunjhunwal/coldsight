import json

from coldsight.annotations import qa_frame, read_cvat_video, read_labelstudio, to_labelstudio_task, write_cvat_video, yolo_lines
from coldsight.synth import H, W, Sequence


def test_cvat_roundtrip(tmp_path):
    seq = Sequence(7, 30)
    gts = [seq.frame(t, render=False)[1] for t in range(30)]
    write_cvat_video(tmp_path / "a.xml", "a", 30, W, H, gts)
    w, h, n, ann = read_cvat_video(tmp_path / "a.xml")
    assert (w, h, n) == (W, H, 30)
    for t in range(30):
        got = sorted((a["track"], a["cls"], tuple(a["box"])) for a in ann.get(t, []))
        exp = sorted((g["track"], g["cls"], tuple(float(v) for v in g["box"])) for g in gts[t])
        assert got == exp
    ids = {a["attrs"]["pallet_id"] for fr in ann.values() for a in fr if a["cls"] == "label"}
    assert ids <= set(seq.pallet_ids.values())


def test_labelstudio_roundtrip():
    boxes = [("pallet", [100, 200, 300, 500], None), ("label", [150, 300, 250, 340], None)]
    task = to_labelstudio_task("img.jpg", W, H, boxes, as_prediction=False)
    back = read_labelstudio([json.loads(json.dumps(task))])[0][1]
    for (c, b, _), (c2, b2) in zip(boxes, back):
        assert c == c2 and all(abs(x - y) < 1e-6 for x, y in zip(b, b2))


def test_predictions_are_not_read_as_annotations():
    task = to_labelstudio_task("img.jpg", W, H, [("pallet", [1, 2, 30, 40], 0.4)], as_prediction=True)
    assert read_labelstudio([task]) == []


def test_qa_flags_bad_boxes():
    assert qa_frame([("pallet", [0, 0, 100, 100]), ("label", [10, 10, 50, 30])], W, H) == []
    issues = qa_frame([("pallet", [0, 0, 100, 100]), ("pallet", [0, 0, 100, 100]), ("label", [500, 500, 560, 520]),
                       ("forklift", [-20, 0, 50, 50]), ("person", [5, 5, 8, 8])], W, H)
    assert any("duplicate" in i for i in issues)
    assert any("not on any pallet" in i for i in issues)
    assert any("out of frame" in i for i in issues)
    assert any("too small" in i for i in issues)


def test_yolo_lines_normalised():
    (line,) = yolo_lines([("label", [0, 0, W, H])], W, H)
    assert line == "1 0.500000 0.500000 1.000000 1.000000"
