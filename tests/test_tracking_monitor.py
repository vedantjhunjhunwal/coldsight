import numpy as np

from coldsight.monitor import Monitor, luma_hist, psi
from coldsight.ocr import TrackReader
from coldsight.tracking import DockPipeline


def test_one_pallet_crossing_counts_once():
    pipe = DockPipeline(line_x=640, ocr=False)
    frame = np.zeros((768, 1280, 3), np.uint8)
    for t in range(60):
        x = 300 + 12 * t  # left -> right across x=640
        xyxy = np.array([[x, 400, x + 150, 600]], np.float32)
        pipe.update(t, frame, xyxy, np.array([0.9], np.float32), np.array([0]))
    ev = pipe.flush()
    assert len(ev) == 1 and ev[0].direction == "in"


def test_psi_separates_dark_from_reference():
    rng = np.random.default_rng(0)
    ref = rng.normal(120, 15, 2000).clip(0, 255)
    same = rng.normal(120, 15, 300).clip(0, 255)
    dark = rng.normal(45, 10, 300).clip(0, 255)
    assert psi(luma_hist(ref), luma_hist(same)) < 0.1
    assert psi(luma_hist(ref), luma_hist(dark)) > 1.0


def test_relabel_queue_respects_gap_and_budget():
    mon = Monitor(np.full(100, 120.0), queue_budget=3, min_gap=5)
    for t in range(40):
        mon.observe("cam", t, np.full(50, 120.0), np.array([0.4]))
    assert len(mon.queue) == 3
    frames = [q[1] for q in mon.queue]
    assert all(b - a >= 5 for a, b in zip(frames, frames[1:]))


def test_track_reader_stops_after_confirmation(monkeypatch):
    import coldsight.ocr as ocr
    monkeypatch.setattr(ocr, "read_label", lambda f, b: "40123455")
    r = TrackReader(every=1, confirm=2)
    for t in range(10):
        r.offer(1, t, None, [0, 0, 1, 1])
    assert r.result(1) == "40123455" and r.ocr_calls == 2


def test_bytetrack_keeps_id_through_low_score_occlusion():
    from coldsight.bytetrack import ByteTrack
    bt = ByteTrack()
    ids = []
    for t in range(30):
        x = 100 + 10 * t
        score = 0.3 if 10 <= t < 16 else 0.9  # partly occluded: low score
        out = bt.update(np.array([[x, 100, x + 120, 260]]), np.array([score]))
        if len(out.tracker_id):
            ids.append(int(out.tracker_id[0]))
    assert len(set(ids)) == 1 and len(ids) >= 28


def test_monitor_flags_dark_camera_not_normal_one():
    rng = np.random.default_rng(1)
    ref = rng.normal(130, 40, 20000).clip(0, 255)
    normal, dark = Monitor(ref, window=40), Monitor(ref, window=40)
    for t in range(60):
        normal.observe("a", t, rng.normal(130, 40, 2560).clip(0, 255), np.array([0.9]))
        dark.observe("b", t, rng.normal(45, 15, 2560).clip(0, 255), np.array([0.9]))
    assert normal.drift() < 0.1 and dark.drift() > 0.25


def test_label_goes_to_front_pallet_when_pallets_overlap():
    from coldsight.tracking import assign_labels
    back = [100, 100, 400, 380]    # farther from camera (higher in frame)
    front = [250, 160, 550, 450]   # closer: larger bottom y
    label = np.array([[300, 250, 420, 300]], float)  # centre lies inside both boxes
    owner = assign_labels(np.array([back, front], float), label)
    assert list(owner) == [1]


def test_confirmed_id_is_not_reused_by_another_track(monkeypatch):
    import coldsight.ocr as ocr
    monkeypatch.setattr(ocr, "read_label", lambda f, b: "40123455")
    r = TrackReader(every=1, confirm=2)
    for t in range(4):
        r.touch(1, t); r.touch(2, t)
        r.offer(1, t, None, [0, 0, 1, 1])
        r.offer(2, t, None, [0, 0, 1, 1])
    assert r.result(1) == "40123455" and r.result(2) is None
    # once track 1 has been gone long enough, the same pallet may come back as a new track
    for t in range(200, 203):
        r.touch(3, t)
        r.offer(3, t, None, [0, 0, 1, 1])
    assert r.result(3) == "40123455"


def test_split_vote_is_not_confirmed(monkeypatch):
    import coldsight.ocr as ocr
    reads = iter(["40123455", "40123455", "11111117", "22222224", "40123455"])
    monkeypatch.setattr(ocr, "read_label", lambda f, b: next(reads))
    r = TrackReader(every=1, confirm=2, min_share=0.6)
    r.offer(1, 0, None, [0, 0, 1, 1]); r.offer(1, 1, None, [0, 0, 1, 1])
    assert r.result(1) == "40123455"  # 2/2 agree


def test_skips_labels_cut_by_frame_edge():
    r = TrackReader(min_width=90)
    frame = np.zeros((768, 1280, 3), np.uint8)
    assert not r.readable(frame, [5, 100, 150, 150])      # cut by the left edge
    assert not r.readable(frame, [100, 100, 160, 150])    # too small
    assert r.readable(frame, [100, 100, 220, 150])
