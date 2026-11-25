import numpy as np

from coldsight.synth import H, W, Sequence


def test_deterministic():
    a, b = Sequence(11, 40), Sequence(11, 40)
    for t in (0, 20, 39):
        assert a.frame(t, render=False)[1] == b.frame(t, render=False)[1]


def test_labels_sit_inside_their_pallet_and_carry_valid_ids():
    from coldsight.gs1 import is_valid
    seq = Sequence(3, 120)
    seen = 0
    for t in range(0, 120, 10):
        _, gts = seq.frame(t, render=False)
        pallets = {g["track"]: g["box"] for g in gts if g["cls"] == "pallet"}
        for g in gts:
            if g["cls"] == "label":
                assert is_valid(g["attrs"]["pallet_id"])
                p = pallets.get(g["track"] - 10000)
                if p:
                    b = g["box"]
                    assert b[0] >= p[0] and b[2] <= p[2] and b[1] >= p[1] and b[3] <= p[3]
                    seen += 1
    assert seen > 0


def test_lowlight_is_darker():
    img_n, _ = Sequence(5, 10, "normal").frame(5)
    img_l, _ = Sequence(5, 10, "lowlight").frame(5)
    assert img_n.shape == img_l.shape == (H, W, 3)
    assert img_l.mean() < 0.75 * img_n.mean()
