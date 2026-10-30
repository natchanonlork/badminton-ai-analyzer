import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd
from bai import court, players, render, ingest
from test_synthetic import CORNERS, make_video

H = court.homography(CORNERS)
FPS = 30
POS = {"A": (1.5, 3.0), "B": (4.5, 3.5), "C": (1.5, 10.0), "D": (4.5, 10.5)}
COL = {"A": (80, 10, 10), "B": (200, 10, 10), "C": (80, 60, 60), "D": (200, 60, 60)}


def row(f, raw, person, jitter=0.0):
    x, y = POS[person]; x += 0.003 * f
    px, py = court.to_pixels(H, [[x, y]])[0]
    L, A, B = COL[person]
    return (f, raw, px, py, x, y, .9, px - 20, py - 80, px + 20, py, L, A, B)


def raw_with_swap_and_loss(n=120):
    rows = []
    for f in range(n):
        # raw1/raw2 SWAP people A<->B at frame 60 (tracker error); C loses its id at 40..45 and returns as raw 9
        for raw, p in ((1, "A" if f < 60 else "B"), (2, "B" if f < 60 else "A"), (4, "D")):
            rows.append(row(f, raw, p))
        if not 40 <= f < 45:
            rows.append(row(f, 3 if f < 40 else 9, "C"))
    return pd.DataFrame(rows, columns=["frame", "raw_id", "x_px", "y_px", "x_m", "y_m", "conf",
                                       "bx1", "by1", "bx2", "by2", "cL", "cA", "cB"])


def test_swap_fixed_and_named_seeds():
    raw = raw_with_swap_and_loss()
    seeds = [(n, *court.to_pixels(H, [list(POS[k])])[0]) for k, n in (("A", "ต้น"), ("B", "บอย"), ("C", "แคท"), ("D", "ดาว"))]
    out = players.finalize_players(raw, 4, seeds, seed_frame=10, fps=FPS)
    assert set(out.player_id) == {"ต้น", "บอย", "แคท", "ดาว"}
    for k, n in (("A", "ต้น"), ("B", "บอย"), ("C", "แคท"), ("D", "ดาว")):
        g = out[out.player_id == n]
        d = np.hypot(g.x_m - (POS[k][0] + 0.003 * g.frame), g.y_m - POS[k][1])
        assert (d < 0.05).mean() > 0.95, (n, (d < 0.05).mean())      # name follows the PERSON through the swap
    assert out.groupby("frame").player_id.nunique().min() >= 3


def test_render_thai_and_shots():
    raw = raw_with_swap_and_loss(60)
    seeds = [("ต้น", *court.to_pixels(H, [list(POS["A"])])[0])]
    pdf = players.finalize_players(raw, 4, seeds, 10, FPS)
    with tempfile.TemporaryDirectory() as d:
        v, o = os.path.join(d, "a.mp4"), os.path.join(d, "o.mp4")
        make_video(v, n=60)
        shots = [{"frame": 20, "hitter": "ต้น", "type": "smash", "rally": 1, "idx": 1, "t_s": 0.66}]
        rs = [{"id": 1, "start_s": 0.5, "end_s": 1.5, "shot_count": 3}]
        render.render(v, o, 30, (1280, 720), H, pdf, pd.DataFrame(columns=["frame", "x_px", "y_px"]), shots, rs)
        assert ingest.probe(o)[0].n_frames == 60


def test_tracknetv3_csv_import(tmp_path=None):
    import tempfile
    from bai import shuttle
    rows = ["Frame,Visibility,X,Y"]
    for f in range(200):                       # real shuttle: moving; plus a static false positive at (900,260)
        rows.append(f"{f},1,{300 + f * 3},{500 - f}")
    for f in range(200, 400):
        rows.append(f"{f},1,900,260")
    rows.append("400,1,50,50")                 # isolated
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.csv"); open(p, "w").write("\n".join(rows))
        v, st = shuttle.from_tracknetv3_csv(p)
    assert st["dropped_hot"] == 200 and len(v) == 200 and v.frame.max() == 199


def test_landmark_homography_without_near_baseline():
    from bai import court as C
    Hg = C.homography(CORNERS)
    inv = np.linalg.inv(Hg)
    names = ["near_short_service_left", "near_short_service_right", "far_left_corner", "far_right_corner",
             "net_left", "centre_far_baseline"]           # none of them is a near-baseline corner
    items = []
    for n in names:
        px = C.to_pixels(Hg, [list(C.LANDMARKS[n])])[0] + np.random.default_rng(0).normal(0, .5, 2)
        items.append({"name": n, "x": float(px[0]), "y": float(px[1])})
    H = C.homography_from_landmarks(items)
    test = np.array([[3.0, 2.0], [1.0, 11.0], [5.5, 6.0]])
    px = C.to_pixels(Hg, test)
    assert np.abs(C.to_metres(H, px) - test).max() < 0.1
