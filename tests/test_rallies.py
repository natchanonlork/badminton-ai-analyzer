import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd
from bai import court, rallies as rl

CORNERS = np.array([[300, 650], [980, 650], [830, 200], [450, 200]], np.float32)
H = court.homography(CORNERS)
FPS = 30


def synth(n_rallies=2, hits_per=5, flight=0.6):
    """Shuttle flies between near (y~3) and far (y~10.5) players in straight lines."""
    near, far = (3.0, 3.0), (3.0, 10.5)
    sh, pl, truth = [], [], []
    f = 10
    for r in range(n_rallies):
        for h in range(hits_per):
            a, b = (near, far) if h % 2 == 0 else (far, near)
            n = int(flight * FPS)
            for k in range(n):
                t = k / n
                xm, ym = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
                px, py = court.to_pixels(H, [[xm, ym]])[0]
                sh.append((f + k, px, py - 30 * np.sin(np.pi * t), .9))     # a little arc
            truth.append(f)
            f += n
        f += 3 * FPS                                                       # dead time between rallies
    for fr in range(f + 30):
        for pid, (x, y) in (("P1", near), ("P2", far)):
            px, py = court.to_pixels(H, [[x, y]])[0]
            pl.append((fr, pid, px, py, x, y, px - 25, py - 100, px + 25, py))
    return (pd.DataFrame(sh, columns=["frame", "x_px", "y_px", "conf"]),
            pd.DataFrame(pl, columns=["frame", "player_id", "x_px", "y_px", "x_m", "y_m", "bx1", "by1", "bx2", "by2"]), truth)


def test_rallies_and_hits():
    sh, pl, truth = synth()
    hits = rl.detect_hits(sh, pl, FPS)
    assert len(hits) >= len(truth) - 4          # end-of-rally shots have no turn after them
    rs = rl.build_rallies(hits, H, FPS, pl)
    assert len(rs) == 2
    assert all(r["shot_count"] >= 3 for r in rs)
    assert rs[0]["shots"][0]["type"] == "serve"
    hitters = [s["hitter"] for s in rs[0]["shots"]]
    assert all(a != b for a, b in zip(hitters, hitters[1:]))             # alternate players
    st = rl.player_stats(pl, FPS, rs)
    assert {s["id"] for s in st} == {"P1", "P2"}
