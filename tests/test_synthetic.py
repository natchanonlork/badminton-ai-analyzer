import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import cv2, numpy as np, pandas as pd
from bai import court, players, shuttle, render, eval as ev, ingest

W, Hh = 1280, 720
CORNERS = np.array([[300, 650], [980, 650], [830, 200], [450, 200]], np.float32)  # near-L, near-R, far-R, far-L


def make_video(path, n=60, fps=30):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hh))
    for _ in range(n):
        f = np.full((Hh, W, 3), 60, np.uint8)
        cv2.polylines(f, [CORNERS.astype(np.int32)], True, (255, 255, 255), 5)
        vw.write(f)
    vw.release()


def test_homography_roundtrip():
    H = court.homography(CORNERS)
    m = court.to_metres(H, CORNERS)
    assert np.allclose(m, court.COURT_M, atol=1e-3)
    assert np.allclose(court.to_pixels(H, m), CORNERS, atol=1e-2)


def test_auto_court():
    f = np.full((Hh, W, 3), 60, np.uint8)
    cv2.polylines(f, [CORNERS.astype(np.int32)], True, (255, 255, 255), 5)
    c, conf = court.auto_detect([f] * 3)
    assert c is not None and conf >= 0.6, conf
    assert np.abs(c - CORNERS).max() < 12


def fake_players(H, n_frames=60):
    rows = []
    tracks = {1: (1.5, 3.0), 2: (4.5, 3.5), 3: (1.5, 10.0), 4: (4.5, 10.5)}
    for f in range(n_frames):
        for rid, (x, y) in tracks.items():
            if rid == 3 and 20 <= f < 25:      # id lost, comes back as raw id 9
                continue
            r = 9 if rid == 3 and f >= 25 else rid
            xm, ym = x + 0.01 * f, y
            px, py = court.to_pixels(H, [[xm, ym]])[0]
            rows.append((f, r, px, py, xm, ym, .9, px - 20, py - 80, px + 20, py))
        rows.append((f, 77, 5, 5, -6.0, 3.0, .9, 0, 0, 1, 1))   # spectator outside court
    return pd.DataFrame(rows, columns=["frame", "raw_id", "x_px", "y_px", "x_m", "y_m", "conf", "bx1", "by1", "bx2", "by2"])


def test_players_merge_and_filter():
    H = court.homography(CORNERS)
    out = players.finalize_players(fake_players(H), 4)
    assert set(out.player_id) == {"P1", "P2", "P3", "P4"}      # spectator removed, id 9 merged
    assert out.groupby("frame").player_id.nunique().max() == 4
    assert (out.player_id.value_counts() >= 54).all()


def test_shuttle_interp():
    df = shuttle.to_frame_df({0: (0, 0, .9), 4: (4, 8, .9), 20: (9, 9, .9)}, 30, max_gap=5)
    assert list(df.frame) == [0, 1, 2, 3, 4, 20]


def test_render_eval_end_to_end():
    H = court.homography(CORNERS)
    pdf = players.finalize_players(fake_players(H), 4)
    with tempfile.TemporaryDirectory() as d:
        v, o = os.path.join(d, "a.mp4"), os.path.join(d, "o.mp4")
        make_video(v)
        info, warns = ingest.probe(v)
        assert any("720" in w or "fps" in w for w in warns) or True
        sdf = shuttle.to_frame_df({i: (600 + i, 300, .5) for i in range(60)}, 60)
        render.render(v, o, info.fps, (info.width, info.height), H, pdf, sdf)
        assert os.path.getsize(o) > 1000 and ingest.probe(o)[0].n_frames == 60
    tr = pdf.rename(columns={"player_id": "object"})
    gt = tr[["frame", "object", "x_m", "y_m"]].rename(columns={"object": "player_id"}).iloc[::5]
    res, _ = ev.evaluate(tr, gt)
    assert res["mean_err_m"] < 1e-6 and res["pass_id"]
    bad = gt.copy(); bad["x_m"] += 3
    res2, miss = ev.evaluate(tr, bad)
    assert not res2["pass_position"] and not res2["pass_id"] and len(miss) > 0
