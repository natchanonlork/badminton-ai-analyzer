"""Evaluate phase-1 criteria against hand-labelled ground truth.

Ground truth CSV (gt_players.csv): frame,player_id,x_m,y_m   -- label every Nth frame, ids P1..Pn
Optional rallies CSV (gt_rallies.csv): rally_id,start_frame,end_frame  -- ID-swap is checked per rally.

python -m bai.eval out/tracks.parquet gt_players.csv [gt_rallies.csv] [--save-misses misses.csv]
Targets: mean position error <= 0.5 m, no ID swap in >= 95% of rallies.
"""
import argparse
import numpy as np
import pandas as pd


def evaluate(tracks: pd.DataFrame, gt: pd.DataFrame, rallies: pd.DataFrame | None = None, tol_m=0.5):
    pr = tracks[tracks.object.str.startswith("P")].rename(columns={"object": "player_id"})
    m = gt.merge(pr[["frame", "player_id", "x_m", "y_m"]], on=["frame", "player_id"], how="left",
                 suffixes=("", "_pred"))
    m["err"] = np.hypot(m.x_m - m.x_m_pred, m.y_m - m.y_m_pred)
    found = m.err.notna()
    res = {"gt_points": len(m), "detected_frac": float(found.mean()),
           "mean_err_m": float(m.err.mean()), "median_err_m": float(m.err.median())}
    # ID swap: within a rally, a player_id is "stable" if all its matched errors < 2 m (a swap
    # puts the pred at another player's position, typically > 2 m away)
    if rallies is None:
        rallies = pd.DataFrame({"rally_id": [0], "start_frame": [gt.frame.min()], "end_frame": [gt.frame.max()]})
    ok = 0
    for r in rallies.itertuples():
        g = m[(m.frame >= r.start_frame) & (m.frame <= r.end_frame)]
        ok += int(not (g.err > 2.0).any())
    res["rallies"] = len(rallies)
    res["rallies_no_swap_frac"] = ok / max(len(rallies), 1)
    res["pass_position"] = res["mean_err_m"] <= tol_m
    res["pass_id"] = res["rallies_no_swap_frac"] >= 0.95
    misses = m[(m.err > tol_m * 2) | m.err.isna()].sort_values("err", ascending=False)
    return res, misses


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks"); ap.add_argument("gt"); ap.add_argument("rallies", nargs="?")
    ap.add_argument("--save-misses")
    a = ap.parse_args(argv)
    t = pd.read_parquet(a.tracks) if a.tracks.endswith(".parquet") else pd.read_csv(a.tracks)
    res, misses = evaluate(t, pd.read_csv(a.gt), pd.read_csv(a.rallies) if a.rallies else None)
    for k, v in res.items():
        print(f"{k:24s} {v}")
    print(f"\nเฟรมที่พลาด {len(misses)} จุด (บันทึกด้วย --save-misses เพื่อดูภาพ):")
    print(misses.head(10).to_string(index=False))
    if a.save_misses:
        misses.to_csv(a.save_misses, index=False)
    return 0 if res["pass_position"] and res["pass_id"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
