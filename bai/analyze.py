"""CLI: python -m bai.analyze clip.mp4 --setup setup.json --out out/
setup.json comes from pick_corners.html (court corners + named players + seed time).
Stages cache to disk: court.json, players_raw.parquet, shuttle.parquet; naming/rallies/render always rerun
(cheap), so changing player names does not need re-tracking. --force recomputes everything."""
import argparse, json, os, sys
import cv2
import numpy as np
import pandas as pd
from . import ingest, court, players as pl, shuttle as sh, render as rd


def save_table(df, path):
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        alt = path.rsplit(".", 1)[0] + ".csv"
        df.to_csv(alt, index=False)
        print(f"[warn] pyarrow ไม่มี -> เขียน {alt} แทน", file=sys.stderr)
        return alt


def load_table(path):
    for p in (path, path.rsplit(".", 1)[0] + ".csv"):
        if os.path.exists(p):
            return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    return None


def progress(stage, frac):
    print(f"\r[{stage}] {frac * 100:5.1f}%", end="", flush=True)


def main(argv=None):
    for _s in (sys.stdout, sys.stderr):          # Windows consoles default to cp1252/cp874: never crash on Thai text
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--setup", help="setup.json จาก pick_corners.html")
    ap.add_argument("--mode", choices=["singles", "doubles"])
    ap.add_argument("--out", default="out")
    ap.add_argument("--corners", type=float, nargs=8, metavar="X",
                    help="near-left near-right far-right far-left (x y x y x y x y)")
    ap.add_argument("--shuttle", choices=["none", "motion", "tracknet"], default="none")
    ap.add_argument("--shuttle-weights")
    ap.add_argument("--shuttle-csv", help="*_ball.csv จาก TrackNetV3 predict.py (Frame,Visibility,X,Y)")
    ap.add_argument("--yolo", default="yolo11m-pose.pt")
    ap.add_argument("--device")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    setup = json.load(open(a.setup, encoding="utf-8")) if a.setup else {}
    video = a.video or setup.get("video")
    if not video:
        sys.exit("ต้องระบุไฟล์วิดีโอ")
    corners_arg = a.corners or setup.get("corners")
    seeds_in = setup.get("players", [])
    mode = a.mode or setup.get("mode") or ("singles" if len(seeds_in) == 2 else "doubles")
    n_players = 2 if mode == "singles" else 4
    os.makedirs(a.out, exist_ok=True)
    P = lambda n: os.path.join(a.out, n)

    info, warns = ingest.probe(video)
    for w in warns:
        print(f"[เตือน] {w}", file=sys.stderr)

    # --- court
    if a.force or not os.path.exists(P("court.json")):
        if setup.get("landmarks"):
            H0 = court.homography_from_landmarks(setup["landmarks"])
            corners, conf = court.corners_from_H(H0), 1.0
        elif corners_arg:
            corners, conf = np.array(corners_arg, np.float32).reshape(4, 2), 1.0
        else:
            cap = cv2.VideoCapture(video)
            step = max(info.n_frames // 20, 1)
            samples = []
            for k in range(20):
                cap.set(cv2.CAP_PROP_POS_FRAMES, k * step)
                ok, f = cap.read()
                if ok:
                    samples.append(f)
            cap.release()
            corners, conf = court.auto_detect(samples)
            if corners is None or conf < 0.6:
                sys.exit(f"หาเส้นคอร์ทไม่เจอ (conf={conf:.2f}) -> ใช้ --setup หรือ --corners")
        cdat = {"corners_px": np.asarray(corners).tolist(), "confidence": float(conf)}
        if setup.get("landmarks"):
            cdat["H"] = H0.tolist()
        json.dump(cdat, open(P("court.json"), "w", encoding="utf-8"))
    cj = json.load(open(P("court.json"), encoding="utf-8"))
    H = np.array(cj["H"], np.float64) if "H" in cj else court.homography(cj["corners_px"])
    print(f"court conf={cj['confidence']:.2f}  mode={mode}")

    # --- players: raw tracking cached; naming/stitching always rerun
    raw = None if a.force else load_table(P("players_raw.parquet"))
    if raw is None or "cL" not in raw:
        raw = pl.track_players(ingest.frames(video), H, a.yolo, a.device, progress, info.n_frames)
        print()
        save_table(raw, P("players_raw.parquet"))
    seeds = [(p["name"], p["x"], p["y"]) for p in seeds_in] or None
    seed_frame = int(round(setup.get("seed_t", 0) * info.fps))
    pdf = pl.finalize_players(raw, n_players, seeds, seed_frame, info.fps)
    save_table(pdf, P("players.parquet"))
    pdf.to_csv(P("players.csv"), index=False, encoding="utf-8-sig")        # plain copies so other tools can read them
    cnt = pdf.groupby("frame").player_id.nunique()
    print(f"ผู้เล่นครบ {n_players} คน: {(cnt == n_players).mean() * 100:.0f}% ของเฟรมที่มีคน | "
          + ", ".join(f"{k}:{v}" for k, v in pdf.player_id.value_counts().items()))

    # --- shuttle
    sdf = None if a.force else load_table(P("shuttle.parquet"))
    if a.shuttle_csv:
        v_, st_ = sh.from_tracknetv3_csv(a.shuttle_csv)
        print(f"TrackNetV3 csv: เห็นลูก {st_['visible_raw']} เฟรม, ตัดจุดนิ่งผิดปกติ {st_['dropped_hot']}, เหลือ {st_['kept']} "
              f"({st_['kept'] / max(st_['total_frames'], 1) * 100:.0f}% ของเฟรม)")
        sdf = sh.to_frame_df({int(r.frame): (r.x_px, r.y_px, r.conf) for r in v_.itertuples()}, info.n_frames)
        m = court.to_metres(H, sdf[["x_px", "y_px"]].values)
        sdf["x_m"], sdf["y_m"] = m[:, 0], m[:, 1]
        save_table(sdf, P("shuttle.parquet"))
    if sdf is None:
        be = sh.get_backend(a.shuttle, a.shuttle_weights)
        dets = {}
        for i, fr in ingest.frames(video):
            r = be.process(fr)
            if r:
                dets[i] = r
            if i % 50 == 0:
                progress("shuttle", i / max(info.n_frames, 1))
        print()
        sdf = sh.to_frame_df(dets, info.n_frames)
        if len(sdf):
            m = court.to_metres(H, sdf[["x_px", "y_px"]].values)
            sdf["x_m"], sdf["y_m"] = m[:, 0], m[:, 1]
        save_table(sdf, P("shuttle.parquet"))
    if a.shuttle == "none" and len(sdf) == 0:
        print("[หมายเหตุ] ยังไม่ได้ตามลูก (--shuttle none) -> ไม่มีแรลลี่/ป้ายช็อต ต้องเทรน TrackNet ก่อน")

    if len(sdf):
        sdf.to_csv(P("shuttle.csv"), index=False, encoding="utf-8-sig")

    # --- phase 2 before render so labels can be drawn
    rs, shots, spd = [], [], None
    if len(sdf) > 20:
        from . import rallies as rl
        dbg = []
        hits = rl.detect_hits(sdf, pdf, info.fps, debug=dbg)
        pd.DataFrame(dbg).to_csv(P("hits_debug.csv"), index=False, encoding="utf-8-sig")   # every turning point + why kept/dropped
        rs = rl.build_rallies(hits, H, info.fps, pdf)   # ความเร็วลูกไม่ถูกคำนวณ: วัดจากกล้องตัวเดียวไม่น่าเชื่อถือ
        shots = [{**s, "rally": r["id"]} for r in rs for s in r["shots"]]
        ana = {"video": {"file": os.path.basename(video), "fps": info.fps, "width": info.width,
                         "height": info.height, "duration_s": round(info.duration_s, 1)},
               "court": {"homography": H.tolist(), "corners_px": cj["corners_px"], "confidence": cj["confidence"]},
               "players": [{"id": p_, "name": p_} for p_ in sorted(pdf.player_id.unique())],
               "rallies": rs, "player_stats": rl.player_stats(pdf, info.fps, rs)}
        json.dump(ana, open(P("analysis.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        pd.DataFrame([{k: v for k, v in s.items() if k not in ("hit_xy_m", "land_xy_m")} for s in shots]
                     ).to_csv(P("shots.csv"), index=False, encoding="utf-8-sig")
        if rs and len(sdf):
            tt = sdf.frame / info.fps
            sdf["in_rally"] = False
            for r_ in rs:
                sdf.loc[(tt >= r_["start_s"] - rl.PRE_S) & (tt <= r_["end_s"] + rl.POST_S), "in_rally"] = True
        print(f"rallies={len(rs)} shots={len(shots)} (ประเภทช็อตเป็นกฎเดา ยังไม่ผ่านการจูน)")

    rd.render(video, P("overlay.mp4"), info.fps, (info.width, info.height), H, pdf, sdf, shots, rs,
              progress=progress, total=info.n_frames, speed=spd)
    print()
    pt = pdf.rename(columns={"player_id": "object"})[["frame", "object", "x_px", "y_px", "x_m", "y_m", "conf"]]
    st = (sdf.assign(object="shuttle")[["frame", "object", "x_px", "y_px"] + [c for c in ("x_m", "y_m", "in_rally") if c in sdf] + ["conf"]]
          if len(sdf) else sdf)
    tr = pd.concat([pt, st], ignore_index=True)
    tr.insert(1, "t_s", tr.frame / info.fps)
    save_table(tr.sort_values(["frame", "object"]), P("tracks.parquet"))
    print(f"เสร็จ -> {a.out}/overlay.mp4")


if __name__ == "__main__":
    main()
