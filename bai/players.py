"""Detect + track people (Ultralytics YOLO pose + ByteTrack), keep people standing in the court,
then stitch raw tracklets into named players using seeds (user clicks), position continuity and shirt colour."""
import numpy as np
import pandas as pd
import cv2
from .court import to_metres, COURT_W, COURT_L

MARGIN_M = 1.0
MIN_TRACKLET = 5


def track_players(frames_iter, H, model_name="yolo11m-pose.pt", device=None, progress=None, total=None) -> pd.DataFrame:
    """Raw detections with ByteTrack ids. Foot point = mean of ankles (pose) else bbox bottom-centre."""
    from ultralytics import YOLO  # lazy: AGPL dependency
    model = YOLO(model_name)
    rows = []
    for i, frame in frames_iter:
        res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False, device=device)[0]
        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            kp = kc = None
            if getattr(res, "keypoints", None) is not None and res.keypoints.xy is not None and len(res.keypoints.xy):
                kp = res.keypoints.xy.cpu().numpy()
                kc = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
            feet = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], 1)
            for k in range(len(ids)):
                if kp is not None and kc is not None:
                    a = [j for j in (15, 16) if kc[k, j] > 0.4]
                    if a:
                        feet[k] = kp[k, a].mean(0)
            m = to_metres(H, feet)
            for k in range(len(ids)):
                x1, y1, x2, y2 = xyxy[k]
                w, h = x2 - x1, y2 - y1
                crop = frame[int(max(y1 + .2 * h, 0)):int(max(y1 + .55 * h, 1)), int(max(x1 + .2 * w, 0)):int(max(x1 + .8 * w, 1))]
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0) if crop.size else (0, 0, 0)
                rows.append((i, int(ids[k]), feet[k, 0], feet[k, 1], m[k, 0], m[k, 1], float(conf[k]),
                             x1, y1, x2, y2, *lab))
        if progress and total and i % 50 == 0:
            progress("players", i / total)
    return pd.DataFrame(rows, columns=["frame", "raw_id", "x_px", "y_px", "x_m", "y_m", "conf",
                                       "bx1", "by1", "bx2", "by2", "cL", "cA", "cB"])


def _split(df, fps):
    """Cut a raw track where the person visibly changes (position jump or shirt-colour jump)."""
    out, nxt = [], 0
    for rid, g in df.sort_values("frame").groupby("raw_id"):
        g = g.reset_index(drop=True)
        cut = [0]
        for i in range(1, len(g)):
            dt = max(g.frame[i] - g.frame[i - 1], 1) / fps
            d = np.hypot(g.x_m[i] - g.x_m[i - 1], g.y_m[i] - g.y_m[i - 1])
            dc = np.linalg.norm(g.loc[i, ["cL", "cA", "cB"]].astype(float).values -
                                g.loc[max(i - 5, cut[-1]):i - 1, ["cL", "cA", "cB"]].astype(float).mean().values) \
                if "cL" in g else 0
            if d > 0.6 + 7.0 * dt or dc > 40:
                cut.append(i)
        cut.append(len(g))
        for a, b in zip(cut[:-1], cut[1:]):
            seg = g.iloc[a:b].copy(); seg["tid"] = nxt; nxt += 1
            out.append(seg)
    return pd.concat(out, ignore_index=True) if out else df.assign(tid=[])


def finalize_players(df, n_players, seeds=None, seed_frame=0, fps=30.0):
    """seeds: list of (name, x_px, y_px) clicked at seed_frame. Returns df with `player_id` (names)."""
    empty = df.assign(player_id=pd.Series(dtype=str))
    if df.empty:
        return empty
    inside = (df.x_m > -MARGIN_M) & (df.x_m < COURT_W + MARGIN_M) & (df.y_m > -MARGIN_M) & (df.y_m < COURT_L + MARGIN_M)
    df = df[inside].copy()
    if df.empty:
        return empty
    has_col = "cL" in df
    if not has_col:
        df[["cL", "cA", "cB"]] = 0.0
    df = _split(df, fps)
    info = {}
    for tid, g in df.groupby("tid"):
        if len(g) < MIN_TRACKLET:
            continue
        info[tid] = dict(a=int(g.frame.min()), b=int(g.frame.max()), n=len(g),
                         col=g[["cL", "cA", "cB"]].mean().values, g=g.sort_values("frame"))
    slots = [dict(tids=[], name=None) for _ in range(n_players)]
    used = set()
    if seeds:
        for s, (name, sx, sy) in enumerate(seeds[:n_players]):
            slots[s]["name"] = name
            best, bd = None, 1e9
            for tid, t in info.items():
                if tid in used or not (t["a"] - 10 <= seed_frame <= t["b"] + 10):
                    continue
                g = t["g"]
                r = g.iloc[(g.frame - seed_frame).abs().argmin()]
                d = np.hypot(r.x_px - sx, r.y_px - sy)
                if d < bd:
                    best, bd = tid, d
            if best is not None and bd < 250:
                slots[s]["tids"].append(best); used.add(best)
    free = [s for s in slots if not s["tids"]]
    for tid in sorted(info, key=lambda t: -info[t]["n"]):      # unseeded slots: longest tracklets that overlap
        if not free:
            break
        if tid in used:
            continue
        free[0]["tids"].append(tid); used.add(tid); free.pop(0)

    def slot_cost(slot, t):
        best = 1e9
        for o in slot["tids"]:
            oi = info[o]
            if not (t["b"] < oi["a"] or t["a"] > oi["b"]):
                return 1e9                                       # overlap -> a different person
        for o in slot["tids"]:
            oi = info[o]
            if oi["b"] < t["a"]:
                p, q, gap = oi["g"].iloc[-1], t["g"].iloc[0], t["a"] - oi["b"]
            else:
                p, q, gap = t["g"].iloc[-1], oi["g"].iloc[0], oi["a"] - t["b"]
            d = np.hypot(p.x_m - q.x_m, p.y_m - q.y_m)
            allowed = 0.8 + 6.0 * gap / fps
            c = d / allowed + np.linalg.norm(oi["col"] - t["col"]) / 40.0 * (1.0 if has_col else 0.0)
            best = min(best, c)
        return best
    for tid in sorted((t for t in info if t not in used), key=lambda t: info[t]["a"]):
        costs = [slot_cost(s, info[tid]) for s in slots]
        k = int(np.argmin(costs))
        if costs[k] <= 2.5:
            slots[k]["tids"].append(tid); used.add(tid)
    # names
    order = sorted(range(n_players), key=lambda k: (np.median([info[o]["g"].y_m.median() for o in slots[k]["tids"]])
                                                    if slots[k]["tids"] else 99))
    for rank, k in enumerate(order):
        if not slots[k]["name"]:
            slots[k]["name"] = f"P{rank + 1}"
    t2n = {tid: s["name"] for s in slots for tid in s["tids"]}
    df = df[df.tid.isin(t2n)].copy()
    df["player_id"] = df.tid.map(t2n)
    df = df.drop_duplicates(["frame", "player_id"]).drop(columns=["tid"])
    return df.sort_values(["frame", "player_id"]).reset_index(drop=True)
