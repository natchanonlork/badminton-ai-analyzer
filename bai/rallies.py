"""Phase 2: hits, rallies, analysis.json. Pure numpy/pandas (no GPU).

hit  = shuttle velocity reverses direction (>90 deg) AND the shuttle is inside a player's (padded) bbox.
rally= run of hits where consecutive hits are <= MAX_HIT_GAP_S apart; needs >= MIN_HITS.
Land point of a shot = next hit's shuttle position projected to court plane (approximation).
"""
import numpy as np
import pandas as pd
from . import shots as S
from .court import to_metres

MAX_HIT_GAP_S = 2.5
MIN_HITS = 2
PRE_S, POST_S = 0.2, 0.8      # shuttle is drawn/kept only inside rally windows (hides shuttles from neighbouring courts)
TURN_MIN = 0.5
REACH = 1.0                  # max shuttle-to-player-bbox distance (in bbox heights) to count as a hit
ALT_MAX_GAP_S = 0.9
G0, K_DIST = 0.35, 1.5       # hit decoder: gain = G0 + turn_score - K_DIST * distance(bbox heights)
SAME_SIDE_GAP_S, SAME_SIDE_PEN = 1.3, 1.5   # two hits from one side within 1.3 s: shuttle never crossed the net -> penalty
MIN_HIT_GAP_S = 0.2
SERVE_REACH = 0.4           # the first hit of a rally (the serve) must be right at the server: stray shuttles from other courts are not
FILL_GAP = 40               # frames: interpolate player boxes across detection gaps up to this long


def _smooth(a, k=3):
    return pd.Series(a).rolling(k, center=True, min_periods=1).mean().to_numpy()


def turning_points(sh: pd.DataFrame, fps: float):
    """Candidate hit frames from the shuttle path alone -> list of (frame, x, y, score).
    score = |v2-v1| / (|v1|+|v2|): 1.0 = full reversal, 0.6 ~ 75 deg turn. Catches lifts/drops whose
    x-direction continues (the old 'cos<0' rule missed those) and tolerates ~30 fps clips."""
    span = max(2, int(round(fps * 0.08)))
    sh = sh.sort_values("frame").reset_index(drop=True)
    if len(sh) < 2 * span + 1:
        return []
    f, x, y = sh.frame.to_numpy(), _smooth(sh.x_px), _smooth(sh.y_px)
    out = []
    for i in range(span, len(sh) - span):
        if f[i + span] - f[i - span] > 3 * span:           # data gap too large
            continue
        v1 = np.array([x[i] - x[i - span], y[i] - y[i - span]])
        v2 = np.array([x[i + span] - x[i], y[i + span] - y[i]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 + n2 < 12:                                    # nearly static (noise)
            continue
        sc = float(np.linalg.norm(v2 - v1) / (n1 + n2))
        if sc >= TURN_MIN:
            out.append((int(f[i]), float(x[i]), float(y[i]), sc))
    res, min_gap = [], int(0.25 * fps)
    for c in out:                                           # keep sharpest turn within a cluster
        if res and c[0] - res[-1][0] < min_gap:
            if c[3] > res[-1][3]:
                res[-1] = c
        else:
            res.append(c)
    return res


def fill_players(pl: pd.DataFrame, max_gap=FILL_GAP) -> pd.DataFrame:
    """Linear-interpolate each player's box/position over detection gaps <= max_gap frames (far players are often lost
    for ~1 s behind the net or near players, exactly when they hit). Longer gaps stay empty."""
    cols = ["x_px", "y_px", "x_m", "y_m", "bx1", "by1", "bx2", "by2"]
    outs = []
    for pid, g in pl.groupby("player_id"):
        g = g.sort_values("frame").drop_duplicates("frame").set_index("frame")
        idx = np.arange(int(g.index.min()), int(g.index.max()) + 1)
        h = g[cols].reindex(idx)
        miss = h.x_px.isna().to_numpy()
        run = np.zeros(len(h), int)
        i = 0
        while i < len(h):                       # length of the NaN run each row belongs to
            if miss[i]:
                j = i
                while j < len(h) and miss[j]:
                    j += 1
                run[i:j] = j - i
                i = j
            else:
                i += 1
        h = h.interpolate(limit_area="inside")
        h[(run > max_gap)] = np.nan
        h = h.dropna()
        h["player_id"] = pid
        outs.append(h.rename_axis("frame").reset_index())
    return pd.concat(outs, ignore_index=True) if outs else pl


def _player_dists(byf, frame, x, y):
    """{player_id: d} d = distance shuttle->bbox in bbox heights (0 = inside). Uses frame +-2 (first offset with data)."""
    for df in range(0, 3):
        out = {}
        for fr in ({frame - df, frame + df}):
            g = byf.get(fr)
            if g is None:
                continue
            for r in g.itertuples():
                dx = max(r.bx1 - x, 0, x - r.bx2)
                dy = max(r.by1 - y, 0, y - r.by2)
                d = float(np.hypot(dx, dy)) / max(r.by2 - r.by1, 1.0)
                if r.player_id not in out or d < out[r.player_id]:
                    out[r.player_id] = d
        if out:
            return out
    return {}


def _nearest_player(byf, frame, x, y):
    dd = _player_dists(byf, frame, x, y)
    if not dd:
        return None, float("inf")
    pid = min(dd, key=dd.get)
    return pid, dd[pid]


def detect_hits(sh: pd.DataFrame, pl: pd.DataFrame, fps: float, span=None, debug=None, reach=REACH, side=None):
    """sh: frame,x_px,y_px ; pl: frame,player_id,bx1,by1,bx2,by2,(x_m,y_m)
    -> DataFrame(frame,player_id,x_px,y_px,score)
    Candidates = shuttle turning points. Each may belong to any player within `reach` bbox-heights; a small DP then picks
    the subset + owners with best total gain, penalising two consecutive hits from one side (the shuttle must cross the net).
    debug: optional list; gets one dict per candidate (+ the chosen player or None)."""
    cols = ["frame", "player_id", "x_px", "y_px", "score"]
    if len(sh) < 5:
        return pd.DataFrame(columns=cols)
    plf = fill_players(pl)
    byf = {k: g for k, g in plf.groupby("frame")}
    if side is None:
        side = {pid: ("near" if g.y_m.median() < 6.7 else "far") for pid, g in pl.groupby("player_id")}
    cands = turning_points(sh, fps)
    opts = []
    for fr, x, y, sc in cands:
        dd = _player_dists(byf, fr, x, y)
        opts.append([(pid, d) for pid, d in sorted(dd.items(), key=lambda t: t[1]) if d <= reach])
    # --- DP over (candidate, owner)
    NEG = -1e9
    best, back = [], []
    for i, (fr, x, y, sc) in enumerate(cands):
        row, brow = {}, {}
        for pid, d in opts[i]:
            gain = G0 + sc - K_DIST * d
            bv, bp = gain, None                                  # start a new chain here
            for j in range(i):
                dt = (fr - cands[j][0]) / fps
                if dt < MIN_HIT_GAP_S:
                    continue
                for q, v in best[j].items():
                    pen = 0.0
                    if side.get(q) == side.get(pid):
                        pen = SAME_SIDE_PEN if dt <= SAME_SIDE_GAP_S else 0.3
                    if v + gain - pen > bv:
                        bv, bp = v + gain - pen, (j, q)
            row[pid], brow[pid] = bv, bp
        best.append(row); back.append(brow)
    chosen = {}
    end = max(((v, i, p) for i, r in enumerate(best) for p, v in r.items()), default=None)
    while end:
        _, i, p = end
        chosen[i] = p
        nxt = back[i][p]
        end = (best[nxt[0]][nxt[1]], nxt[0], nxt[1]) if nxt else None
    # a rally's first hit must be close to its hitter (serve); otherwise drop it and let the next one lead
    dist_of = [dict(o) for o in opts]
    order = sorted(chosen)
    prev_t, changed = None, True
    while changed:
        changed = False
        order = sorted(chosen)
        prev_t = None
        for i in order:
            t = cands[i][0] / fps
            if (prev_t is None or t - prev_t > MAX_HIT_GAP_S) and dist_of[i][chosen[i]] > SERVE_REACH:
                del chosen[i]; changed = True
                break
            prev_t = t
    rows = []
    for i, (fr, x, y, sc) in enumerate(cands):
        if debug is not None:
            debug.append({"frame": fr, "t_s": round(fr / fps, 3), "x_px": round(x), "y_px": round(y), "score": round(sc, 2),
                          "options": " ".join(f"{p}:{d:.2f}" for p, d in opts[i]), "chosen": chosen.get(i)})
        if i in chosen:
            rows.append((fr, chosen[i], x, y, sc))
    return pd.DataFrame(rows, columns=cols)


def enforce_alternation(hits: pd.DataFrame, side: dict, fps: float, max_gap_s=ALT_MAX_GAP_S):
    """Two consecutive hits from the same side within max_gap_s cannot both be real (the shuttle must cross the
    net in between) -> keep the sharper turn. Hits further apart are kept (an intermediate hit was probably missed)."""
    keep = []
    for r in hits.itertuples():
        if keep and side.get(r.player_id) == side.get(keep[-1].player_id) and (r.frame - keep[-1].frame) / fps <= max_gap_s:
            if r.score > keep[-1].score:
                keep[-1] = r
        else:
            keep.append(r)
    return pd.DataFrame([tuple(k)[1:] for k in keep], columns=hits.columns) if keep else hits.iloc[0:0]


def distance_cum(g: pd.DataFrame, fps: float, step_s=0.5, dead_m=0.05) -> dict:
    """frame -> cumulative distance (m) of one player. Foot points jitter (esp. far side, small bbox), and summing
    per-frame steps turns that jitter into fake running. So: median-smooth, sample every step_s, ignore steps < dead_m,
    ignore gaps in the data. Slightly under-counts tight curves; far less wrong than the naive sum."""
    g = g.sort_values("frame").drop_duplicates("frame")
    if len(g) < 2:
        return {}
    f0, f1 = int(g.frame.min()), int(g.frame.max())
    idx = np.arange(f0, f1 + 1)
    x = pd.Series(g.x_m.to_numpy(), index=g.frame.to_numpy()).reindex(idx)
    y = pd.Series(g.y_m.to_numpy(), index=g.frame.to_numpy()).reindex(idx)
    lim = max(int(fps), 5)
    sm = lambda v: (v.interpolate(limit=lim, limit_area="inside").rolling(7, center=True, min_periods=1).median()
                    .rolling(max(int(fps * 0.5), 5), center=True, min_periods=1).mean())
    x, y = sm(x), sm(y)
    k = max(int(round(fps * step_s)), 1)
    xs, ys = x.to_numpy()[::k], y.to_numpy()[::k]
    step = np.hypot(np.diff(xs), np.diff(ys))
    step = np.where(np.isfinite(step) & (step >= dead_m) & (step < 4.0), step, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(step)])
    return {int(fr): float(cs[min(j // k, len(cs) - 1)]) for j, fr in enumerate(idx)}


def shuttle_speed_kmh(sdf: pd.DataFrame, fps: float, smooth=5, vmax=500.0) -> dict:
    """frame -> estimated shuttle speed, km/h. ESTIMATE: the shuttle is projected onto the court plane although it flies
    above it, so a single camera over/under-states the true speed (error can reach tens of %). Use for relative comparison."""
    if sdf is None or len(sdf) < 3 or "x_m" not in sdf:
        return {}
    d = sdf.sort_values("frame").reset_index(drop=True)
    f = d.frame.to_numpy()
    xm = pd.Series(d.x_m.to_numpy()).rolling(3, center=True, min_periods=1).median().to_numpy()
    ym = pd.Series(d.y_m.to_numpy()).rolling(3, center=True, min_periods=1).median().to_numpy()
    sp = np.full(len(d), np.nan)
    for i in range(1, len(d)):
        df = f[i] - f[i - 1]
        if 0 < df <= 2:
            sp[i] = np.hypot(xm[i] - xm[i - 1], ym[i] - ym[i - 1]) / (df / fps) * 3.6
    sp = pd.Series(sp).rolling(smooth, center=True, min_periods=2).median().clip(upper=vmax)
    return {int(a): float(b) for a, b in zip(f, sp) if not np.isnan(b)}


class _Feet:
    """Ground position of players from tracked foot points (reliable), nearest frame within tol."""
    def __init__(self, players_df, tol=6):
        self.tol, self.d = tol, {}
        for pid, g in players_df.groupby("player_id"):
            g = g.sort_values("frame")
            self.d[pid] = (g.frame.to_numpy(), g.x_m.to_numpy(), g.y_m.to_numpy())

    def at(self, pid, frame):
        if pid not in self.d:
            return None
        f, x, y = self.d[pid]
        j = int(np.searchsorted(f, frame))
        c = [k for k in (j - 1, j) if 0 <= k < len(f)]
        if not c:
            return None
        k = min(c, key=lambda k: abs(f[k] - frame))
        return (float(x[k]), float(y[k])) if abs(f[k] - frame) <= self.tol else None


def build_rallies(hits: pd.DataFrame, H, fps: float, players_df: pd.DataFrame, speed=None):
    if hits.empty:
        return []
    team = {}
    for pid, g in players_df.groupby("player_id"):
        team[pid] = "near" if g.y_m.median() < 6.7 else "far"
    hits = hits.reset_index(drop=True)
    if hits.empty:
        return []
    m = to_metres(H, hits[["x_px", "y_px"]].values)
    hits = hits.assign(x_m=m[:, 0], y_m=m[:, 1], t_s=hits.frame / fps).reset_index(drop=True)
    # The shuttle is airborne, so projecting it onto the court plane gives nonsense (y = -113 m, 1370 m seen).
    # Use the hitter's FEET at the hit (and the next hitter's feet as the landing spot): both are on the ground.
    feet = _Feet(players_df)
    fx = [feet.at(p_, f_) for p_, f_ in zip(hits.player_id, hits.frame)]
    hits["fx_m"] = [q[0] if q else np.nan for q in fx]
    hits["fy_m"] = [q[1] if q else np.nan for q in fx]
    groups, cur = [], [0]
    for i in range(1, len(hits)):
        if hits.t_s[i] - hits.t_s[i - 1] <= MAX_HIT_GAP_S:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    rallies = []
    for gi in [g for g in groups if len(g) >= MIN_HITS]:
        shots = []
        for n, i in enumerate(gi):
            nxt = gi[n + 1] if n + 1 < len(gi) else None
            hxy = (float(hits.fx_m[i]), float(hits.fy_m[i]))
            if nxt is None or np.isnan(hxy[0]) or np.isnan(hits.fx_m[nxt]):
                typ, conf, land, flight = "unknown", 0.0, None, None   # last shot / missing feet: unknown
                if n == 0 and not np.isnan(hxy[0]):
                    typ, conf = "serve", 0.6
            else:
                land = (float(hits.fx_m[nxt]), float(hits.fy_m[nxt])); flight = float(hits.t_s[nxt] - hits.t_s[i])
                typ, conf = S.classify(hxy, land, flight, is_first=(n == 0))
            sp = None
            if speed:   # peak (80th pct) speed in the 0.3 s after contact
                w = [speed[k] for k in range(int(hits.frame[i]), int(hits.frame[i] + 0.3 * fps) + 1) if k in speed]
                sp = round(float(np.percentile(w, 80)), 0) if len(w) >= 3 else None
            shots.append({"idx": n + 1, "t_s": round(float(hits.t_s[i]), 3), "frame": int(hits.frame[i]),
                          "hitter": hits.player_id[i], "type": typ, "type_conf": round(conf, 2),
                          "hit_xy_m": [round(hxy[0], 2), round(hxy[1], 2)] if not np.isnan(hxy[0]) else None,
                          "land_xy_m": [round(land[0], 2), round(land[1], 2)] if land else None,
                          **({"speed_kmh_est": sp} if speed else {})})
        rallies.append({"id": len(rallies) + 1, "start_s": shots[0]["t_s"], "end_s": shots[-1]["t_s"],
                        "shot_count": len(shots), "winner_team": None, "end_reason": "unknown", "shots": shots})
    return rallies


def player_stats(players_df: pd.DataFrame, fps: float, rallies):
    out = []
    for pid, g in players_df.groupby("player_id"):
        g = g.sort_values("frame")
        dc = distance_cum(g, fps)
        cnt = {}
        for r in rallies:
            for s in r["shots"]:
                if s["hitter"] == pid:
                    cnt[s["type"]] = cnt.get(s["type"], 0) + 1
        out.append({"id": pid, "distance_m": round(max(dc.values(), default=0.0), 1), "shots": cnt})
    return out
