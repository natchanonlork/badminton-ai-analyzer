"""Overlay renderer: foot circles + names (Thai OK via PIL), shuttle ring + trail, shot labels, mini court."""
import os
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .court import COURT_W, COURT_L

COLORS = [(60, 76, 231), (219, 152, 52), (113, 204, 46), (15, 196, 241)]  # BGR
TRAIL = 10
MM_W, MM_H = 110, 246
SHOT_TH = {"serve": "เสิร์ฟ", "clear": "เคลียร์", "smash": "ตบ", "drop": "หยอด", "net": "เน็ต",
           "drive": "ไดรฟ์", "lift": "ลิฟต์", "push": "ผลัก"}
SHOW_S = 1.0
_FONTS = ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/LeelawUI.ttf", "C:/Windows/Fonts/leelawad.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
_cache = {}


def _font(size):
    for p in _FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def put_text(fr, text, xy, color_bgr, size=26, bg=True):
    """Draw unicode text (cached RGBA patch alpha-blended onto the frame). xy = top-left."""
    key = (text, size, color_bgr, bg)
    if key not in _cache:
        f = _font(size)
        l, t, r, b = f.getbbox(text)
        w, h = r - l + 10, b - t + 8
        im = Image.new("RGBA", (w, h), (0, 0, 0, 150 if bg else 0))
        ImageDraw.Draw(im).text((5 - l, 4 - t), text, font=f, fill=(color_bgr[2], color_bgr[1], color_bgr[0], 255))
        _cache[key] = np.asarray(im)
    p = _cache[key]
    h, w = p.shape[:2]
    x, y = int(xy[0]), int(xy[1])
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, fr.shape[1]), min(y + h, fr.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    q = p[y0 - y:y1 - y, x0 - x:x1 - x]
    a = q[..., 3:4].astype(np.float32) / 255.0
    roi = fr[y0:y1, x0:x1].astype(np.float32)
    fr[y0:y1, x0:x1] = (roi * (1 - a) + q[..., [2, 1, 0]].astype(np.float32) * a).astype(np.uint8)


def _minimap(players_now):
    img = np.full((MM_H, MM_W, 3), 40, np.uint8)
    sx, sy = (MM_W - 10) / COURT_W, (MM_H - 10) / COURT_L
    p = lambda x, y: (int(5 + x * sx), int(MM_H - 5 - y * sy))
    cv2.rectangle(img, p(0, 0), p(COURT_W, COURT_L), (230, 230, 230), 1)
    cv2.line(img, p(0, COURT_L / 2), p(COURT_W, COURT_L / 2), (230, 230, 230), 1)
    for _, (x, y, col) in players_now.items():
        cv2.circle(img, p(x, y), 5, col, -1)
    return img


def _cum_dist(players_df, fps=30.0):
    """player_id -> {frame: cumulative distance m} (same smoothing/deadband as player_stats)."""
    from .rallies import distance_cum
    return {pid: distance_cum(g, fps) for pid, g in players_df.groupby("player_id")}


def _fmt_t(sec):
    sec = max(sec, 0.0)
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def _rect(fr, x0, y0, x1, y1, alpha=0.55, color=(30, 20, 20)):
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), fr.shape[1]), min(int(y1), fr.shape[0])
    if x1 > x0 and y1 > y0:
        roi = fr[y0:y1, x0:x1].astype(np.float32)
        fr[y0:y1, x0:x1] = (roi * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def _brackets(fr, c, r, col, t=2):
    x, y = c
    k = max(r // 2, 6)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cv2.line(fr, (x + sx * r, y + sy * r), (x + sx * (r - k), y + sy * r), col, t, cv2.LINE_AA)
            cv2.line(fr, (x + sx * r, y + sy * r), (x + sx * r, y + sy * (r - k)), col, t, cv2.LINE_AA)


def render(video_path, out_path, fps, size, H, players_df, shuttle_df, shots=None, rallies=None,
           progress=None, total=None, speed=None):
    """shots: list of dict(frame, hitter, type, rally, idx). rallies: list of dict(start_s,end_s,id)."""
    w, h = size
    cap = cv2.VideoCapture(video_path)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-i", video_path, "-map", "0:v:0", "-map", "1:a:0?",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-shortest", out_path],
        stdin=subprocess.PIPE)
    pids = sorted(players_df.player_id.unique()) if len(players_df) else []
    col = {pid: COLORS[k % len(COLORS)] for k, pid in enumerate(pids)}
    by_frame = {f: g for f, g in players_df.groupby("frame")} if len(players_df) else {}
    sh = {int(r.frame): (r.x_px, r.y_px) for r in shuttle_df.itertuples()} if len(shuttle_df) else {}
    all_shots = list(shots or [])
    shots = [s for s in all_shots if s["type"] != "unknown"]
    rallies = rallies or []
    if rallies:   # draw the shuttle only inside rally windows: stray shuttles from other courts appear outside them
        from .rallies import PRE_S, POST_S
        sh = {k: v for k, v in sh.items()
              if any(r["start_s"] - PRE_S <= k / fps <= r["end_s"] + POST_S for r in rallies)}
    speed = speed or {}
    cum = _cum_dist(players_df, fps) if len(players_df) else {}
    sc = max(h / 1080.0, 0.6)
    fz = lambda v: max(int(v * sc), 12)
    keep = int(SHOW_S * fps)
    last_pos = {}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        now = {}
        g = by_frame.get(i)
        if g is not None:
            for r in g.itertuples():
                c = col[r.player_id]
                last_pos[r.player_id] = (int(r.x_px), int(r.y_px), int(r.by1) if hasattr(r, "by1") else int(r.y_px) - 120)
                cv2.ellipse(fr, (int(r.x_px), int(r.y_px)), (max(int((r.bx2 - r.bx1) * 0.6), 20), 12), 0, 0, 360, c, 3)
                dist = cum.get(r.player_id, {}).get(i)
                tag = f"{r.player_id}  {dist:.0f} m" if dist is not None else str(r.player_id)
                put_text(fr, tag, (int(r.x_px) - 40, int(r.by1) - fz(40)), c, fz(24))
                now[r.player_id] = (r.x_m, r.y_m, c)
        pts = [sh[j] for j in range(i - TRAIL, i + 1) if j in sh]
        for k in range(1, len(pts)):
            cv2.line(fr, tuple(map(int, pts[k - 1])), tuple(map(int, pts[k])), (0, 255, 255), 2, cv2.LINE_AA)
        if pts:
            p_ = tuple(map(int, pts[-1]))
            _brackets(fr, p_, fz(18), (0, 255, 255), 2)
            v_ = speed.get(i)
            if v_ is not None and v_ >= 5:
                put_text(fr, f"~{v_:.0f} km/h", (p_[0] + fz(24), p_[1] - fz(36)), (0, 255, 255), fz(22))
        for s in shots:
            if 0 <= i - s["frame"] < keep and s["hitter"] in last_pos:
                x, _, top = last_pos[s["hitter"]]
                put_text(fr, f'{s["hitter"]} · {SHOT_TH.get(s["type"], s["type"])}', (x - 60, top - fz(84)),
                         col.get(s["hitter"], (255, 255, 255)), 28)
        t = i / fps
        past = [r for r in rallies if r["start_s"] <= t + 0.05]
        r = past[-1] if past else None
        if r is not None and t <= r["end_s"] + 1.5:
            rt = min(t, r["end_s"]) - r["start_s"]
            n = sum(1 for s in all_shots if s["rally"] == r["id"] and s["frame"] <= i)
            last = [s for s in all_shots if s["rally"] == r["id"] and s["frame"] <= i]
            spd_txt = (f'{last[-1]["hitter"]} · {SHOT_TH.get(last[-1]["type"], "ไม่ทราบ")}' if last else "-")
            rid = r["id"]
        else:
            rt, n, spd_txt, rid = 0.0, 0, "-", "-"
        bw, bh = int(w * 0.46), fz(74)
        bx = (w - bw) // 2
        _rect(fr, bx, 8, bx + bw, 8 + bh)
        for k, (lab, val) in enumerate((("เวลาแรลลี่", _fmt_t(rt)), (f"จำนวนช็อต (แรลลี่ {rid})", str(n)),
                                        ("ช็อตล่าสุด", spd_txt))):
            x0 = bx + int(bw * (0.0, 0.34, 0.70)[k]) + fz(10)
            put_text(fr, lab, (x0, 10), (200, 200, 200), fz(15), bg=False)
            put_text(fr, val, (x0, 10 + fz(22)), (255, 255, 255), fz(34), bg=False)
        # player stats panel (bottom-left)
        if pids:
            rh = fz(34)
            px0, py0 = 10, h - 10 - rh * len(pids) - fz(8)
            _rect(fr, px0, py0, px0 + fz(300), h - 10)
            for k, pid in enumerate(pids):
                dd = cum.get(pid, {})
                dv = dd.get(i) if i in dd else (max((v for f_, v in dd.items() if f_ <= i), default=0.0) if dd else 0.0)
                ns = sum(1 for s in all_shots if s["hitter"] == pid and s["frame"] <= i)
                y0 = py0 + fz(4) + k * rh
                cv2.rectangle(fr, (px0 + 6, y0 + 4), (px0 + 6 + fz(14), y0 + 4 + fz(22)), col[pid], -1)
                put_text(fr, f"{pid}   {dv:.1f} m   {ns} ช็อต", (px0 + fz(26), y0), (255, 255, 255), fz(20), bg=False)
        mm = _minimap(now)
        fr[10:10 + MM_H, w - MM_W - 10:w - 10] = mm
        ff.stdin.write(fr.tobytes())
        if progress and total and i % 50 == 0:
            progress("render", i / total)
        i += 1
    ff.stdin.close(); ff.wait(); cap.release()
