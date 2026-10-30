"""Clip cutting helpers (ffmpeg). NEVER overwrites the source or an existing file.

  python -m bai.cut scenes in.mp4 [--min-sec 8] [--court]     list scene cuts; with --court: stretches where the court is visible AND the camera is static
  python -m bai.cut clip   in.mp4 --start 1:23.5 --end 2:00    cut one range -> clips/in_<start>-<end>.mp4
  python -m bai.cut auto   in.mp4 [--min-sec 10] [--top 5]     cut the longest court-visible shots automatically
Outputs go to ./clips next to the input (or --out-dir). Video is re-encoded (H.264) so cuts are frame-accurate; audio kept (AAC) when the source has it.
"""
import argparse, os, re, subprocess, sys
import cv2
from . import court


def parse_t(s: str) -> float:
    s = str(s).strip()
    if ":" in s:
        parts = [float(x) for x in s.split(":")]
        t = 0.0
        for p in parts:
            t = t * 60 + p
        return t
    return float(s)


def fmt_t(t: float) -> str:
    return f"{int(t // 60)}m{t % 60:05.2f}s"


def duration(path):
    cap = cv2.VideoCapture(path)
    d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    return d


def scene_cuts(path, thr=0.3):
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-vf", f"select='gt(scene,{thr})',showinfo", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    return sorted(float(m) for m in re.findall(r"pts_time:([0-9.]+)", p.stderr))


def segments(path, thr=0.3):
    cuts = [0.0] + scene_cuts(path, thr) + [duration(path)]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 0.2]


def court_visible(path, a, b, n=3):
    """True if auto court detection succeeds on >= 2 of n frames sampled inside [a,b]."""
    cap = cv2.VideoCapture(path)
    ok = 0
    frames = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, (a + (b - a) * (k + 1) / (n + 1)) * 1000)
        r, f = cap.read()
        if r:
            frames.append(f)
    cap.release()
    for f in frames:
        _, conf = court.auto_detect([f])
        ok += conf >= 0.6
    return ok >= 2


def stable_court_runs(path, step_s=0.5, tol_px=25.0, min_sec=8.0):
    """Runs where the court is detected AND its outline stays put (static broadcast camera).
    Returns [(start_s, end_s, median_corners 4x2)]. Zooms/pans/close-ups break a run."""
    import numpy as np
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(int(round(step_s * fps)), 1)
    runs, cur, i = [], [], 0
    def flush():
        nonlocal cur
        if cur and cur[-1][0] - cur[0][0] >= min_sec:
            runs.append((cur[0][0], cur[-1][0], np.median([c for _, c in cur], axis=0)))
        cur = []
    while True:
        if not cap.grab():
            break
        if i % stride == 0:
            ok, f = cap.retrieve()
            c, conf = court.auto_detect([f]) if ok else (None, 0)
            t = i / fps
            if c is not None and conf >= 0.6:
                if cur and np.abs(c - cur[-1][1]).max() > tol_px:
                    flush()
                cur.append((t, c))
            else:
                flush()
        i += 1
    flush()
    cap.release()
    return runs


def cut(path, start, end, out_dir=None, out=None):
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(path)), "clips")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = out or os.path.join(out_dir, f"{stem}_{start:.1f}-{end:.1f}.mp4")
    if os.path.abspath(out) == os.path.abspath(path):
        raise SystemExit("ปฏิเสธ: ไฟล์ปลายทางคือไฟล์ต้นฉบับ")
    if os.path.exists(out):
        raise SystemExit(f"ปฏิเสธ: มีไฟล์นี้อยู่แล้ว (ไม่เขียนทับ) {out}")
    subprocess.run(["ffmpeg", "-v", "error", "-n", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", path,
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", out], check=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scenes", "clip", "auto"])
    ap.add_argument("video")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--min-sec", type=float, default=8.0)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.3, help="scene-cut sensitivity (lower = more cuts)")
    ap.add_argument("--court", action="store_true", help="scenes: keep only court-visible shots")
    ap.add_argument("--out-dir")
    a = ap.parse_args(argv)
    if a.cmd == "clip":
        if a.start is None or a.end is None:
            sys.exit("ต้องใส่ --start และ --end")
        s, e = parse_t(a.start), parse_t(a.end)
        if e <= s:
            sys.exit("--end ต้องมากกว่า --start")
        print(cut(a.video, s, e, a.out_dir))
        return
    if a.cmd == "scenes" and not a.court:
        rows = [(x, y, None) for x, y in segments(a.video, a.thr) if y - x >= a.min_sec]
        for x, y, _ in sorted(rows, key=lambda r: -(r[1] - r[0])):
            print(f"{fmt_t(x)} -> {fmt_t(y)}  ({y - x:5.1f}s)")
    else:
        rows = [(x, y, c) for x, y, c in stable_court_runs(a.video, min_sec=a.min_sec)]
        rows.sort(key=lambda r: -(r[1] - r[0]))
        if a.cmd == "auto":
            rows = rows[:a.top]
        for x, y, c in rows:
            print(f"{fmt_t(x)} -> {fmt_t(y)}  ({y - x:5.1f}s)  มุมคอร์ท(ประมาณ ใกล้ซ้าย,ใกล้ขวา,ไกลขวา,ไกลซ้าย)="
                  + " ".join(str(int(round(v))) for v in c.reshape(-1)))
        if rows:
            print("หมายเหตุ: มุมคอร์ทที่พิมพ์เป็นค่าประมาณหยาบจากขอบพื้นสนาม คลาดได้หลายสิบพิกเซล -> ใช้ดูว่า 'เห็นคอร์ท+กล้องนิ่ง' เท่านั้น "
                  "ค่าที่ใช้วัดจริงให้คลิปมุมเองใน pick_corners.html")
        if a.cmd == "auto":
            for x, y, _ in rows:
                print("cut ->", cut(a.video, x + 0.3, y - 0.3, a.out_dir))
    if not rows:
        print("ไม่พบช่วงที่ตรงเงื่อนไข (ลองลด --min-sec หรือเพิ่ม --thr)")


if __name__ == "__main__":
    main()
