"""Probe the video and warn when it violates the shooting conditions in the spec."""
from dataclasses import dataclass
import cv2


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int
    duration_s: float


def probe(path: str) -> tuple[VideoInfo, list[str]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"เปิดวิดีโอไม่ได้: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    warns0 = []
    try:                                   # screen recordings are variable-frame-rate: nominal fps lies
        import subprocess
        o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                            "-show_entries", "stream=nb_read_packets:format=duration", "-of", "default=nw=1:nk=0", path],
                           capture_output=True, text=True, timeout=120).stdout
        kv = dict(l.split("=", 1) for l in o.split() if "=" in l)
        nb, dur = float(kv.get("nb_read_packets", 0)), float(kv.get("duration", 0))
        if nb > 0 and dur > 0:
            eff = nb / dur
            if fps and abs(eff - fps) / fps > 0.08:
                warns0.append(f"วิดีโอเป็น VFR: fps ระบุ {fps:.1f} แต่จริง ~{eff:.1f} -> ใช้ {eff:.1f} (เวลาช็อต/ความเร็วจะถูกต้องกว่า)")
                fps, n = eff, int(nb)
    except Exception:
        pass
    info = VideoInfo(path, fps, w, h, n, n / fps if fps else 0.0)
    warns = list(warns0)
    if fps < 29.5:
        warns.append(f"fps {fps:.1f} < 30: ลูกจะเบลอ/หายบ่อย")
    elif fps < 59:
        warns.append(f"fps {fps:.1f}: แนะนำ 60 fps")
    if h < 720 or w < 1280:
        warns.append(f"ความละเอียด {w}x{h} < 1280x720: ลูกเล็กเกินไป")
    if h > w:
        warns.append("วิดีโอแนวตั้ง: ไม่ผ่านเงื่อนไข (ต้องแนวนอน)")
    if info.duration_s > 3600:
        warns.append("คลิปยาว > 60 นาที: ประมวลผลนาน")
    return info, warns


def frames(path: str):
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield i, f
        i += 1
    cap.release()
