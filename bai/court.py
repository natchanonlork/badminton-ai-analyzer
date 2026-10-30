"""Court geometry: homography pixel -> metres.

Metres: origin (0,0) = near-left corner, x along the baseline (0..6.10),
y away from the camera (0..13.40). Doubles court.
Corner order everywhere: near-left, near-right, far-right, far-left.
"""
import cv2
import numpy as np

COURT_W, COURT_L = 6.10, 13.40
COURT_M = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], np.float32)


def homography(corners_px) -> np.ndarray:
    c = np.asarray(corners_px, np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(c, COURT_M)


def to_metres(H: np.ndarray, pts_px) -> np.ndarray:
    p = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def to_pixels(H: np.ndarray, pts_m) -> np.ndarray:
    p = np.asarray(pts_m, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.linalg.inv(H)).reshape(-1, 2)


def _order(q: np.ndarray) -> np.ndarray:
    q = q.reshape(4, 2)
    q = q[np.argsort(q[:, 1])]            # top two first
    far, near = q[:2], q[2:]
    far = far[np.argsort(far[:, 0])]      # far-left, far-right
    near = near[np.argsort(near[:, 0])]   # near-left, near-right
    return np.array([near[0], near[1], far[1], far[0]], np.float32)


def auto_detect(sample_frames: list[np.ndarray]):
    """Best-effort: outer quad of the white-line mask. Returns (corners_px, confidence).

    Only works when the whole court outline is visible and the camera is static.
    Confidence is low (<0.6) when the guess is doubtful -> caller must ask for --corners.
    """
    if not sample_frames:
        return None, 0.0
    med = np.median(np.stack(sample_frames), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.08, 15):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            quad = a
            break
    if quad is None:
        return None, 0.0
    frame_area = med.shape[0] * med.shape[1]
    q_area = cv2.contourArea(quad)
    fill = cv2.contourArea(cnt) / max(q_area, 1)      # lines are thin -> small fill is normal
    corners = _order(quad)
    # plausibility: area share, far side narrower than near side (camera behind court)
    share = q_area / frame_area
    near_w = np.linalg.norm(corners[1] - corners[0])
    far_w = np.linalg.norm(corners[2] - corners[3])
    conf = 0.0
    if 0.15 < share < 0.95 and far_w < near_w:
        conf = float(min(0.9, 0.5 + share / 2))
    return corners, conf * (1.0 if fill > 0.02 else 0.5)


# ---- landmark-based calibration (works when the near baseline is cut off) ----
_X1, _X2, _XC, _XS1, _XS2 = 0.0, COURT_W, COURT_W / 2, 0.46, COURT_W - 0.46
LANDMARKS = {
    "near_left_corner": (0, 0), "near_right_corner": (COURT_W, 0),
    "far_right_corner": (COURT_W, COURT_L), "far_left_corner": (0, COURT_L),
    "near_short_service_left": (0, 4.72), "near_short_service_right": (COURT_W, 4.72),
    "far_short_service_left": (0, 8.68), "far_short_service_right": (COURT_W, 8.68),
    "net_left": (0, 6.7), "net_right": (COURT_W, 6.7),
    "near_long_service_left": (0, 0.76), "near_long_service_right": (COURT_W, 0.76),
    "far_long_service_left": (0, 12.64), "far_long_service_right": (COURT_W, 12.64),
    "centre_near_short_service": (_XC, 4.72), "centre_far_short_service": (_XC, 8.68),
    "centre_near_baseline": (_XC, 0), "centre_far_baseline": (_XC, COURT_L),
    "singles_near_short_left": (_XS1, 4.72), "singles_near_short_right": (_XS2, 4.72),
    "singles_far_short_left": (_XS1, 8.68), "singles_far_short_right": (_XS2, 8.68),
    "singles_near_baseline_left": (_XS1, 0), "singles_near_baseline_right": (_XS2, 0),
    "singles_far_baseline_left": (_XS1, COURT_L), "singles_far_baseline_right": (_XS2, COURT_L),
}


def homography_from_landmarks(items):
    """items: [{name,x,y}] pixel clicks of named landmarks (>=4, not all collinear) -> H pixel->metres."""
    if len(items) < 4:
        raise ValueError("ต้องมีจุดอ้างอิงอย่างน้อย 4 จุด")
    src = np.array([[i["x"], i["y"]] for i in items], np.float32)
    dst = np.array([LANDMARKS[i["name"]] for i in items], np.float32)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise ValueError("คำนวณไม่ได้ (จุดอาจอยู่แนวเดียวกันหมด)")
    return H


def corners_from_H(H):
    """Project the 4 outer court corners back into pixels (may lie outside the frame)."""
    return to_pixels(H, COURT_M)
