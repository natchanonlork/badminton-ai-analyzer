"""Shuttle detection backends.

TrackNetV3 needs its official code + weights, which are not bundled here (licence check required,
and no weights available offline). `TrackNetBackend` is the adapter slot; `MotionBackend` is a
crude frame-difference fallback so the pipeline runs end to end. Do not trust MotionBackend accuracy.
"""
import cv2
import numpy as np
import pandas as pd


class ShuttleBackend:
    def process(self, frame):  # -> (x, y, conf) or None
        raise NotImplementedError


class NullBackend(ShuttleBackend):
    def process(self, frame):
        return None


class MotionBackend(ShuttleBackend):
    """Smallest bright moving blob. Placeholder only."""
    def __init__(self):
        self.prev = None

    def process(self, frame):
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        out = None
        if self.prev is not None:
            d = cv2.threshold(cv2.absdiff(g, self.prev), 40, 255, cv2.THRESH_BINARY)[1]
            n, _, st, cen = cv2.connectedComponentsWithStats(d)
            cand = [(st[k, cv2.CC_STAT_AREA], cen[k]) for k in range(1, n)
                    if 4 <= st[k, cv2.CC_STAT_AREA] <= 150]
            if cand:
                a, c = min(cand, key=lambda t: t[0])
                out = (float(c[0]), float(c[1]), 0.3)
        self.prev = g
        return out


def TrackNetBackend(weights):
    if not weights:
        raise SystemExit("--shuttle tracknet ต้องมี --shuttle-weights (เทรนเองด้วย python -m bai.tracknet train)")
    from .tracknet import TrackNetInfer
    return TrackNetInfer(weights)


def get_backend(name: str, weights: str | None = None) -> ShuttleBackend:
    if name == "none":
        return NullBackend()
    if name == "motion":
        return MotionBackend()
    if name == "tracknet":
        return TrackNetBackend(weights)
    raise ValueError(name)


def to_frame_df(dets: dict[int, tuple], n_frames: int, max_gap: int = 5) -> pd.DataFrame:
    """Per-frame table with short gaps (<= max_gap frames) linearly interpolated."""
    x = np.full(n_frames, np.nan); y = np.full(n_frames, np.nan); c = np.zeros(n_frames)
    for f, (px, py, cf) in dets.items():
        x[f], y[f], c[f] = px, py, cf
    idx = np.where(~np.isnan(x))[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            x[a + 1:b] = x[a] + t * (x[b] - x[a])
            y[a + 1:b] = y[a] + t * (y[b] - y[a])
            c[a + 1:b] = 0.1   # interpolated -> low confidence
    ok = ~np.isnan(x)
    return pd.DataFrame({"frame": np.where(ok)[0], "x_px": x[ok], "y_px": y[ok], "conf": c[ok]})


def from_tracknetv3_csv(path: str, hot_frac: float = 0.05, cell: int = 40):
    """Import TrackNetV3 `*_ball.csv` (Frame,Visibility,X,Y) and clean it:
    - drop 'hot cells' holding > hot_frac of all detections (a shuttle can't sit in one spot: static false positive)
    - drop isolated detections (no other detection within +-3 frames and 200 px)
    Returns (DataFrame frame,x_px,y_px,conf ; stats dict)."""
    d = pd.read_csv(path)
    v = d[d.Visibility == 1][["Frame", "X", "Y"]].rename(columns={"Frame": "frame", "X": "x_px", "Y": "y_px"}).copy()
    n0 = len(v)
    cx, cy = (v.x_px // cell).astype(int), (v.y_px // cell).astype(int)
    key = cx.astype(str) + "_" + cy.astype(str)
    share = key.map(key.value_counts() / max(n0, 1))
    hot = share > hot_frac
    v = v[~hot].reset_index(drop=True)
    fr, x, y = v.frame.to_numpy(), v.x_px.to_numpy(), v.y_px.to_numpy()
    keep = np.zeros(len(v), bool)
    j0 = 0
    for i in range(len(v)):
        while fr[j0] < fr[i] - 3:
            j0 += 1
        j = j0
        while j < len(v) and fr[j] <= fr[i] + 3:
            if j != i and np.hypot(x[j] - x[i], y[j] - y[i]) < 200:
                keep[i] = True
                break
            j += 1
    v = v[keep].reset_index(drop=True)
    v["conf"] = 0.8
    return v, {"total_frames": len(d), "visible_raw": n0, "dropped_hot": int(hot.sum()), "kept": len(v),
               "hot_cells": sorted(set(key[hot]))[:5]}

