"""Rule-based shot-type guess from one camera's court-plane data (metres, net at y=6.70).

Inputs per shot: hit_xy (hitter's feet ~ hit point), land_xy (next shot's hit point or detected landing),
flight_s (time to next contact). Hit HEIGHT is unknown from one camera, so smash vs fast clear/drive
is ambiguous -> we return a confidence and 'unknown' when rules conflict.
Thresholds are first guesses: CALIBRATE on labelled shots (ShuttleSet) before trusting any number.
"""
import math
from dataclasses import dataclass

NET_Y = 6.70


@dataclass
class ShotFeatures:
    hit_depth: float     # m from net on hitter's side
    land_depth: float    # m from net on opponent's side (negative = landed on own side/no crossing)
    dist: float          # horizontal distance hit->land, m
    flight_s: float
    speed_ms: float      # mean horizontal speed, m/s


def features(hit_xy, land_xy, flight_s) -> ShotFeatures:
    d = math.dist(hit_xy, land_xy)
    hit_depth = abs(hit_xy[1] - NET_Y)
    same_side = (hit_xy[1] - NET_Y) * (land_xy[1] - NET_Y) > 0
    land_depth = abs(land_xy[1] - NET_Y) * (-1 if same_side else 1)
    return ShotFeatures(hit_depth, land_depth, d, flight_s, d / flight_s if flight_s > 0 else float("inf"))


def classify(hit_xy, land_xy, flight_s, is_first=False):
    """-> (type, confidence 0..1).  hit_xy = hitter's feet, land_xy = next hitter's feet (both on the court plane).
    Geometry (m from the net) + flight time only; hit HEIGHT is unknown from one camera, so every rule is a guess:
      front hit (<3.5): net (lands <3.5) | lift (lands deep, slow) | push (lands deep, fast)
      rear hit (>=4.0): drop (lands <4.3; smash if the flight is <0.5 s) | clear (lands deep, slow) | drive (deep, fast)
      mid  (3.5-4.0):   clear (deep) else drop
    Calibrated only on ONE labelled clip (tests/ + truth/test4_shots.csv): validate on other clips before trusting."""
    if is_first:
        return "serve", 0.6          # first contact of a rally; long vs short serve not split here
    f = features(hit_xy, land_xy, flight_s)
    if f.land_depth < 0:
        return "unknown", 0.1        # didn't cross: miss / data error
    ft = f.flight_s
    front, rear = f.hit_depth < 3.5, f.hit_depth >= 4.0
    deep = f.land_depth >= 4.3
    if front:
        if f.land_depth < 3.5:
            return "net", 0.55
        return ("lift", 0.5) if ft >= 0.7 else ("push", 0.4)
    if rear:
        if not deep:
            return ("smash", 0.4) if ft < 0.5 else ("drop", 0.55)
        return ("clear", 0.55) if ft >= 0.7 else ("drive", 0.4)
    return ("clear", 0.4) if deep else ("drop", 0.4)
