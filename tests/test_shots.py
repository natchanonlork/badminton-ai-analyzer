import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bai import shots, tracknet
import numpy as np

CASES = [  # hit_xy, land_xy, flight_s, expected
    ((3, 12.5), (3, 4.5), 0.9, "drop"),                          # rear -> just over net, 8.9 m/s
    ((3, 12.5), (3, 4.5), 0.2, "smash"),                         # 8m in .2s = 40 m/s
    ((3, 12.8), (3, 13.0 - 0.0), 0.0, None),                     # skipped (zero flight)
    ((3, 5.0), (3, 12.0), 1.0, "lift"),                         # front court underarm to rear
    ((3, 5.5), (3, 7.5), 0.5, "net"),
    ((3, 12.5), (3, 1.2), 1.4, "clear"),                         # far-end clear to the other rear
]


def test_cases():
    # clear: rear hit at y=12.5 (5.8 m behind net) landing y=1.2 (5.5 m deep on other side) in 1.4 s
    for h, l, t, exp in CASES:
        if exp is None or t <= 0:
            continue
        got, c = shots.classify(h, l, t)
        print(h, l, t, got, round(c, 2))
        assert got == exp, (h, l, t, got, exp)


def test_first_is_serve():
    assert shots.classify((3, 1), (3, 10), 1.0, is_first=True)[0] == "serve"


def test_not_crossing_unknown():
    assert shots.classify((3, 12), (3, 9), 0.5)[0] == "unknown"


def test_tracknet_numpy_parts():
    hm = tracknet.gaussian_heatmap(640, 360, 1280, 720)
    x, y, _ = tracknet.decode(hm, (1280, 720))
    assert abs(x - 640) < 4 and abs(y - 360) < 4
    assert tracknet.decode(np.zeros((tracknet.IN_H, tracknet.IN_W), np.float32), (1280, 720)) is None
    assert tracknet.prep([np.zeros((720, 1280, 3), np.uint8)] * 3).shape == (9, 288, 512)
