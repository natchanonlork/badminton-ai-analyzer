"""TrackNet-style shuttle detector, written from scratch (NOT the official TrackNetV3 and NOT trained).

Input : 3 consecutive frames stacked (9 ch), resized to 512x288.
Output: 3 heatmaps (one per input frame); we use the last one at inference.
Without trained weights this detects NOTHING useful. Train with `python -m bai.tracknet train ...`.

Label CSV per video (one row per frame): frame,visible,x,y   (x,y in original pixels)
UNTESTED here: the build sandbox has no torch. Run `python -m bai.tracknet selftest` on your GPU box first.
"""
import argparse, sys
import numpy as np
import cv2

IN_W, IN_H = 512, 288


def _torch():
    import torch, torch.nn as nn
    return torch, nn


def build_model():
    torch, nn = _torch()

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3, self.e4 = blk(9, 32), blk(32, 64), blk(64, 128), blk(128, 256)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.d3, self.d2, self.d1 = blk(256 + 128, 128), blk(128 + 64, 64), blk(64 + 32, 32)
            self.out = nn.Conv2d(32, 3, 1)

        def forward(self, x):
            a = self.e1(x); b = self.e2(self.pool(a)); c = self.e3(self.pool(b)); d = self.e4(self.pool(c))
            u = self.d3(torch.cat([self.up(d), c], 1))
            u = self.d2(torch.cat([self.up(u), b], 1))
            u = self.d1(torch.cat([self.up(u), a], 1))
            return torch.sigmoid(self.out(u))
    return Net()


def prep(frames):
    """3 BGR frames -> float32 (9,H,W)"""
    ims = [cv2.resize(f, (IN_W, IN_H)).astype(np.float32) / 255.0 for f in frames]
    return np.concatenate([i.transpose(2, 0, 1) for i in ims], 0)


def gaussian_heatmap(x, y, sx, sy, sigma=3.0):
    """x,y original px -> (H,W) heatmap; zeros if invisible (x is None)."""
    hm = np.zeros((IN_H, IN_W), np.float32)
    if x is None:
        return hm
    cx, cy = x * IN_W / sx, y * IN_H / sy
    xs, ys = np.arange(IN_W), np.arange(IN_H)[:, None]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def decode(hm, size, thr=0.5):
    """(H,W) heatmap -> (x,y,conf) in original px, or None."""
    m = (hm > thr).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cx, cy = cen[k]
    return float(cx * size[0] / IN_W), float(cy * size[1] / IN_H), float(hm.max())


class TrackNetInfer:
    def __init__(self, weights, device=None):
        torch, _ = _torch()
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_model().to(self.dev).eval()
        self.net.load_state_dict(torch.load(weights, map_location=self.dev))
        self.buf = []

    def process(self, frame):
        self.buf = (self.buf + [frame])[-3:]
        if len(self.buf) < 3:
            return None
        x = self.torch.from_numpy(prep(self.buf))[None].to(self.dev)
        with self.torch.no_grad():
            hm = self.net(x)[0, 2].cpu().numpy()
        return decode(hm, (frame.shape[1], frame.shape[0]))


# ---------------- training ----------------
def wbce(pred, gt, eps=1e-7):
    torch, _ = _torch()
    pred = pred.clamp(eps, 1 - eps)
    return -(((1 - pred) ** 2) * gt * torch.log(pred) + (pred ** 2) * (1 - gt) * torch.log(1 - pred)).mean()


def train(pairs, out, epochs=30, bs=8, lr=1.0, device=None):
    """pairs: list of (video_path, labels_csv). Adadelta lr=1.0 as in TrackNet papers."""
    import pandas as pd
    torch, _ = _torch()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    samples = []
    for vp, lp in pairs:
        cap = cv2.VideoCapture(vp)
        sx, sy = int(cap.get(3)), int(cap.get(4)); cap.release()
        samples.append((vp, pd.read_csv(lp).set_index("frame"), sx, sy))

    def get(i_sample, f):
        vp, lab, sx, sy = samples[i_sample]
        cap = cv2.VideoCapture(vp); cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        fr = [cap.read()[1] for _ in range(3)]; cap.release()
        if any(x is None for x in fr):
            return None
        hms = []
        for k in range(3):
            r = lab.loc[f + k] if (f + k) in lab.index else None
            vis = r is not None and int(r.visible) == 1
            hms.append(gaussian_heatmap(r.x if vis else None, r.y if vis else None, sx, sy))
        return prep(fr), np.stack(hms)
    # NOTE: re-opening the video per sample is slow; fine for a first experiment, cache frames for real runs.
    net = build_model().to(dev)
    opt = torch.optim.Adadelta(net.parameters(), lr=lr)
    idx = [(s, int(f)) for s, (_, lab, _, _) in enumerate(samples) for f in lab.index
           if (f + 1) in lab.index and (f + 2) in lab.index]   # only fully-labelled 3-frame windows
    for ep in range(epochs):
        np.random.shuffle(idx); tot = n = 0
        for b in range(0, len(idx) - bs, bs):
            batch = [g for g in (get(s, f) for s, f in idx[b:b + bs]) if g]
            if not batch:
                continue
            x = torch.tensor(np.stack([g[0] for g in batch])).to(dev)
            y = torch.tensor(np.stack([g[1] for g in batch])).to(dev)
            loss = wbce(net(x), y); opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); n += 1
        print(f"epoch {ep + 1}/{epochs} loss {tot / max(n, 1):.5f}")
        torch.save(net.state_dict(), out)


def selftest():
    torch, _ = _torch()
    net = build_model()
    y = net(torch.zeros(1, 9, IN_H, IN_W))
    assert y.shape == (1, 3, IN_H, IN_W), y.shape
    hm = gaussian_heatmap(640, 360, 1280, 720)
    assert decode(hm, (1280, 720)) is not None
    x, yy, _ = decode(hm, (1280, 720))
    assert abs(x - 640) < 4 and abs(yy - 360) < 4
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "selftest"])
    ap.add_argument("--pair", nargs=2, action="append", metavar=("VIDEO", "LABELS_CSV"))
    ap.add_argument("--out", default="tracknet.pt"); ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else train(a.pair, a.out, a.epochs)
