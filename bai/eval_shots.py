"""Compare shots.csv against hand-labelled truth: python -m bai.eval_shots out/shots.csv truth/test4_shots.csv [--tol 1.0]

truth csv columns: t_s,hitter,type   (type in serve|clear|net|drop|lift|push|smash|drive|unknown; times may be rough)
Matching: order-preserving alignment by time (within --tol) and hitter."""
import argparse
import sys
import pandas as pd

FAMILY = {"net": "drop"}      # one camera cannot tell a net shot from a drop reliably -> scored as the same "หยอด" family
fam = lambda t: FAMILY.get(t, t)


def match(sys_df: pd.DataFrame, truth: pd.DataFrame, tol: float):
    """Order-preserving alignment (DP). Truth times are rough (whole seconds), so time alone mis-pairs neighbouring
    shots; pairs are scored by time closeness (needs |dt| <= tol) plus a bonus for the same hitter."""
    n, m = len(truth), len(sys_df)
    T, Sx = truth.t_s.to_numpy(), sys_df.t_s.to_numpy()
    th, sh = truth.hitter.astype(str).to_numpy(), sys_df.hitter.astype(str).to_numpy()
    sc = lambda i, k: (-1e9 if abs(T[i] - Sx[k]) > tol else (tol - abs(T[i] - Sx[k])) + (0.8 if th[i] == sh[k] else 0.0))
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for k in range(m - 1, -1, -1):
            D[i][k] = max(D[i + 1][k], D[i][k + 1], sc(i, k) + D[i + 1][k + 1])
    pair, i, k = {}, 0, 0
    while i < n and k < m:
        if sc(i, k) > -1e8 and abs(D[i][k] - (sc(i, k) + D[i + 1][k + 1])) < 1e-9:
            pair[i] = k; i += 1; k += 1
        elif abs(D[i][k] - D[i + 1][k]) < 1e-9:
            i += 1
        else:
            k += 1
    rows = [(t, pair.get(i)) for i, t in enumerate(truth.itertuples())]
    return rows, set(pair.values())


def evaluate(sys_df, truth, tol=1.0):
    rows, used = match(sys_df, truth, tol)
    out, n_h, ok_h, n_t, ok_t = [], 0, 0, 0, 0
    for t, k in rows:
        if k is None:
            out.append((t.t_s, t.hitter, t.type, "-", "-", "-", "MISSED", ""))
            continue
        s = sys_df.iloc[k]
        h_ok = str(s.hitter) == str(t.hitter)
        ty_ok = (t.type == "unknown") or (fam(s["type"]) == fam(t.type))
        n_h += 1; ok_h += h_ok
        if t.type != "unknown":
            n_t += 1; ok_t += (fam(s["type"]) == fam(t.type))
        out.append((t.t_s, t.hitter, t.type, round(s.t_s, 2), s.hitter, s["type"], "ok" if h_ok else "HITTER", "ok" if ty_ok else "TYPE"))
    extra = [sys_df.iloc[k] for k in range(len(sys_df)) if k not in used]
    summ = {"truth": len(truth), "system": len(sys_df), "detected": n_h, "hitter_ok": ok_h, "type_scored": n_t, "type_ok": ok_t,
            "extra": len(extra)}
    return out, extra, summ


def main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("shots"); ap.add_argument("truth")
    ap.add_argument("--tol", type=float, default=1.0)
    a = ap.parse_args(argv)
    sd = pd.read_csv(a.shots, encoding="utf-8-sig").sort_values("t_s").reset_index(drop=True)
    tr = pd.read_csv(a.truth, encoding="utf-8-sig").sort_values("t_s").reset_index(drop=True)
    out, extra, sm = evaluate(sd, tr, a.tol)
    print(f"{'เฉลย t':>7} {'คนตี':<10} {'ประเภท':<8} | {'ระบบ t':>7} {'คนตี':<10} {'ประเภท':<8} {'ผู้ตี':<7} {'ประเภท'}")
    for r in out:
        print(f"{r[0]:>7} {str(r[1]):<10} {r[2]:<8} | {str(r[3]):>7} {str(r[4]):<10} {str(r[5]):<8} {r[6]:<7} {r[7]}")
    for e in extra:
        print(f"  ส่วนเกิน (ระบบจับแต่ไม่อยู่ในเฉลย): t={e.t_s} {e.hitter} {e['type']}")
    print(f"\nจับได้ {sm['detected']}/{sm['truth']} | ผู้ตีถูก {sm['hitter_ok']}/{sm['detected']} | "
          f"ประเภทถูก {sm['type_ok']}/{sm['type_scored']} | ส่วนเกิน {sm['extra']}")


if __name__ == "__main__":
    main()
