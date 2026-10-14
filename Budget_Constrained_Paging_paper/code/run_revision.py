"""Revision experiments on CIC-IDS2017 minute windows.

1. Repairs the 12-hour clock: afternoon flows stored as 1:00-5:00 without PM
   land before each morning. Windows in the first block of each capture day
   are shifted by +720 minutes, which restores true time order.
2. Adds calibrated-threshold and incident-deduplicated baselines, a 2x2
   mask/Lagrangian x stub ablation, an F_max sweep, incident-level metrics,
   rolling-origin day folds and block-bootstrap intervals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cic_windows import load_cache
from risk_models import PORT, STRUCT, lexical_stub, matrix, mlp_risk, standardize, temporal_ema, train_mlp
from sklearn.ensemble import IsolationForest

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
SEEDS = range(10)
FMAX = 0.10
SWEEP = [0.02, 0.05, 0.10, 0.15, 0.20]
EDGE = True  # rising-edge paging: one page per alert episode


def repair_clock(feat: dict) -> dict:
    """Return windows in true time order with a capture-day index."""
    w = feat["win"].astype(np.int64)
    gap_idx = np.where(np.diff(w) > 60)[0]
    seg = np.zeros(len(w), dtype=int)
    for i in gap_idx:
        seg[i + 1:] += 1
    if seg.max() + 1 == 5:  # cache built with the repaired parser
        out = dict(feat)
        out["day"] = seg
        out["clock_min"] = (w - w.min()) % 1440
        out["ema_out"] = temporal_ema(out["max_out"])
        return out
    if seg.max() + 1 != 10:
        raise RuntimeError(f"expected 10 blocks (5 days x AM/PM), got {seg.max() + 1}")
    shifted = w + np.where(seg % 2 == 0, 720, 0)  # first block of each day is the afternoon
    order = np.argsort(shifted, kind="stable")
    out = {k: v[order] for k, v in feat.items()}
    out["win"] = shifted[order]
    out["day"] = (seg // 2)[order]
    out["clock_min"] = (out["win"] - out["win"].min()) % 1440
    out["ema_out"] = temporal_ema(out["max_out"])
    return out


def incidents(y: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(y):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(y) - 1))
    return runs


def metrics(y: np.ndarray, a: np.ndarray) -> dict:
    y, a = y.astype(int), a.astype(int)
    tp, fp = int(((y == 1) & (a == 1)).sum()), int(((y == 0) & (a == 1)).sum())
    pos, neg = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    inc = incidents(y)
    hit, delay = 0, []
    for s, e in inc:
        p = np.where(a[s:e + 1] == 1)[0]
        if len(p):
            hit += 1
            delay.append(int(p[0]))
    return {
        "tpr": tp / pos,
        "far": fp / neg,
        "interrupt": float(a.mean()),
        "precision": tp / max(tp + fp, 1),
        "inc_recall": hit / max(len(inc), 1),
        "inc_hit": hit,
        "n_inc": len(inc),
        "delay_min": float(np.mean(delay)) if delay else float("nan"),
    }


def thresh(r, thr):
    return (r >= thr).astype(np.int8)


def calibrated_thr(r_train: np.ndarray, fmax: float) -> float:
    return float(np.quantile(r_train, 1.0 - fmax))


def lagrangian(r, fmax, base, stub=None, lr=0.25, lmax=8.0, slope=0.08, edge=False):
    out = np.zeros(len(r), dtype=np.int8)
    lam, n_int, prev = 0.0, 0, False
    for t in range(len(r)):
        hot = r[t] >= base + slope * lam and (stub is None or stub[t] == 1)
        ok = hot and not (edge and prev)
        prev = hot
        out[t] = int(ok)
        n_int += int(ok)
        lam = float(np.clip(lam + lr * (n_int / (t + 1) - fmax), 0.0, lmax))
    return out


def hybrid(r, fmax, base, stub=None, lr=0.25, lmax=8.0, slope=0.08):
    """Dual threshold inside the hard mask: adapts like the Lagrangian, capped like the mask.

    The dual is driven by the requested (pre-mask) paging rate, so the threshold rises
    whenever demand exceeds the budget even though realised pages never do.
    """
    out = np.zeros(len(r), dtype=np.int8)
    lam, n_int, n_want = 0.0, 0, 0
    for t in range(len(r)):
        want = r[t] >= base + slope * lam and (stub is None or stub[t] == 1)
        n_want += int(want)
        if want and (n_int + 1) / (t + 1) <= fmax:
            out[t] = 1
            n_int += 1
        lam = float(np.clip(lam + lr * (n_want / (t + 1) - fmax), 0.0, lmax))
    return out


def mask(r, fmax, base, stub=None, edge=False):
    out = np.zeros(len(r), dtype=np.int8)
    n_int, prev = 0, False
    for t in range(len(r)):
        hot = r[t] >= base and (stub is None or stub[t] == 1)
        want = hot and not (edge and prev)
        prev = hot
        if want and (n_int + 1) / (t + 1) <= fmax:
            out[t] = 1
            n_int += 1
    return out


def iforest(x_tr, x_all, seed):
    clf = IsolationForest(n_estimators=80, contamination=0.2, random_state=seed, n_jobs=1).fit(x_tr)
    s_tr, s = -clf.score_samples(x_tr), -clf.score_samples(x_all)
    lo, hi = np.percentile(s_tr, 1), np.percentile(s_tr, 99)  # train-only scaling
    return np.clip((s - lo) / max(hi - lo, 1e-9), 0, 1)


def auroc(y, r):
    pos, neg = r[y == 1], r[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def scores(feat, tr, seed):
    y = feat["y"].astype(int)
    kf, ks = STRUCT + PORT + ["ema_out"], STRUCT + ["ema_out"]
    xf = standardize(matrix(feat, kf)[tr], matrix(feat, kf))
    xs = standardize(matrix(feat, ks)[tr], matrix(feat, ks))
    r_full = mlp_risk(train_mlp(xf[tr], y[tr], seed=seed), xf)
    r_struct = mlp_risk(train_mlp(xs[tr], y[tr], seed=seed + 17), xs)
    r_if = iforest(xf[tr], xf, seed)
    deg_thr = max(np.quantile(feat["max_out"][tr], 0.70), 1.0)
    r_deg = np.clip(feat["max_out"] / deg_thr, 0, 1)
    return {"full": r_full, "struct": r_struct, "if": r_if, "deg": r_deg}, lexical_stub(feat, tr)


def policies(R, stub, tr, te, fmax):
    s = stub[te]
    rf, rs = R["full"][te], R["struct"][te]
    cf, cs = calibrated_thr(R["full"][tr], fmax), calibrated_thr(R["struct"][tr], fmax)
    return {
        "degree": thresh(R["deg"][te], 1.0),
        "iforest": thresh(R["if"][te], 0.5),
        "lexical": s.astype(np.int8),
        "mlp_full": thresh(rf, 0.5),
        "mlp_struct": thresh(rs, 0.5),
        "mlp_full+lex": ((rf >= 0.35) & (s == 1)).astype(np.int8),
        "calib_full": thresh(rf, cf),
        "calib_struct": thresh(rs, cs),
        "lag_full": lagrangian(rf, fmax, 0.5),
        "lag_full+lex": lagrangian(rf, fmax, 0.5, s),
        "lag_struct": lagrangian(rs, fmax, 0.5),
        "mask_full": mask(rf, fmax, 0.5),
        "mask_full+lex": mask(rf, fmax, 0.5, s),
        "mask_full+lex@0.35": mask(rf, fmax, 0.35, s),
        "mask_struct": mask(rs, fmax, 0.5),
        "hyb_full": hybrid(rf, fmax, 0.5),
        "hyb_full+lex": hybrid(rf, fmax, 0.5, s),
        "lag_full+lex+edge": lagrangian(rf, fmax, 0.5, s, edge=True),
        "mask_full+lex+edge": mask(rf, fmax, 0.5, s, edge=True),
    }


def agg(rows: list[dict], fmax: float = FMAX) -> dict:
    out = {}
    for name in rows[0]:
        out[name] = {}
        for k in rows[0][name]:
            v = np.array([r[name][k] for r in rows], dtype=float)
            out[name][k] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1))}
        out[name]["seeds_over_cap"] = int(sum(r[name]["interrupt"] > fmax + 1e-12 for r in rows))
    return out


def block_bootstrap_diff(y, a, b, block=30, n_boot=2000, rng=None):
    """95% interval of TPR(a) - TPR(b) under a moving-block bootstrap over test windows."""
    rng = rng or np.random.default_rng(0)
    n = len(y)
    starts = np.arange(n - block + 1)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n // block + 1)])[:n]
        yy = y[idx]
        if yy.sum() == 0:
            continue
        diffs.append((a[idx][yy == 1].mean()) - (b[idx][yy == 1].mean()))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def load_unsw(label_thr: float) -> dict:
    z = np.load(ROOT / "results" / "unsw_windows.npz")
    feat = {k: z[k] for k in z.files if k != "start_unix"}
    feat["y"] = (feat["atk_frac"] >= label_thr).astype(np.int8)
    w = feat["win"]
    feat["day"] = np.concatenate([[0], np.cumsum(np.diff(w) > 60)])
    feat["clock_min"] = (w - w.min()) % 1440
    feat["ema_out"] = temporal_ema(feat["max_out"])
    return feat


def main(dataset: str = "cic", label_thr: float = 0.5) -> None:
    feat = repair_clock(load_cache()) if dataset == "cic" else load_unsw(label_thr)
    names = DAYS if dataset == "cic" else [f"block{i}" for i in range(int(feat["day"].max()) + 1)]
    y = feat["y"].astype(int)
    n = len(y)
    ntr = int(0.7 * n)
    tr, te = np.arange(ntr), np.arange(ntr, n)
    res = {
        "data": {
            "n_windows": n,
            "n_train": ntr,
            "n_test": n - ntr,
            "train_pos": int(y[tr].sum()),
            "test_pos": int(y[te].sum()),
            "windows_per_day": [int((feat["day"] == d).sum()) for d in range(5)],
            "pos_per_day": [int(y[feat["day"] == d].sum()) for d in range(5)],
            "split_day": names[int(feat["day"][ntr])],
            "label_thr": label_thr,
            "split_clock": f"{int(feat['clock_min'][ntr]) // 60:02d}:{int(feat['clock_min'][ntr]) % 60:02d} (relative to first flow)",
            "train_incident_lengths": [e - s + 1 for s, e in incidents(y[tr])],
            "test_incident_lengths": [e - s + 1 for s, e in incidents(y[te])],
        }
    }
    yte = y[te]
    rows, sweep_rows, aucs, seed0 = [], {f: [] for f in SWEEP}, [], None
    for sd in SEEDS:
        R, stub = scores(feat, tr, sd)
        aucs.append({k: auroc(yte, R[k][te]) for k in ("full", "struct", "if", "deg")})
        P = policies(R, stub, tr, te, FMAX)
        rows.append({k: metrics(yte, a) for k, a in P.items()})
        if sd == 0:
            seed0 = P
        for f in SWEEP:
            Q = {
                "calib_full": thresh(R["full"][te], calibrated_thr(R["full"][tr], f)),
                "lag_full": lagrangian(R["full"][te], f, 0.5),
                "lag_full+lex": lagrangian(R["full"][te], f, 0.5, stub[te]),
                "mask_full+lex": mask(R["full"][te], f, 0.5, stub[te]),
                "hyb_full": hybrid(R["full"][te], f, 0.5),
                "hyb_full+lex": hybrid(R["full"][te], f, 0.5, stub[te]),
                "lag_full+lex+edge": lagrangian(R["full"][te], f, 0.5, stub[te], edge=True),
            }
            sweep_rows[f].append({k: metrics(yte, a) for k, a in Q.items()})
    res["main"] = agg(rows)
    res["auroc"] = {k: {"mean": float(np.mean([a[k] for a in aucs])), "std": float(np.std([a[k] for a in aucs], ddof=1))} for k in aucs[0]}
    res["sweep"] = {str(f): agg(sweep_rows[f], f) for f in SWEEP}
    res["bootstrap_seed0_tpr_diff"] = {
        "lag_full+lex - lag_full": block_bootstrap_diff(yte, seed0["lag_full+lex"], seed0["lag_full"]),
        "mask_full+lex - mask_full+lex@0.35": block_bootstrap_diff(yte, seed0["mask_full+lex"], seed0["mask_full+lex@0.35"]),
        "lag_full+lex - mask_full+lex": block_bootstrap_diff(yte, seed0["lag_full+lex"], seed0["mask_full+lex"]),
    }
    # Rolling-origin day folds: train on all earlier days, test on one day.
    folds = {}
    nb = int(feat["day"].max()) + 1
    for d in range(max(1, nb - 2), nb):
        trd, ted = np.where(feat["day"] < d)[0], np.where(feat["day"] == d)[0]
        frows = []
        for sd in SEEDS:
            R, stub = scores(feat, trd, sd)
            P = policies(R, stub, trd, ted, FMAX)
            frows.append({k: metrics(y[ted], P[k]) for k in ("degree", "calib_full", "lag_full", "lag_full+lex", "mask_full+lex", "mask_full+lex@0.35", "hyb_full", "hyb_full+lex", "lag_full+lex+edge")})
        folds[names[d]] = {"n_test": len(ted), "test_pos": int(y[ted].sum()), "test_incidents": len(incidents(y[ted])), "res": agg(frows)}
    res["day_folds"] = folds
    tag = "" if dataset == "cic" else f"_unsw_{int(label_thr * 100)}"
    out = ROOT / "results" / f"revision_metrics{tag}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["data"], indent=2))
    hdr = f"{'policy':22s} {'TPR':>11s} {'FAR':>11s} {'Int':>11s} {'Prec':>6s} {'IncRec':>7s} {'Delay':>6s} over"
    print(hdr)
    for k, v in res["main"].items():
        print(f"{k:22s} {v['tpr']['mean']:.3f}+-{v['tpr']['std']:.3f} {v['far']['mean']:.3f}+-{v['far']['std']:.3f} "
              f"{v['interrupt']['mean']:.3f}+-{v['interrupt']['std']:.3f} {v['precision']['mean']:.3f} "
              f"{v['inc_recall']['mean']:.3f} {v['delay_min']['mean']:6.2f} {v['seeds_over_cap']}")
    print("AUROC", json.dumps(res["auroc"]))
    print("boot", json.dumps(res["bootstrap_seed0_tpr_diff"]))
    for f in SWEEP:
        print("Fmax", f, {k: (round(v["tpr"]["mean"], 3), round(v["interrupt"]["mean"], 3), round(v["inc_recall"]["mean"], 2)) for k, v in res["sweep"][str(f)].items()})
    for d, v in folds.items():
        print(d, v["n_test"], v["test_pos"], v["test_incidents"], {k: (round(x["tpr"]["mean"], 3), round(x["interrupt"]["mean"], 3), round(x["inc_recall"]["mean"], 2)) for k, x in v["res"].items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cic", float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
