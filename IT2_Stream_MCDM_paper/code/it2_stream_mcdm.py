"""Honest C-MAPSS FD001 experiment for interval type-2 fuzzy TOPSIS ranking.

Decision task: at each cycle t, rank engines that have a completed W-cycle
window ending at t by maintenance urgency. Ground truth is true remaining
useful life (RUL) on the training trajectories (run-to-failure).

This is a single-host process-pool simulation of a map-reduce pattern, not
a Spark/YARN cluster. Numbers written to results/metrics.json are measured
on the machine that ran the script.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train_FD001.txt"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# FD001 sensor columns in the 26-column NASA layout (0-based).
# Sensors 1,5,6,10,16,18,19 are constant or near-constant on FD001.
SENSOR_COLS = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 21, 24, 25]
W = 20
HEALTHY = 30
T_START = 80
T_STEP = 10
N_WORKERS_LIST = (1, 2, 4)
NOISE_SEEDS = (11, 23, 41)
NOISE_SIGMA = 0.05
RUL_ALERT = 30
NDCG_K = 10


def load_train(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def per_engine(arr: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for u in np.unique(arr[:, 0].astype(int)):
        out[int(u)] = arr[arr[:, 0] == u]
    return out


def healthy_stats(eng: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    blocks = []
    for a in eng.values():
        n = min(HEALTHY, len(a))
        blocks.append(a[:n][:, SENSOR_COLS])
    X = np.vstack(blocks)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu, sd


def window_stats(row_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = row_block[:, SENSOR_COLS]
    return X.mean(axis=0), X.std(axis=0)


def it2_from_z(z_lo: np.ndarray, z_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map |z| interval through a saturating linear membership to [0,1].

    Urgency membership: 0 at |z|=0, 1 at |z|>=3. Interval endpoints give
    lower and upper memberships (IT2 interval type-reduction later uses both).
    """
    a = np.minimum(np.abs(z_lo), np.abs(z_hi))
    b = np.maximum(np.abs(z_lo), np.abs(z_hi))
    umf = np.clip(b / 3.0, 0.0, 1.0)
    lmf = np.clip(a / 3.0, 0.0, 1.0)
    return lmf, umf


def topsis_crisp(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    V = Z * w
    vp = V.max(axis=0)
    vn = V.min(axis=0)
    dp = np.sqrt(((V - vp) ** 2).sum(axis=1))
    dn = np.sqrt(((V - vn) ** 2).sum(axis=1))
    return dn / (dp + dn + 1e-12)


def topsis_it2(L: np.ndarray, U: np.ndarray, w: np.ndarray) -> np.ndarray:
    Lw, Uw = L * w, U * w
    vp_l, vp_u = Lw.max(axis=0), Uw.max(axis=0)
    vn_l, vn_u = Lw.min(axis=0), Uw.min(axis=0)
    dp = np.sqrt(0.5 * ((Lw - vp_l) ** 2 + (Uw - vp_u) ** 2).sum(axis=1))
    dn = np.sqrt(0.5 * ((Lw - vn_l) ** 2 + (Uw - vn_u) ** 2).sum(axis=1))
    return dn / (dp + dn + 1e-12)


def ndcg_at_k(rel: np.ndarray, k: int) -> float:
    k = min(k, len(rel))
    if k == 0:
        return 0.0
    gains = (2.0 ** rel[:k] - 1.0) / np.log2(np.arange(2, k + 2))
    dcg = gains.sum()
    ideal = np.sort(rel)[::-1]
    igains = (2.0 ** ideal[:k] - 1.0) / np.log2(np.arange(2, k + 2))
    idcg = igains.sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n < 2:
        return 1.0
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s1 = np.sign(a[i] - a[j])
            s2 = np.sign(b[i] - b[j])
            if s1 == 0 or s2 == 0:
                continue
            if s1 == s2:
                c += 1
            else:
                d += 1
    return float((c - d) / (c + d)) if (c + d) else 1.0


def _fuzzify_one(payload: tuple) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    uid, mean, std, mu, sd, noise, seed = payload
    rng = np.random.default_rng(seed + uid)
    mean = mean + rng.normal(0.0, noise * sd, size=mean.shape)
    z = (mean - mu) / sd
    z_lo = (mean - std - mu) / sd
    z_hi = (mean + std - mu) / sd
    lmf, umf = it2_from_z(z_lo, z_hi)
    crisp = np.clip(np.abs(z) / 3.0, 0.0, 1.0)
    return uid, lmf, umf, crisp, float(np.abs(z).mean())


def collect_windows(eng: dict[int, np.ndarray], t: int) -> list[tuple]:
    items = []
    for uid, a in eng.items():
        cyc = a[:, 1].astype(int)
        if t not in set(cyc):
            continue
        idx = np.where(cyc == t)[0][0]
        if idx + 1 < W:
            continue
        block = a[idx + 1 - W : idx + 1]
        m, s = window_stats(block)
        rul = float(len(a) - t)
        items.append((uid, m, s, rul))
    return items


def rank_batch(items, mu, sd, w, noise, seed, n_workers):
    payloads = [(uid, m, s, mu, sd, noise, seed) for uid, m, s, _ in items]
    if n_workers <= 1:
        mapped = [_fuzzify_one(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            mapped = list(ex.map(_fuzzify_one, payloads, chunksize=8))
    uids = [x[0] for x in mapped]
    L = np.vstack([x[1] for x in mapped])
    U = np.vstack([x[2] for x in mapped])
    C = np.vstack([x[3] for x in mapped])
    mag = np.array([x[4] for x in mapped])
    it2 = topsis_it2(L, U, w)
    t1 = topsis_crisp(C, w)
    return uids, it2, t1, mag


def evaluate(scores: np.ndarray, ruls: np.ndarray) -> dict:
    # higher score = more urgent; should track lower RUL
    rho = spearman(scores, -ruls)
    order = np.argsort(-scores)
    r_ord = ruls[order]
    rel = (r_ord <= RUL_ALERT).astype(float)
    prec = float(rel[: min(NDCG_K, len(rel))].mean()) if len(rel) else 0.0
    # graded relevance: more weight on smaller RUL
    graded = 1.0 / (1.0 + r_ord)
    ndcg = ndcg_at_k(graded, NDCG_K)
    return {"spearman": rho, "prec_at_10": prec, "ndcg_at_10": ndcg}


def main():
    arr = load_train(DATA)
    eng = per_engine(arr)
    mu, sd = healthy_stats(eng)
    m = len(SENSOR_COLS)
    w = np.ones(m) / m
    max_t = max(int(a[:, 1].max()) for a in eng.values())
    times = list(range(T_START, max_t + 1, T_STEP))

    # IsolationForest on healthy windows (fit once).
    from sklearn.ensemble import IsolationForest

    healthy_rows = []
    for a in eng.values():
        n = min(HEALTHY, len(a))
        healthy_rows.append(a[:n][:, SENSOR_COLS])
    H = np.vstack(healthy_rows)
    ifor = IsolationForest(n_estimators=100, random_state=0, contamination=0.05)
    ifor.fit(H)

    recs = []
    lat = {k: [] for k in N_WORKERS_LIST}
    for t in times:
        items = collect_windows(eng, t)
        if len(items) < 15:
            continue
        ruls = np.array([x[3] for x in items])
        t0 = time.perf_counter()
        uids, it2, t1, mag = rank_batch(items, mu, sd, w, 0.0, 0, 1)
        lat[1].append((time.perf_counter() - t0) * 1000.0)
        means = np.vstack([x[1] for x in items])
        if_score = -ifor.score_samples(means)  # higher = more anomalous
        ev_it2 = evaluate(it2, ruls)
        ev_t1 = evaluate(t1, ruls)
        ev_if = evaluate(if_score, ruls)
        ev_mag = evaluate(mag, ruls)
        recs.append(
            {
                "t": t,
                "n": len(items),
                "it2": ev_it2,
                "t1_topsis": ev_t1,
                "iforest": ev_if,
                "mean_abs_z": ev_mag,
            }
        )

    # worker sweep on a fixed late cycle for latency (same work, different pool size)
    late = [t for t in times if len(collect_windows(eng, t)) >= 40]
    t_lat = late[len(late) // 2] if late else times[-1]
    items_lat = collect_windows(eng, t_lat)
    for nw in N_WORKERS_LIST:
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            rank_batch(items_lat, mu, sd, w, 0.0, 0, nw)
            ts.append((time.perf_counter() - t0) * 1000.0)
        lat[nw] = ts

    # ranking stability under additive window-mean noise
    taus = []
    items_st = collect_windows(eng, t_lat)
    _, base, _, _ = rank_batch(items_st, mu, sd, w, 0.0, 0, 1)
    for s in NOISE_SEEDS:
        _, sc, _, _ = rank_batch(items_st, mu, sd, w, NOISE_SIGMA, s, 1)
        taus.append(kendall_tau(base, sc))

    def agg(key_path):
        vals = [r[key_path[0]][key_path[1]] for r in recs]
        a = np.array(vals, dtype=float)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "n_batches": int(len(a))}

    metrics = {
        "dataset": "NASA C-MAPSS FD001 train (run-to-failure)",
        "n_engines": len(eng),
        "n_rows": int(len(arr)),
        "sensors": SENSOR_COLS,
        "window": W,
        "healthy_cycles": HEALTHY,
        "alert_rul": RUL_ALERT,
        "n_eval_batches": len(recs),
        "latency_cycle_t": t_lat,
        "n_engines_latency_batch": len(items_lat),
        "latency_ms": {str(k): {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "n": len(v)} for k, v in lat.items()},
        "speedup_vs_1": {str(k): float(np.mean(lat[1]) / np.mean(lat[k])) for k in N_WORKERS_LIST},
        "kendall_tau_noise": {"mean": float(np.mean(taus)), "std": float(np.std(taus, ddof=1)), "seeds": list(NOISE_SEEDS), "sigma": NOISE_SIGMA},
        "it2": {m: agg(("it2", m)) for m in ("spearman", "prec_at_10", "ndcg_at_10")},
        "t1_topsis": {m: agg(("t1_topsis", m)) for m in ("spearman", "prec_at_10", "ndcg_at_10")},
        "iforest": {m: agg(("iforest", m)) for m in ("spearman", "prec_at_10", "ndcg_at_10")},
        "mean_abs_z": {m: agg(("mean_abs_z", m)) for m in ("spearman", "prec_at_10", "ndcg_at_10")},
        "implementation": "CPython process pool on one host; not a Spark cluster",
        "host": "local workstation, measured wall-clock",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
