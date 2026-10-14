"""Early/late split and vectorized scale-up. Reads the same FD001 file."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from it2_stream_mcdm import (
    DATA,
    OUT,
    SENSOR_COLS,
    T_START,
    T_STEP,
    W,
    collect_windows,
    evaluate,
    healthy_stats,
    load_train,
    per_engine,
    rank_batch,
    topsis_it2,
)

def main():
    arr = load_train(DATA)
    eng = per_engine(arr)
    mu, sd = healthy_stats(eng)
    w = np.ones(len(SENSOR_COLS)) / len(SENSOR_COLS)
    max_t = max(int(a[:, 1].max()) for a in eng.values())
    times = list(range(T_START, max_t + 1, T_STEP))
    early, late = [], []
    for t in times:
        items = collect_windows(eng, t)
        if len(items) < 15:
            continue
        ruls = np.array([x[3] for x in items])
        _, it2, _, _ = rank_batch(items, mu, sd, w, 0.0, 0, 1)
        ev = evaluate(it2, ruls)
        ev["t"] = t
        ev["n"] = len(items)
        ev["mean_rul"] = float(ruls.mean())
        if ruls.mean() > 80:
            early.append(ev)
        else:
            late.append(ev)

    def pack(xs):
        if not xs:
            return None
        return {
            "n_batches": len(xs),
            "spearman": [float(np.mean([x["spearman"] for x in xs])), float(np.std([x["spearman"] for x in xs], ddof=1))],
            "prec_at_10": [float(np.mean([x["prec_at_10"] for x in xs])), float(np.std([x["prec_at_10"] for x in xs], ddof=1))],
            "ndcg_at_10": [float(np.mean([x["ndcg_at_10"] for x in xs])), float(np.std([x["ndcg_at_10"] for x in xs], ddof=1))],
        }

    # Vectorized IT2 TOPSIS scale-up (single process, no pool).
    rng = np.random.default_rng(0)
    scale = {}
    for n in (100, 500, 2000, 8000, 20000):
        L = rng.random((n, 13))
        U = np.minimum(1.0, L + rng.random((n, 13)) * 0.15)
        t0 = time.perf_counter()
        for _ in range(20):
            topsis_it2(L, U, w)
        ms = (time.perf_counter() - t0) * 1000.0 / 20.0
        scale[str(n)] = ms

    extra = {"early_mean_rul_gt_80": pack(early), "late_mean_rul_le_80": pack(late), "vectorized_it2_topsis_ms": scale}
    (OUT / "extra.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    print(json.dumps(extra, indent=2))


if __name__ == "__main__":
    main()
