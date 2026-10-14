from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from it2_stream_mcdm import (
    DATA,
    OUT,
    SENSOR_COLS,
    T_START,
    T_STEP,
    collect_windows,
    evaluate,
    healthy_stats,
    load_train,
    per_engine,
    rank_batch,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

arr = load_train(DATA)
eng = per_engine(arr)
mu, sd = healthy_stats(eng)
w = np.ones(len(SENSOR_COLS)) / len(SENSOR_COLS)
max_t = max(int(a[:, 1].max()) for a in eng.values())
ts, rho, p10, nr = [], [], [], []
for t in range(T_START, max_t + 1, T_STEP):
    items = collect_windows(eng, t)
    if len(items) < 15:
        continue
    ruls = np.array([x[3] for x in items])
    _, it2, _, _ = rank_batch(items, mu, sd, w, 0.0, 0, 1)
    ev = evaluate(it2, ruls)
    ts.append(t)
    rho.append(ev["spearman"])
    p10.append(ev["prec_at_10"])
    nr.append(ruls.mean())

fig, ax = plt.subplots(figsize=(5.2, 2.6))
ax.plot(ts, rho, "o-", color="#1f4e79", ms=4, label="Spearman vs $-$RUL")
ax.plot(ts, p10, "s--", color="#8b1e1e", ms=4, label="Precision@10 (RUL$\\leq$30)")
ax.set_xlabel("Cycle $t$")
ax.set_ylabel("Score")
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_rank_vs_t.pdf")
fig.savefig(FIG / "fig_rank_vs_t.png", dpi=160)

extra = json.loads((OUT / "extra.json").read_text())
ns = [int(k) for k in extra["vectorized_it2_topsis_ms"]]
ms = [extra["vectorized_it2_topsis_ms"][str(k)] for k in ns]
fig, ax = plt.subplots(figsize=(5.2, 2.6))
ax.loglog(ns, ms, "o-", color="#1f4e79")
ax.set_xlabel("Number of alternatives $n$")
ax.set_ylabel("IT2-TOPSIS time (ms)")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_scale.pdf")
fig.savefig(FIG / "fig_scale.png", dpi=160)
print("wrote figs")
