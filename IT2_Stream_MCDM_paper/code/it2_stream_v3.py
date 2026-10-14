"""Round 2 experiment (v3): all four C-MAPSS subsets, fair FOU analysis, stricter statistics.

Reuses the model, scorers and streaming code of it2_stream_v2.py. Additions:
  * FD002 and FD004 normalised per operating condition (k-means on the three settings,
    fitted on the training file), FD003 and FD004 add a second fault mode
  * engine-block bootstrap with 2000 replicates and Holm correction; pooled-reference row included
  * test set: age-only baseline and R-precision (precision at k = number of engines with RUL <= 30)
  * FOU analysis: AUC per fault type, detection at a matched realised false-alarm rate,
    mean-plus-spread and upper-membership detectors, a zero-mean variance fault
  * delay bound: Proposition 3 with measured delta and Proposition 4 with measured D^m, D^r
Output: results/metrics_v3.json and figures paper/figs/*_v3.*
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

import it2_stream_v2 as V2
from it2_stream_v2 import (DATA, FIG, OUT, SENSORS, H, K, W, ZSTAR, Ref, Scorers, batch,
                           detrended_std, fou_cell, it2_map, ndcg, topsis_it2, train_times,
                           trend_sign, window)

ALERT = 30
B_BOOT = 2000
OURS = "IT2-TOPSIS (ours)"


# ----------------------------------------------------------------------------- data
def load_set(fd):
    """Train/test dicts per engine; multi-condition sets normalised per operating condition."""
    tr = np.loadtxt(DATA / f"train_{fd}.txt")
    te = np.loadtxt(DATA / f"test_{fd}.txt")
    rul = np.loadtxt(DATA / f"RUL_{fd}.txt").astype(int)
    if fd in ("FD002", "FD004"):
        km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(tr[:, 2:5])
        ctr, cte = km.labels_, km.predict(te[:, 2:5])
        for c in range(6):
            X = tr[ctr == c][:, SENSORS]
            mu, sd = X.mean(0), X.std(0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            tr[np.ix_(ctr == c, SENSORS)] = (tr[ctr == c][:, SENSORS] - mu) / sd
            te[np.ix_(cte == c, SENSORS)] = (te[cte == c][:, SENSORS] - mu) / sd
    split = lambda a: {int(u): a[a[:, 0] == u] for u in np.unique(a[:, 0])}
    test = split(te)
    return split(tr), test, dict(zip(sorted(test), rul))


# ----------------------------------------------------------------------------- metrics
def evaluate(scores, rul):
    pos = rul <= ALERT
    npos = int(pos.sum())
    top = np.argsort(-scores)
    return dict(rho=float(spearmanr(scores, -rul)[0]),
                p10=float(pos[top[:10]].mean()) if npos else None,
                chance=float(pos.mean()),
                rprec=float(pos[top[:npos]].mean()) if npos else None,
                ndcg=ndcg(scores, rul))


def summarise(rows):
    out = {}
    for key in ("rho", "p10", "chance", "rprec", "ndcg"):
        v = np.array([r[key] for r in rows if r[key] is not None], float)
        out[key] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, int(len(v))] if len(v) else None
    return out


def holm(p):
    names = sorted(p, key=p.get)
    adj, run = {}, 0.0
    for i, n in enumerate(names):
        run = max(run, min(1.0, (len(names) - i) * p[n]))
        adj[n] = run
    return adj


def corr_ranked(ra, rb):
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return (ra * rb).sum() / den if den > 0 else np.nan


def block_bootstrap(batches, methods, rng):
    ids = np.unique(np.concatenate([b["ids"] for b in batches]))
    diffs = {m: [] for m in methods if m != OURS}
    for _ in range(B_BOOT):
        cnt = dict(zip(ids, rng.multinomial(len(ids), np.ones(len(ids)) / len(ids))))
        rho = {m: [] for m in methods}
        for b in batches:
            rep = np.repeat(np.arange(len(b["ids"])), [cnt[u] for u in b["ids"]])
            if len(rep) < 5:
                continue
            rr = rankdata(-b["rul"][rep])
            for m in methods:
                rho[m].append(corr_ranked(rankdata(b["scores"][m][rep]), rr))
        base = np.nanmean(rho[OURS])
        for m in diffs:
            diffs[m].append(base - np.nanmean(rho[m]))
    out, p = {}, {}
    for m, v in diffs.items():
        v = np.array(v)
        p[m] = max(1.0 / B_BOOT, min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))
        out[m] = dict(diff=float(v.mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                      p=float(p[m]))
    for m, a in holm(p).items():
        out[m]["p_holm"] = float(a)
    return out


# ----------------------------------------------------------------------------- per dataset
def run_dataset(fd, rng, full=False):
    t0 = time.perf_counter()
    train, test, rul_end = load_set(fd)
    sign = trend_sign(train, SENSORS)
    ref, ref_pool = Ref(train, SENSORS, H), Ref(train, SENSORS, H, pooled=True)
    sc = Scorers(train, ref, sign, W, SENSORS)
    w = np.ones(len(SENSORS)) / len(SENSORS)
    batches, res = [], {}
    for t in train_times(train, W):
        alive = lambda u, a, t=t: t if len(a) >= t else None
        b = batch(train, ref, alive, W, SENSORS)
        bp = batch(train, ref_pool, alive, W, SENSORS)
        b["t"] = t
        b["scores"] = sc.all(b)
        La, Ua = it2_map(bp["z"], bp["sd"])
        b["scores"]["IT2, pooled reference"] = topsis_it2(La, Ua, w)[0]
        batches.append(b)
        for m, v in b["scores"].items():
            res.setdefault(m, []).append(evaluate(v, b["rul"]))
    late = [i for i, b in enumerate(batches) if b["rul"].mean() <= 80]
    early = [i for i in range(len(batches)) if i not in late]
    out = {"n_train_engines": len(train), "n_test_engines": len(test),
           "batch_t": [b["t"] for b in batches], "batch_n": [len(b["ids"]) for b in batches],
           "train": {m: summarise(r) for m, r in res.items()},
           "split": {m: {"early": summarise([r[i] for i in early]) if early else None,
                         "late": summarise([r[i] for i in late])} for m, r in res.items()},
           "per_batch_rho": {m: [x["rho"] for x in r] for m, r in res.items()}}
    out["bootstrap_vs_ours"] = block_bootstrap(batches, list(batches[0]["scores"]), rng)

    # held-out test engines at their last cycle
    ref_te = Ref(test, SENSORS, H)
    ref_te.sig = ref.sig
    bt = batch(test, ref_te, lambda u, a: len(a), W, SENSORS, rul_end)
    s = sc.all(bt)
    X, y = [], []
    for u, a in train.items():
        mu = ref.mean_of(u, a)
        for t in range(W, len(a) + 1, 5):
            X.append((window(a, t, W, SENSORS).mean(0) - mu) / ref.sig)
            y.append(min(len(a) - t, 125))
    s["Ridge health index (supervised)"] = -Ridge(alpha=1.0).fit(X, y).predict(bt["z"])
    s["Age only (cycles observed)"] = np.array([len(test[u]) for u in bt["ids"]], float)
    boot = {}
    for m, v in s.items():
        r = [spearmanr(v[i], -bt["rul"][i])[0] for i in (rng.integers(0, len(v), len(v)) for _ in range(1000))]
        boot[m] = [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    out["test"] = {"n": len(bt["ids"]), "metrics": {m: evaluate(v, bt["rul"]) for m, v in s.items()},
                   "rho_ci": boot}
    out["fou"] = fou_analysis(train, ref, rng)
    out["delay"] = delay_bounds(train, ref)
    out["seconds"] = time.perf_counter() - t0
    if full:
        out["sensitivity"] = V2.sensitivity(train, sign)
        out["packet_loss"] = V2.packet_loss(train, ref, sign, batches)
        out["stream_replay"] = V2.stream_replay(train, ref, sign)
        out["vectorised_ms"] = V2.vectorised_scale(rng)
    print(fd, "done in", round(time.perf_counter() - t0, 1), "s", flush=True)
    return out


# ----------------------------------------------------------------------------- FOU analysis
KINDS = ("noise burst", "bias step", "spike", "variance increase (zero mean)")
DETECTORS = ("FOU width, detrended (ours)", "FOU width, raw std", "Upper membership",
             "Mean plus spread", "Rolling std ratio", "Detrended std ratio", "Bootstrap CI width")


def detectors(X, mu, ref, rng):
    z = (X.mean(0) - mu) / ref.sig
    sr, sd = X.std(0) / ref.sig, detrended_std(X) / ref.sig
    bm = X[rng.integers(0, len(X), (200, len(X)))].mean(1)
    ci = (np.percentile(bm, 97.5, 0) - np.percentile(bm, 2.5, 0)) / ref.sig
    return {DETECTORS[0]: fou_cell(z, sd), DETECTORS[1]: fou_cell(z, sr),
            DETECTORS[2]: it2_map(z, sd)[1], DETECTORS[3]: np.abs(z) + K * sr,
            DETECTORS[4]: sr, DETECTORS[5]: sd, DETECTORS[6]: ci}


def fou_analysis(train, ref, rng):
    """Faults in raw windows. Engines in the first half calibrate a 1% false-alarm threshold."""
    half = sorted(train)[len(train) // 2]
    cal = {d: [] for d in DETECTORS}
    clean = {d: [] for d in DETECTORS}
    fault = {d: {k: [] for k in KINDS} for d in DETECTORS}
    for t in train_times(train, W):
        for u, a in train.items():
            X = window(a, t, W, SENSORS)
            if X is None:
                continue
            mu = ref.mean_of(u, a)
            if u < half:
                for d, v in detectors(X, mu, ref, rng).items():
                    cal[d].extend(v)
                continue
            j = rng.integers(len(SENSORS))
            kind = KINDS[rng.integers(len(KINDS))]
            Xf, sg = X.copy(), ref.sig[j]
            if kind == "noise burst":
                Xf[-10:, j] += rng.normal(0, 2 * sg, 10)
            elif kind == "bias step":
                Xf[-10:, j] += 3 * sg * rng.choice([-1, 1])
            elif kind == "spike":
                Xf[rng.integers(W), j] += 8 * sg * rng.choice([-1, 1])
            else:
                e = rng.normal(0, 1.5 * sg, W)
                Xf[:, j] += e - e.mean()  # window mean unchanged
            for d, v in detectors(Xf, mu, ref, rng).items():
                fault[d][kind].append(v[j])
                clean[d].extend(np.delete(v, j))
    out = {}
    for d in DETECTORS:
        c = np.array(clean[d])
        thr = np.percentile(cal[d], 99)
        thr_eval = np.percentile(c, 99)
        r = {"fa_calibrated": float((c > thr).mean())}
        for k in KINDS:
            f = np.array(fault[d][k])
            yy, ss = np.r_[np.zeros(len(c)), np.ones(len(f))], np.r_[c, f]
            r[k] = {"auc": float(roc_auc_score(yy, ss)),
                    "pauc_fpr05": float(roc_auc_score(yy, ss, max_fpr=0.05)),
                    "tpr_calibrated": float((f > thr).mean()),
                    "tpr_at_1pct_realised": float((f > thr_eval).mean())}
        out[d] = r
    return out


# ----------------------------------------------------------------------------- delay bounds
def delay_bounds(train, ref):
    m = len(SENSORS)
    w = np.ones(m) / m
    out = {}
    for tau in (1, 2, 5):
        obs, b3, b4 = [], [], []
        for t in train_times(train, W):
            b0 = batch(train, ref, lambda u, a: t if len(a) >= t else None, W, SENSORS)
            b1 = batch(train, ref, lambda u, a: t - tau if len(a) >= t else None, W, SENSORS)
            L0, U0 = it2_map(b0["z"], b0["sd"])
            L1, U1 = it2_map(b1["z"], b1["sd"])
            c0, dp0, dn0 = topsis_it2(L0, U0, w)
            c1, dp1, dn1 = topsis_it2(L1, U1, w)
            gamma = (dp1 + dn1).min()
            delta3 = max(abs(L0 - L1).max(), abs(U0 - U1).max()) / m
            # Proposition 4: D^m, D^r in z units (already divided by sigma)
            dm = abs(b0["z"] - b1["z"]).max(0)
            dr = abs(b0["sd"] - b1["sd"]).max(0)
            delta4 = (w * (dm + K * dr) / ZSTAR).max()
            obs.append(abs(c0 - c1).max())
            b3.append(2 * np.sqrt(m) * delta3 / gamma)
            b4.append(2 * np.sqrt(m) * delta4 / gamma)
        obs, b3, b4 = map(np.array, (obs, b3, b4))
        out[str(tau)] = {"observed": float(obs.mean()), "bound_prop3": float(b3.mean()),
                         "bound_prop4": float(b4.mean()), "prop3_holds": bool((b3 >= obs).all()),
                         "prop4_holds": bool((b4 >= obs).all()), "ratio_prop3": float(b3.mean() / obs.mean()),
                         "ratio_prop4": float(b4.mean() / obs.mean())}
    return out


# ----------------------------------------------------------------------------- main
def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.4), sharey=True)
    styles = ((OURS, "-o"), ("IT2, earlier membership", ":x"), ("Hotelling T2", "--s"),
              ("Mean |z|", "-.^"), ("Isolation Forest", ":d"))
    for ax, (fd, r) in zip(axs.flat, res.items()):
        for m, st in styles:
            ax.plot(r["batch_t"], r["per_batch_rho"][m], st, ms=2, lw=0.8, label=m)
        ax.set_title(fd, fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=6)
    axs[1, 0].set_xlabel("cycle t")
    axs[1, 1].set_xlabel("cycle t")
    axs[0, 0].set_ylabel("Spearman rho")
    axs[1, 0].set_ylabel("Spearman rho")
    axs[0, 0].legend(fontsize=5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_rho_vs_t_v3.pdf")
    fig.savefig(FIG / "fig_rho_vs_t_v3.png", dpi=200)

    fig, axs = plt.subplots(1, 4, figsize=(3.5, 1.6), sharey=True)
    for ax, (p, vals) in zip(axs, res["FD001"]["sensitivity"].items()):
        ax.bar(list(vals), list(vals.values()), color="0.5")
        ax.set_title({"zstar": "z*"}.get(p, p))
        ax.tick_params(labelsize=6)
    axs[0].set_ylabel("mean rho")
    axs[0].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sensitivity_v3.pdf")
    fig.savefig(FIG / "fig_sensitivity_v3.png", dpi=200)


def main():
    # the earlier-membership ablation is labelled clearly in v3 outputs
    rng = np.random.default_rng(2026)
    res = {}
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        res[fd] = run_dataset(fd, rng, full=(fd == "FD001"))
    for r in res.values():
        for key in ("train", "split", "per_batch_rho", "bootstrap_vs_ours"):
            d = r[key]
            if "IT2 v1 (original)" in d:
                d["IT2, earlier membership"] = d.pop("IT2 v1 (original)")
        r["test"]["metrics"]["IT2, earlier membership"] = r["test"]["metrics"].pop("IT2 v1 (original)")
        r["test"]["rho_ci"]["IT2, earlier membership"] = r["test"]["rho_ci"].pop("IT2 v1 (original)")
    (OUT / "metrics_v3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    figures(res)
    print("all done")


if __name__ == "__main__":
    main()
