"""Revised experiment (v2) for windowed IT2-TOPSIS on C-MAPSS FD001.

Changes against v1 (it2_stream_mcdm.py), following the panel review:
  * lower membership fixed: distance from 0 to the z-interval, not min(|z_lo|,|z_hi|)
  * 14 informative sensors (sensor 15, BPR, restored)
  * per-engine healthy reference with pooled within-engine sigma (pooled v1 kept as ablation)
  * FOU built from the detrended window residual std, so drift is not read as noise
  * absolute |z| map as in v1 (fixed); a directional map (sign of each sensor's trend) as ablation
  * baselines: interval TOPSIS (Jahanshahloo et al.), Hotelling T2, PCA-SPE, Isolation
    Forest, mean z, ridge health index (held-out test file only)
  * engine-block bootstrap for paired differences, chance level for Precision@10,
    early/late split for every method, sensitivity sweep, fault injection, packet loss,
    numerical check of the delay bound, record-at-a-time streaming replay.
Everything runs on one CPU. Output: results/metrics_v2.json and two figures.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
FIG = ROOT / "paper" / "figs"

SENSORS = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
NAMES = ["T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi", "NRf", "NRc", "BPR",
         "htBleed", "W31", "W32"]
SENSORS_V1 = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 21, 24, 25]
W, H, ZSTAR, K = 20, 30, 3.0, 1.0
T_START, T_STEP, MIN_N = 80, 10, 15
ALERT, TOPK, EPS = 30, 10, 1e-12
B_BOOT = 300


# ----------------------------------------------------------------------------- data
def load(name):
    a = np.loadtxt(DATA / name)
    return {int(u): a[a[:, 0] == u] for u in np.unique(a[:, 0])}


def detrended_std(X, cyc=None):
    """Population std of residuals after a least-squares line in time, per column."""
    n = len(X)
    tt = np.arange(n, dtype=float) if cyc is None else cyc.astype(float)
    tt = tt - tt.mean()
    stt = (tt ** 2).sum()
    Xc = X - X.mean(0)
    slope = (tt @ Xc) / stt if stt > 0 else np.zeros(X.shape[1])
    return (Xc - np.outer(tt, slope)).std(0)


class Ref:
    """Healthy reference: per-engine means and pooled within-engine sigma (or pooled v1)."""

    def __init__(self, eng, cols, h, pooled=False):
        self.cols, self.h, self.pooled = cols, h, pooled
        pre = {u: a[:h][:, cols] for u, a in eng.items()}
        if pooled:
            X = np.vstack(list(pre.values()))
            self.mu_all, self.sig = X.mean(0), X.std(0)
        else:
            self.mu = {u: p.mean(0) for u, p in pre.items()}
            self.sig = np.vstack([p - p.mean(0) for p in pre.values()]).std(0)
        self.sig = np.where(self.sig < 1e-9, 1.0, self.sig)

    def mean_of(self, u, a=None):
        if self.pooled:
            return self.mu_all
        if u in self.mu:
            return self.mu[u]
        return a[:self.h][:, self.cols].mean(0)


def trend_sign(eng, cols):
    """Direction of degradation per sensor from correlation with cycle (no RUL used)."""
    A = np.vstack(list(eng.values()))
    return np.sign([np.corrcoef(A[:, 1], A[:, c])[0, 1] for c in cols])


def window(a, t, w, cols):
    """Rows of cycles t-w+1..t (cycles are contiguous from 1)."""
    if t < w or t > len(a):
        return None
    return a[t - w:t][:, cols]


def batch(eng, ref, t_of, w, cols, rul_end=None):
    """Window statistics for the engines live at their evaluation cycle."""
    ids, M, Sr, Sd, R = [], [], [], [], []
    for u, a in eng.items():
        t = t_of(u, a)
        if t is None:
            continue
        X = window(a, t, w, cols)
        if X is None:
            continue
        mu = ref.mean_of(u, a)
        ids.append(u)
        M.append((X.mean(0) - mu) / ref.sig)
        Sr.append(X.std(0) / ref.sig)
        Sd.append(detrended_std(X) / ref.sig)
        R.append(len(a) - t + (0 if rul_end is None else rul_end[u]))
    return dict(ids=np.array(ids), z=np.array(M), sr=np.array(Sr), sd=np.array(Sd),
                rul=np.array(R, float))


# ----------------------------------------------------------------------------- IT2 map
def it2_map(z, s, sign=None, k=K, zstar=ZSTAR, v1=False):
    lo, hi = z - k * s, z + k * s
    if v1:  # original formula, kept to document the bug
        return (np.clip(np.minimum(abs(lo), abs(hi)) / zstar, 0, 1),
                np.clip(np.maximum(abs(lo), abs(hi)) / zstar, 0, 1))
    if sign is not None:  # directional: urgency only in the degradation direction
        a, b = np.sort(np.stack([sign * lo, sign * hi]), axis=0)
        return np.clip(np.maximum(a, 0) / zstar, 0, 1), np.clip(np.maximum(b, 0) / zstar, 0, 1)
    dist0 = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(abs(lo), abs(hi)))
    return np.clip(dist0 / zstar, 0, 1), np.clip(np.maximum(abs(lo), abs(hi)) / zstar, 0, 1)


def topsis_it2(L, U, w):
    Lw, Uw = L * w, U * w
    dp = np.sqrt(0.5 * ((Lw - Lw.max(0)) ** 2 + (Uw - Uw.max(0)) ** 2).sum(1))
    dn = np.sqrt(0.5 * ((Lw - Lw.min(0)) ** 2 + (Uw - Uw.min(0)) ** 2).sum(1))
    return dn / (dp + dn + EPS), dp, dn


def topsis_interval(L, U, w):
    """Jahanshahloo et al. (2006) interval TOPSIS, benefit criteria."""
    Lw, Uw = L * w, U * w
    ap, an = Uw.max(0), Lw.min(0)
    dp = np.sqrt(((Lw - ap) ** 2 + (Uw - ap) ** 2).sum(1))
    dn = np.sqrt(((Lw - an) ** 2 + (Uw - an) ** 2).sum(1))
    return dn / (dp + dn + EPS)


def topsis_t1(Mu, w):
    V = Mu * w
    dp = np.sqrt(((V - V.max(0)) ** 2).sum(1))
    dn = np.sqrt(((V - V.min(0)) ** 2).sum(1))
    return dn / (dp + dn + EPS)


# ----------------------------------------------------------------------------- metrics
def ndcg(scores, rul, k=TOPK):
    g = 1.0 / (1.0 + rul)
    k = min(k, len(g))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (g[np.argsort(-scores)][:k] * disc).sum()
    idcg = (np.sort(g)[::-1][:k] * disc).sum()
    return float(dcg / idcg)


def evaluate(scores, rul):
    pos = rul <= ALERT
    top = np.argsort(-scores)[:TOPK]
    return dict(rho=float(spearmanr(scores, -rul)[0]),
                p10=float(pos[top].mean()) if pos.any() else None,
                chance=float(pos.mean()), ndcg=ndcg(scores, rul))


def summarise(per_batch):
    out = {}
    for key in ("rho", "p10", "chance", "ndcg"):
        v = np.array([b[key] for b in per_batch if b[key] is not None], float)
        out[key] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, int(len(v))]
    return out


# ----------------------------------------------------------------------------- scorers
class Scorers:
    def __init__(self, eng, ref, sign, w_len, cols):
        m = len(cols)
        self.w = np.ones(m) / m
        self.sign = sign
        Z = []  # healthy window means in z units, windows fully inside the prefix
        for u, a in eng.items():
            mu = ref.mean_of(u, a)
            for t in range(w_len, ref.h + 1):
                Z.append((window(a, t, w_len, cols).mean(0) - mu) / ref.sig)
        Z = np.array(Z)
        self.cov_inv = np.linalg.pinv(np.cov(Z.T) + 1e-6 * np.eye(m))
        self.zbar = Z.mean(0)
        self.pca = PCA(n_components=0.9, svd_solver="full").fit(Z)
        # Isolation Forest is fitted on single healthy samples (as in v1), which gives it far
        # more training points than the healthy window means
        Zs = np.vstack([(a[:ref.h][:, cols] - ref.mean_of(u, a)) / ref.sig for u, a in eng.items()])
        self.iforest = IsolationForest(n_estimators=200, random_state=0).fit(Zs)

    def all(self, b):
        z, sd, sr, w, sg = b["z"], b["sd"], b["sr"], self.w, self.sign
        L, U = it2_map(z, sd, sg)
        La, Ua = it2_map(z, sd)
        Lr, Ur = it2_map(z, sr)
        Lv, Uv = it2_map(z, sr, v1=True)
        d = z - self.zbar
        rec = self.pca.inverse_transform(self.pca.transform(z))
        return {
            "IT2-TOPSIS (ours)": topsis_it2(La, Ua, w)[0],
            "IT2, directional map": topsis_it2(L, U, w)[0],
            "IT2, raw std FOU": topsis_it2(Lr, Ur, w)[0],
            "IT2 v1 (original)": topsis_it2(Lv, Uv, w)[0],
            "Interval TOPSIS": topsis_interval(La, Ua, w),
            "Type-1 TOPSIS": topsis_t1(np.clip(np.abs(z) / ZSTAR, 0, 1), w),
            "Mean directional z": np.maximum(sg * z, 0).mean(1),
            "Mean |z|": np.abs(z).mean(1),
            "Hotelling T2": np.einsum("ij,jk,ik->i", d, self.cov_inv, d),
            "PCA-SPE": ((z - rec) ** 2).sum(1),
            "Isolation Forest": -self.iforest.score_samples(z),
        }


def train_times(eng, w):
    tmax = max(len(a) for a in eng.values())
    ts = []
    for t in range(T_START, tmax + 1, T_STEP):
        if sum(len(a) >= t for a in eng.values()) >= MIN_N:
            ts.append(t)
    return ts


def run_train(eng, ref, sign, w_len=W, cols=SENSORS, zstar=ZSTAR, k=K, only_ours=False):
    sc = Scorers(eng, ref, sign, w_len, cols)
    batches, res = [], {}
    for t in train_times(eng, w_len):
        b = batch(eng, ref, lambda u, a, t=t: t if len(a) >= t else None, w_len, cols)
        b["t"] = t
        if only_ours:
            L, U = it2_map(b["z"], b["sd"], k=k, zstar=zstar)
            s = {"IT2-TOPSIS (ours)": topsis_it2(L, U, sc.w)[0]}
        else:
            s = sc.all(b)
        b["scores"] = s
        batches.append(b)
        for name, v in s.items():
            res.setdefault(name, []).append(evaluate(v, b["rul"]))
    return batches, res


# ----------------------------------------------------------------------------- bootstrap
def block_bootstrap(batches, methods, ref_name, rng):
    """Resample engines with replacement (paired across methods); mean Spearman over batches."""
    all_ids = np.unique(np.concatenate([b["ids"] for b in batches]))
    diffs = {m: [] for m in methods if m != ref_name}
    for _ in range(B_BOOT):
        cnt = dict(zip(all_ids, rng.multinomial(len(all_ids), np.ones(len(all_ids)) / len(all_ids))))
        rho = {m: [] for m in methods}
        for b in batches:
            rep = np.repeat(np.arange(len(b["ids"])), [cnt[u] for u in b["ids"]])
            if len(rep) < 5:
                continue
            for m in methods:
                rho[m].append(spearmanr(b["scores"][m][rep], -b["rul"][rep])[0])
        base = np.nanmean(rho[ref_name])
        for m in diffs:
            diffs[m].append(base - np.nanmean(rho[m]))
    out = {}
    for m, v in diffs.items():
        v = np.array(v)
        out[m] = dict(diff=float(v.mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                      p_two_sided=float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))))
    return out


# ----------------------------------------------------------------------------- experiments
def held_out_test(train, ref_tr, sign):
    """Rank the 100 test engines at their last observed cycle (standard C-MAPSS setting)."""
    test = load("test_FD001.txt")
    rul_end = dict(zip(range(1, 101), np.loadtxt(DATA / "RUL_FD001.txt").astype(int)))
    ref_te = Ref(test, SENSORS, H)
    ref_te.sig = ref_tr.sig  # sigma from training engines only
    b = batch(test, ref_te, lambda u, a: len(a), W, SENSORS, rul_end)
    sc = Scorers(train, ref_tr, sign, W, SENSORS)
    s = sc.all(b)
    # supervised reference: ridge health index trained on training windows, capped RUL
    X, y = [], []
    for u, a in train.items():
        mu = ref_tr.mean_of(u, a)
        for t in range(W, len(a) + 1, 5):
            X.append((window(a, t, W, SENSORS).mean(0) - mu) / ref_tr.sig)
            y.append(min(len(a) - t, 125))
    s["Ridge health index (supervised)"] = -Ridge(alpha=1.0).fit(X, y).predict(b["z"])
    res = {m: evaluate(v, b["rul"]) for m, v in s.items()}
    rng = np.random.default_rng(7)
    boot = {}
    for m, v in s.items():
        r = [spearmanr(v[i], -b["rul"][i])[0] for i in
             (rng.integers(0, len(v), len(v)) for _ in range(1000))]
        boot[m] = [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    return dict(n=len(b["ids"]), metrics=res, rho_ci=boot)


def sensitivity(train, sign):
    out = {}
    for name, grid in (("W", (10, 20, 30)), ("H", (20, 30, 40)), ("zstar", (2.0, 3.0, 4.0)),
                       ("k", (0.5, 1.0, 2.0))):
        out[name] = {}
        for v in grid:
            w_len = v if name == "W" else W
            h = v if name == "H" else H
            ref = Ref(train, SENSORS, h)
            _, r = run_train(train, ref, sign, w_len=w_len, zstar=v if name == "zstar" else ZSTAR,
                             k=v if name == "k" else K, only_ours=True)
            out[name][str(v)] = summarise(r["IT2-TOPSIS (ours)"])["rho"][0]
    return out


def fou_cell(z, s, k=K, zstar=ZSTAR):
    L, U = it2_map(z, s, k=k, zstar=zstar)
    return U - L


def fault_injection(train, ref, rng):
    """Sensor faults injected into raw windows; thresholds at 1% FPR on engines 1-50."""
    kinds = ("noise burst", "bias step", "spike")
    det_names = ("FOU width, detrended (ours)", "FOU width, raw std (v1)", "Rolling std ratio",
                 "Bootstrap CI width of mean")
    stats = {d: {"clean_cal": [], "clean_eval": [], "clean_eval_late": [],
                 **{kd: [] for kd in kinds}} for d in det_names}

    def detectors(X, mu):
        z = (X.mean(0) - mu) / ref.sig
        sr, sd = X.std(0) / ref.sig, detrended_std(X) / ref.sig
        idx = rng.integers(0, len(X), (200, len(X)))
        bm = X[idx].mean(1)
        ci = (np.percentile(bm, 97.5, 0) - np.percentile(bm, 2.5, 0)) / ref.sig
        return {det_names[0]: fou_cell(z, sd), det_names[1]: fou_cell(z, sr),
                det_names[2]: sr, det_names[3]: ci}

    for t in train_times(train, W):
        for u, a in train.items():
            X = window(a, t, W, SENSORS)
            if X is None:
                continue
            mu = ref.mean_of(u, a)
            late = (len(a) - t) <= 80
            if u <= 50:
                for d, v in detectors(X, mu).items():
                    stats[d]["clean_cal"].extend(v)
                continue
            j = rng.integers(len(SENSORS))
            kind = kinds[rng.integers(3)]
            Xf = X.copy()
            sg = ref.sig[j]
            if kind == "noise burst":
                Xf[-10:, j] += rng.normal(0, 2 * sg, 10)
            elif kind == "bias step":
                Xf[-10:, j] += 3 * sg * rng.choice([-1, 1])
            else:
                Xf[rng.integers(W), j] += 8 * sg * rng.choice([-1, 1])
            for d, v in detectors(Xf, mu).items():
                stats[d][kind].append(v[j])
                clean = np.delete(v, j)
                stats[d]["clean_eval"].extend(clean)
                if late:
                    stats[d]["clean_eval_late"].extend(clean)
    out = {}
    for d, s in stats.items():
        thr = float(np.percentile(s["clean_cal"], 99))
        out[d] = {"threshold": thr,
                  "fpr_eval": float(np.mean(np.array(s["clean_eval"]) > thr)),
                  "fpr_late_life": float(np.mean(np.array(s["clean_eval_late"]) > thr)),
                  **{f"tpr_{kd}": float(np.mean(np.array(s[kd]) > thr)) for kd in kinds}}
    return out


def packet_loss(train, ref, sign, batches_clean):
    """Drop whole rows (packet loss) and add sensor noise; window needs >= W/2 rows."""
    sc = Scorers(train, ref, sign, W, SENSORS)
    keep = ("IT2-TOPSIS (ours)", "Type-1 TOPSIS", "Mean directional z", "Hotelling T2",
            "Isolation Forest")
    out = {}
    for p in (0.1, 0.2, 0.3):
        taus, rhos, dropped = {m: [] for m in keep}, {m: [] for m in keep}, []
        for seed in (11, 23, 41):
            rng = np.random.default_rng(seed)
            for bc in batches_clean:
                t = bc["t"]
                ids, M, Sr, Sd, R = [], [], [], [], []
                for u, a in train.items():
                    if len(a) < t:
                        continue
                    rows = a[t - W:t]
                    rows = rows[rng.random(W) > p]
                    if len(rows) < W // 2:
                        continue
                    X = rows[:, SENSORS] + rng.normal(0, 0.1 * ref.sig, (len(rows), len(SENSORS)))
                    mu = ref.mean_of(u, a)
                    ids.append(u)
                    M.append((X.mean(0) - mu) / ref.sig)
                    Sr.append(X.std(0) / ref.sig)
                    Sd.append(detrended_std(X, rows[:, 1]) / ref.sig)
                    R.append(len(a) - t)
                b = dict(ids=np.array(ids), z=np.array(M), sr=np.array(Sr), sd=np.array(Sd),
                         rul=np.array(R, float))
                dropped.append(1 - len(ids) / len(bc["ids"]))
                s = sc.all(b)
                pos = np.searchsorted(bc["ids"], b["ids"])
                for m in keep:
                    taus[m].append(kendalltau(bc["scores"][m][pos], s[m])[0])
                    rhos[m].append(spearmanr(s[m], -b["rul"])[0])
        out[str(p)] = {"engines_dropped": float(np.mean(dropped)),
                       **{m: {"kendall_vs_clean": float(np.mean(taus[m])),
                              "rho": float(np.mean(rhos[m]))} for m in keep}}
    return out


def delay_bound(train, ref, sign):
    """Observed max |C(t)-C(t-tau)| versus the Proposition 3 bound with measured delta, gamma."""
    m = len(SENSORS)
    w = np.ones(m) / m
    out = {}
    for tau in (1, 2, 5):
        obs, bnd = [], []
        for t in train_times(train, W):
            f0 = lambda u, a: t if len(a) >= t else None
            b0 = batch(train, ref, f0, W, SENSORS)
            b1 = batch(train, ref, lambda u, a: t - tau if len(a) >= t else None, W, SENSORS)
            L0, U0 = it2_map(b0["z"], b0["sd"])
            L1, U1 = it2_map(b1["z"], b1["sd"])
            c0, dp0, dn0 = topsis_it2(L0, U0, w)
            c1, dp1, dn1 = topsis_it2(L1, U1, w)
            delta = max(abs(L0 - L1).max(), abs(U0 - U1).max()) / m
            gamma = min((dp0 + dn0).min(), (dp1 + dn1).min())
            obs.append(abs(c0 - c1).max())
            bnd.append(2 * np.sqrt(m) * delta / gamma)
        out[str(tau)] = {"observed_max_dC": float(np.mean(obs)), "bound": float(np.mean(bnd)),
                         "bound_ge_observed_all": bool(np.all(np.array(bnd) >= np.array(obs)))}
    return out


class Sliding:
    """O(m) per arrival: sliding sums for mean, std and detrended std over the last W rows."""

    def __init__(self, m, w, centre):
        self.w, self.c = w, centre
        self.buf = np.zeros((w, m))
        self.n = self.head = 0
        self.s1 = np.zeros(m)
        self.s2 = np.zeros(m)
        self.sj = np.zeros(m)

    def push(self, x):
        x = x - self.c
        if self.n < self.w:
            self.sj += self.n * x
            self.buf[self.n] = x
            self.n += 1
            self.s1 += x
            self.s2 += x * x
            return
        old = self.buf[self.head]
        self.sj += -(self.s1 - old) + (self.w - 1) * x - 0 * old
        self.s1 += x - old
        self.s2 += x * x - old * old
        self.buf[self.head] = x
        self.head = (self.head + 1) % self.w

    def stats(self):
        w = self.w
        mean = self.s1 / w
        sxx = np.maximum(self.s2 - self.s1 ** 2 / w, 0)
        tbar, stt = (w - 1) / 2, w * (w * w - 1) / 12
        sxt = self.sj - tbar * self.s1
        return mean + self.c, np.sqrt(sxx / w), np.sqrt(np.maximum(sxx - sxt ** 2 / stt, 0) / w)


def stream_replay(train, ref, sign):
    """Replay the training file row by row in cycle order; rank live engines at each cycle end."""
    rows = np.vstack(list(train.values()))
    rows = rows[np.lexsort((rows[:, 0], rows[:, 1]))]
    centre = rows[:, SENSORS].mean(0)
    w = np.ones(len(SENSORS)) / len(SENSORS)
    state, lat, max_err = {}, [], 0.0
    t0 = time.perf_counter()
    cur = rows[0, 1]
    live = []

    def rank(live, t):
        nonlocal max_err
        z, sd = [], []
        for u in live:
            mean, _, sdet = state[u].stats()
            z.append((mean - ref.mean_of(u)) / ref.sig)
            sd.append(sdet / ref.sig)
        L, U = it2_map(np.array(z), np.array(sd))
        if t % 50 == 0:  # check incremental statistics against a batch recompute
            X = window(train[live[0]], int(t), W, SENSORS)
            max_err = max(max_err, float(abs(detrended_std(X) / ref.sig - sd[0]).max()))
        return topsis_it2(L, U, w)[0]

    for r in rows:
        if r[1] != cur:
            ready = [u for u in live if state[u].n == W and cur >= H]
            if len(ready) >= 2:
                s = time.perf_counter()
                rank(ready, cur)
                lat.append((time.perf_counter() - s) * 1e3)
            cur, live = r[1], []
        u = int(r[0])
        state.setdefault(u, Sliding(len(SENSORS), W, centre)).push(r[SENSORS])
        live.append(u)
    total = time.perf_counter() - t0
    return {"rows": int(len(rows)), "seconds": float(total), "rows_per_s": float(len(rows) / total),
            "rank_ms_p50": float(np.percentile(lat, 50)), "rank_ms_p99": float(np.percentile(lat, 99)),
            "cycles_ranked": len(lat), "max_abs_err_vs_batch": max_err}


def vectorised_scale(rng):
    out = {}
    for n in (1000, 10000, 100000):
        L = rng.random((n, 14))
        U = np.minimum(1, L + 0.2 * rng.random((n, 14)))
        s = time.perf_counter()
        for _ in range(10):
            topsis_it2(L, U, np.ones(14) / 14)
        out[str(n)] = (time.perf_counter() - s) * 100
    return out


# ----------------------------------------------------------------------------- main
def main():
    t_all = time.perf_counter()
    train = load("train_FD001.txt")
    sign = trend_sign(train, SENSORS)
    ref = Ref(train, SENSORS, H)
    batches, res = run_train(train, ref, sign)
    out = {"setup": dict(W=W, H=H, zstar=ZSTAR, k=K, sensors=NAMES, sign=sign.tolist(),
                         n_batches=len(batches), batch_sizes=[len(b["ids"]) for b in batches])}

    # pooled-reference ablation (v1 reference, fixed membership)
    ref_pool = Ref(train, SENSORS, H, pooled=True)
    _, rp = run_train(train, ref_pool, sign, only_ours=True)
    res["IT2, pooled reference"] = rp["IT2-TOPSIS (ours)"]

    # zero-straddling cells under the v1 setting
    rv1 = Ref(train, SENSORS_V1, H, pooled=True)
    st = tot = 0
    for b in run_train(train, rv1, trend_sign(train, SENSORS_V1), cols=SENSORS_V1, only_ours=True)[0]:
        lo, hi = b["z"] - b["sr"], b["z"] + b["sr"]
        st += int(((lo < 0) & (hi > 0)).sum())
        tot += lo.size
    out["v1_zero_straddling_cells"] = [st, tot]

    out["train"] = {m: summarise(r) for m, r in res.items()}
    late = [i for i, b in enumerate(batches) if b["rul"].mean() <= 80]
    early = [i for i in range(len(batches)) if i not in late]
    out["split"] = {m: {"early": summarise([r[i] for i in early]),
                        "late": summarise([r[i] for i in late])} for m, r in res.items()
                    if m != "IT2, pooled reference"}
    out["per_batch_rho"] = {m: [r_["rho"] for r_ in r] for m, r in res.items()}
    out["batch_t"] = [b["t"] for b in batches]
    print("main done", round(time.perf_counter() - t_all, 1), "s", flush=True)

    rng = np.random.default_rng(2026)
    out["bootstrap_vs_ours"] = block_bootstrap(batches, list(batches[0]["scores"]),
                                               "IT2-TOPSIS (ours)", rng)
    print("bootstrap done", round(time.perf_counter() - t_all, 1), "s", flush=True)
    out["test_last_cycle"] = held_out_test(train, ref, sign)
    out["sensitivity"] = sensitivity(train, sign)
    print("sensitivity done", round(time.perf_counter() - t_all, 1), "s", flush=True)
    out["fault_injection"] = fault_injection(train, ref, rng)
    out["packet_loss"] = packet_loss(train, ref, sign, batches)
    print("robustness done", round(time.perf_counter() - t_all, 1), "s", flush=True)
    out["delay_bound"] = delay_bound(train, ref, sign)
    out["stream_replay"] = stream_replay(train, ref, sign)
    out["vectorised_ms"] = vectorised_scale(rng)
    out["runtime_s"] = time.perf_counter() - t_all
    (OUT / "metrics_v2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    figures(out)
    print("all done", round(out["runtime_s"], 1), "s")


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    ts = out["batch_t"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for m, st in (("IT2-TOPSIS (ours)", "-o"), ("IT2 v1 (original)", ":x"), ("Hotelling T2", "--s"),
                  ("Mean directional z", "-.^"), ("Isolation Forest", ":d")):
        ax.plot(ts, out["per_batch_rho"][m], st, ms=3, lw=1, label=m)
    ax.set_xlabel("cycle t")
    ax.set_ylabel("Spearman rho vs. -RUL")
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_rho_vs_t_v2.pdf")
    fig.savefig(FIG / "fig_rho_vs_t_v2.png", dpi=200)

    fig, axs = plt.subplots(1, 4, figsize=(3.5, 1.6), sharey=True)
    for ax, (p, vals) in zip(axs, out["sensitivity"].items()):
        ax.bar(list(vals), list(vals.values()), color="0.5")
        ax.set_title({"zstar": "z*"}.get(p, p))
        ax.tick_params(labelsize=6)
    axs[0].set_ylabel("mean rho")
    axs[0].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sensitivity_v2.pdf")
    fig.savefig(FIG / "fig_sensitivity_v2.png", dpi=200)


if __name__ == "__main__":
    main()
