"""Stream generators: controlled Gaussian, Digits (exact-validity protocol), Covertype, Electricity."""
from __future__ import annotations

import gzip
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from tssd_core import Source

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

# ============================================================================ Gaussian
G_D, G_K = 10, 3
G_PI = np.full(G_K, 1 / G_K)
G_SHIFT = np.full(G_D, 2.0)
G_LABEL_Q = np.array([0.70, 0.20, 0.10])


def g_means():
    return np.random.default_rng(0).normal(size=(G_K, G_D))


def g_draw(rng, n, means, prior=G_PI):
    y = rng.choice(G_K, size=n, p=prior)
    return means[y] + rng.normal(size=(n, G_D)), y


def g_bayes(rng, x, means):
    post = np.exp(x @ means.T - 0.5 * (means ** 2).sum(1))
    post /= post.sum(1, keepdims=True)
    u = rng.random(len(x))
    return (u[:, None] > post.cumsum(1)).sum(1).clip(0, G_K - 1)


def g_source():
    means = g_means()
    x, y = g_draw(np.random.default_rng(2026), 2000, means)
    return means, Source(x, y, G_K, l2=1e-4)


def g_stream(seed, cond, means, T=400, t0=200, rate=0.05, cov_scale=1.0, label_gamma=1.0,
             concept_frac=1.0):
    """cond in null, covariate, label, concept, covariate_label, covariate_concept, label_concept."""
    rng = np.random.default_rng(seed)
    x = np.zeros((T, G_D)); y = np.zeros(T, int)
    x[:t0], y[:t0] = g_draw(rng, t0, means)
    n = T - t0
    q = (1 - label_gamma) * G_PI + label_gamma * G_LABEL_Q
    shift = cov_scale * G_SHIFT
    if n > 0:
        if cond == "null":
            x[t0:], y[t0:] = g_draw(rng, n, means)
        elif cond == "covariate":
            yl = rng.choice(G_K, size=n, p=G_PI)
            x[t0:] = means[yl] + shift + rng.normal(size=(n, G_D))
            y[t0:] = g_bayes(rng, x[t0:], means)
        elif cond == "label":
            x[t0:], y[t0:] = g_draw(rng, n, means, q)
        elif cond == "concept":
            x[t0:], yl = g_draw(rng, n, means)
            flip = rng.random(n) < concept_frac
            y[t0:] = np.where(flip, (yl + 1) % G_K, yl)
        elif cond == "covariate_label":
            yl = rng.choice(G_K, size=n, p=q)
            x[t0:] = means[yl] + shift + rng.normal(size=(n, G_D))
            y[t0:] = yl
        elif cond == "covariate_concept":
            yl = rng.choice(G_K, size=n, p=G_PI)
            x[t0:] = means[yl] + shift + rng.normal(size=(n, G_D))
            y[t0:] = (g_bayes(rng, x[t0:], means) + 1) % G_K
        elif cond == "label_concept":
            x[t0:], yl = g_draw(rng, n, means, q)
            y[t0:] = (yl + 1) % G_K
        else:
            raise ValueError(cond)
    xr, yr = g_draw(rng, T, means)
    r = rng.binomial(1, rate, size=T).astype(int)
    return dict(x_tgt=x, y_tgt=y, x_ref=xr, y_ref=yr, r=r, t0=t0, cond=cond)


# ============================================================================ Digits
def _sklearn_data_dir():
    import site
    for base in [site.getusersitepackages()] + site.getsitepackages():
        p = Path(base) / "sklearn" / "datasets" / "data" / "digits.csv.gz"
        if p.exists():
            return p
    raise FileNotFoundError("digits.csv.gz (ships with scikit-learn) not found")


def load_digits():
    with gzip.open(_sklearn_data_dir(), "rt") as f:
        arr = np.loadtxt(f, delimiter=",")
    return arr[:, :-1], arr[:, -1].astype(int)


class DigitsSetup:
    """Disjoint fit / reference / target pools; draws WITHOUT replacement (exact null validity)."""

    def __init__(self, seed=2026):
        X, y = load_digits()
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n_fit = 719
        fit, rest = idx[:n_fit], idx[n_fit:]
        half = len(rest) // 2
        mu, sd = X[fit].mean(0), X[fit].std(0) + 1.0
        Z = (X - mu) / sd
        self.K = 10
        self.src = Source(Z[fit], y[fit], 10, l2=1e-2)
        self.x_ref, self.y_ref = Z[rest[:half]], y[rest[:half]]
        self.x_tg, self.y_tg = Z[rest[half:]], y[rest[half:]]
        self.label_q = np.array([0.25, 0.25, 0.25] + [0.25 / 7] * 7)

    def stream(self, seed, cond, T=300, t0=150, rate=0.05):
        rng = np.random.default_rng(seed)
        ir = rng.permutation(len(self.y_ref))[:T]
        xr, yr = self.x_ref[ir], self.y_ref[ir]
        perm = rng.permutation(len(self.y_tg))
        pre, post_pool = perm[:t0], perm[t0:]
        x0, y0 = self.x_tg[pre], self.y_tg[pre]
        n = T - t0
        yp = self.y_tg[post_pool]
        if cond in ("label", "label_concept", "covariate_label"):
            # prevalence change by class-weighted sampling without replacement
            w = self.label_q[yp] / np.bincount(yp, minlength=10)[yp]
            sel = rng.choice(len(post_pool), n, replace=False, p=w / w.sum())
        else:
            sel = np.arange(n)
        x1, y1 = self.x_tg[post_pool[sel]].copy(), self.y_tg[post_pool[sel]].copy()
        if cond in ("covariate", "covariate_concept", "covariate_label"):
            x1 = 1.35 * x1 + rng.normal(0, 0.45, size=x1.shape)
        if cond in ("concept", "covariate_concept", "label_concept"):
            y1 = (y1 + 1) % 10
        x = np.vstack([x0, x1]); y = np.concatenate([y0, y1])
        r = rng.binomial(1, rate, size=T).astype(int)
        return dict(x_tgt=x, y_tgt=y, x_ref=xr, y_ref=yr, r=r, t0=t0, cond=cond)


# ============================================================================ real data downloads
def _download(url, path):
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def load_covertype():
    cache = DATA / "covtype.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    z = _download("https://archive.ics.uci.edu/static/public/31/covertype.zip", DATA / "covertype.zip")
    with zipfile.ZipFile(z) as zf:
        name = [n for n in zf.namelist() if n.endswith(".gz")][0]
        raw = gzip.decompress(zf.read(name))
    arr = np.loadtxt(io.BytesIO(raw), delimiter=",", dtype=np.float32)
    X, y = arr[:, :-1], arr[:, -1].astype(int) - 1
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def load_electricity():
    cache = DATA / "elec2.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    p = _download("https://www.openml.org/data/download/2419/electricity-normalized.arff",
                  DATA / "electricity-normalized.arff")
    rows = []
    started = False
    for line in open(p, encoding="utf8", errors="ignore"):
        line = line.strip()
        if not started:
            started = line.lower().startswith("@data")
            continue
        if line and not line.startswith("%"):
            rows.append(line.split(","))
    # columns: date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer, class
    day = np.array([float(r[1]) if r[1].replace(".", "").isdigit() else 0 for r in rows])
    num = np.array([[float(v) for v in r[2:8]] for r in rows])
    X = np.column_stack([num, np.sin(2 * np.pi * day / 7), np.cos(2 * np.pi * day / 7)])
    y = np.array([1 if r[8].strip().strip("'\"").upper() == "UP" else 0 for r in rows])
    np.savez_compressed(cache, X=X, y=y)
    return X, y


class CovertypeSetup:
    """Natural domain shift between wilderness areas (Rawah=0, Neota=1, Comanche=2, Poudre=3).

    Source = 50% of area 0 (fit), 25% reference pool, 25% held-out same-domain targets (exact null).
    Features: 10 numeric cartographic variables (standardised on the fit split).
    """

    def __init__(self, seed=2026, n_fit=4000):
        X, y = load_covertype()
        area = X[:, 10:14].argmax(1)
        num = X[:, :10].astype(float)
        rng = np.random.default_rng(seed)
        a0 = rng.permutation(np.flatnonzero(area == 0))
        fit, ref, held = a0[:n_fit], a0[n_fit:n_fit + 20000], a0[n_fit + 20000:]
        mu, sd = num[fit].mean(0), num[fit].std(0) + 1e-9
        Z = (num - mu) / sd
        self.K = 7
        self.src = Source(Z[fit], y[fit], 7, l2=1e-3)
        self.x_ref, self.y_ref = Z[ref], y[ref]
        self.domains = {"area0_heldout": held}
        for a in (1, 2, 3):
            self.domains[f"area{a}"] = np.flatnonzero(area == a)
        self.Z, self.y = Z, y

    def stream(self, seed, target, T=600, t0=200, rate=0.05):
        rng = np.random.default_rng(seed)
        ir = rng.choice(len(self.y_ref), T, replace=False)
        held = self.domains["area0_heldout"]
        tg = self.domains[target]
        i0 = rng.choice(held, t0, replace=False)
        rest = np.setdiff1d(tg, i0) if target == "area0_heldout" else tg
        i1 = rng.choice(rest, T - t0, replace=False)
        idx = np.concatenate([i0, i1])
        return dict(x_tgt=self.Z[idx], y_tgt=self.y[idx], x_ref=self.x_ref[ir], y_ref=self.y_ref[ir],
                    r=rng.binomial(1, rate, size=T).astype(int), t0=t0, cond=target)


class ElectricitySetup:
    """Chronological Elec2. Source = first n_fit rows; reference pool = next n_ref rows
    (drawn without replacement); monitored windows are consecutive later segments."""

    def __init__(self, n_fit=4000, n_ref=4000):
        X, y = load_electricity()
        self.X, self.y = X, y
        fit = np.arange(n_fit)
        ref = np.arange(n_fit, n_fit + n_ref)
        mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-9
        self.Z = (X - mu) / sd
        self.K = 2
        self.src = Source(self.Z[fit], y[fit], 2, l2=1e-3)
        perm = np.random.default_rng(7).permutation(ref)
        self.ref, self.src_tg = np.sort(perm[: n_ref // 2]), np.sort(perm[n_ref // 2:])
        self.start = n_fit + n_ref

    def stream(self, seed, seg_start, T=1000, rate=0.05):
        rng = np.random.default_rng(seed)
        ir = rng.choice(self.ref, T, replace=False)
        idx = np.arange(seg_start, seg_start + T)
        return dict(x_tgt=self.Z[idx], y_tgt=self.y[idx], x_ref=self.Z[ir], y_ref=self.y[ir],
                    r=rng.binomial(1, rate, size=T).astype(int), t0=0, cond=f"seg{seg_start}")
