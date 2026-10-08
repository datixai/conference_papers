"""Re-implemented comparators. Each returns a statistic trace (length T); an alarm is the first
time the trace exceeds a threshold. 'native' thresholds are each method's own guarantee where it
has one; all methods are additionally calibrated to a matched 5% horizon false-alarm rate.

B1  BBSD-KS             Rabanser, Guennemann & Lipton (NeurIPS 2019)
B2  Classifier-CUSUM    Evans & G'Sell (PLOS ONE 2024)            (label shift)
B3  Page-Hinkley        Page (1954) on revealed log loss
B4  DDM                 Gama et al. (SBIA 2004) on revealed 0/1 errors
B5  ADWIN               Bifet & Gavalda (SDM 2007) on revealed 0/1 errors
B6  Risk-tracking       Podkopaev & Ramdas (ICLR 2022): betting e-process for an error-rate increase
B7  Kernel 2ST-betting  Shekhar & Ramdas (IEEE TIT 2023): paired MMD witness + ONS betting
B8  Seq. classifier 2ST Jang, Park, Lee & Bastani (ICML 2022): online discriminator + ONS betting
B9  Conformal TM        Vovk et al. (2021) / Vovk (2020): conformal test martingale on log-loss scores
B10 WATCH-style WCTM    Prinster et al. (ICML 2025): covariate-weighted conformal test martingale
B11 Specialist ensemble Bonferroni union of B1+B2+B3 with typing by which detector fired
"""
from __future__ import annotations

import math

import numpy as np

from tssd_core import FLOOR, gauss_kernel


class ONS:
    """Online-Newton-step betting (Cutkosky & Orabona; as used by Shekhar & Ramdas)."""

    def __init__(self):
        self.lam, self.a, self.logw = 0.0, 1.0, 0.0
        self.c = 2.0 / (2.0 - math.log(3.0))

    def step(self, d):
        self.logw += math.log(max(1.0 + self.lam * d, FLOOR))
        g = -d / (1.0 + self.lam * d)
        self.a += g * g
        self.lam = float(np.clip(self.lam - self.c * g / self.a, -0.5, 0.5))
        return self.logw


def bbsd(p_tgt, p_src, window=50, step=50):
    out = np.zeros(len(p_tgt))
    srt = [np.sort(p_src[:, j]) for j in range(p_src.shape[1])]
    n = len(p_src)
    ir = np.arange(1, window + 1) / window
    il = np.arange(0, window) / window
    last = 0.0
    for t in range(len(p_tgt)):
        if t >= window - 1 and (t - window + 1) % step == 0:
            rec = p_tgt[t - window + 1:t + 1]
            ds = []
            for j in range(p_src.shape[1]):
                b = np.sort(rec[:, j])
                f = np.searchsorted(srt[j], b, side="right") / n
                ds.append(max((ir - f).max(), (f - il).max()))
            last = max(ds)
        out[t] = last
    return out


def prior_grid(K, pi, rng, n=168):
    g = rng.dirichlet(np.ones(K), size=4 * n)
    g = g[np.abs(g - pi).sum(1) >= 0.10][:n]
    return g


def cusum(p_tgt, pi, grid):
    w = grid / pi[None, :]
    st = np.zeros(len(grid))
    out = np.zeros(len(p_tgt))
    for t, p in enumerate(p_tgt):
        st = np.maximum(st, 0) + np.log(np.maximum(w @ p, 1e-12))
        out[t] = st.max()
    return out


def page_hinkley(p_tgt, y, r, delta=0.005, a=0.9999, mmin=5):
    out = np.zeros(len(y)); mu = s = 0.0; n = 0; last = 0.0
    for t in range(len(y)):
        if r[t]:
            loss = -math.log(max(p_tgt[t, y[t]], 1e-8)); n += 1
            mu += (loss - mu) / n
            s = max(0.0, a * s + loss - mu - delta)
            if n >= mmin:
                last = s
        out[t] = last
    return out


def ddm(err, r, mmin=5):
    out = np.zeros(len(err)); n = 0; k = 0; pmin = smin = None; last = 0.0
    for t in range(len(err)):
        if r[t]:
            n += 1; k += err[t]
            p = (k + 0.5) / (n + 1)
            s = math.sqrt(p * (1 - p) / (n + 1))
            if n >= mmin:
                if pmin is None or p + s < pmin + smin:
                    pmin, smin = p, s
                last = (p + s - pmin) / max(smin, 1e-9)  # drift if >= 3
        out[t] = last
    return out


def adwin(err, r, delta=0.002):
    out = np.zeros(len(err)); buf = []; last = 0.0
    for t in range(len(err)):
        if r[t]:
            buf.append(err[t])
            b = np.asarray(buf, float); n = len(b)
            best = 0.0
            if n >= 6:
                cs = np.cumsum(b)
                for i in range(3, n - 2):
                    n0, n1 = i, n - i
                    m = 1.0 / (1.0 / n0 + 1.0 / n1)
                    eps = math.sqrt(math.log(4.0 * n / delta) / (2.0 * m))
                    gap = abs(cs[i - 1] / n0 - (cs[-1] - cs[i - 1]) / n1)
                    best = max(best, gap / eps)
                if best >= 1.0:  # ADWIN shrinks its window on detection
                    buf = buf[-max(1, n // 2):]
            last = best
        out[t] = last
    return out


def risk_tracking(err, r, r0, eps_tol=0.0):
    """Podkopaev & Ramdas: H0 target risk <= r0 + eps. One-sided betting e-process (ONS)."""
    out = np.zeros(len(err)); ons = ONS(); thr = r0 + eps_tol; logw = 0.0
    scale = max(thr, 1 - thr)
    for t in range(len(err)):
        if r[t]:
            d = (err[t] - thr) / scale
            logw = ons.step(d)
            ons.lam = max(ons.lam, 0.0)  # one-sided: bet only on an increase
        out[t] = logw
    return out


def kernel_2st(x_tgt, x_ref, bw, window=50):
    out = np.zeros(len(x_tgt)); ons = ONS()
    for t in range(1, len(x_tgt)):
        s = max(0, t - window)
        pts = np.vstack([x_tgt[t], x_ref[t]])
        h = gauss_kernel(pts, x_tgt[s:t], bw).mean(1) - gauss_kernel(pts, x_ref[s:t], bw).mean(1)
        out[t] = ons.step(float(np.clip((h[0] - h[1]) / 2.0, -1, 1)))
    return out


class OnlineLogit:
    def __init__(self, d, lr=0.1, l2=1e-3):
        self.w = np.zeros(d + 1); self.lr, self.l2, self.n = lr, l2, 0

    def prob(self, x):
        z = x @ self.w[:-1] + self.w[-1]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def update(self, x, y):
        self.n += 1
        g = (self.prob(x) - y)
        eta = self.lr / math.sqrt(self.n)
        self.w[:-1] -= eta * (g * x + self.l2 * self.w[:-1])
        self.w[-1] -= eta * g


def classifier_2st(x_tgt, x_ref):
    """Jang et al. (2022): predict-then-update discriminator; bet on its paired score gap."""
    d = x_tgt.shape[1]; clf = OnlineLogit(d); ons = ONS(); out = np.zeros(len(x_tgt))
    for t in range(len(x_tgt)):
        ft, fr = clf.prob(x_tgt[t]), clf.prob(x_ref[t])
        out[t] = ons.step(float(ft - fr))
        clf.update(x_tgt[t], 1.0); clf.update(x_ref[t], 0.0)
    return out


def _power_mix(logs_eps, p, eps_grid):
    return logs_eps + np.log(eps_grid) + (eps_grid - 1) * math.log(max(p, 1e-12))


def conformal_tm(score_tgt, r, cal_scores, rng, weights_t=None, weights_cal=None):
    """Smoothed conformal p-values of revealed nonconformity scores, power-martingale mixture
    (eps in a grid). Unweighted: calibration set grows with past revealed test points
    (exchangeable under the null). Weighted (WATCH-style): weights w(x) on the calibration set."""
    eps = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75])
    logs = np.zeros(len(eps)); out = np.zeros(len(score_tgt))
    cal = list(cal_scores); last = 0.0
    for t in range(len(score_tgt)):
        if r[t]:
            s = score_tgt[t]; u = rng.random()
            if weights_t is None:
                c = np.asarray(cal)
                p = ((c > s).sum() + u * ((c == s).sum() + 1)) / (len(c) + 1)
                cal.append(s)
            else:
                wc, wt = weights_cal, weights_t[t]
                c = np.asarray(cal_scores)
                p = ((wc * (c > s)).sum() + u * ((wc * (c == s)).sum() + wt)) / (wc.sum() + wt)
            logs = logs + np.log(eps) + (eps - 1) * math.log(max(p, 1e-12))
            last = float(np.logaddexp.reduce(logs) - math.log(len(eps)))
        out[t] = last
    return out


def watch_weights(x_tgt, x_cal, refit_every=25):
    """Online density-ratio weights w(x)=f/(1-f) from a target-vs-source discriminator trained
    only on past unlabeled targets (predictable)."""
    d = x_tgt.shape[1]; clf = OnlineLogit(d, lr=0.2)
    rng = np.random.default_rng(0)
    wt = np.ones(len(x_tgt)); wcal_t = []
    for t in range(len(x_tgt)):
        f = float(np.clip(clf.prob(x_tgt[t]), 0.02, 0.98)); wt[t] = f / (1 - f)
        if t % refit_every == 0:
            fc = np.clip(clf.prob(x_cal), 0.02, 0.98); wcal_t.append((t, fc / (1 - fc)))
        clf.update(x_tgt[t], 1.0); clf.update(x_cal[rng.integers(len(x_cal))], 0.0)
    return wt, wcal_t


def watch_tm(score_tgt, r, cal_scores, x_tgt, x_cal, rng):
    wt, wcal = watch_weights(x_tgt, x_cal)
    eps = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75])
    logs = np.zeros(len(eps)); out = np.zeros(len(score_tgt)); last = 0.0; j = 0
    c = np.asarray(cal_scores)
    for t in range(len(score_tgt)):
        while j + 1 < len(wcal) and wcal[j + 1][0] <= t:
            j += 1
        if r[t]:
            wc = wcal[j][1]; s = score_tgt[t]; u = rng.random()
            p = ((wc * (c > s)).sum() + u * ((wc * (c == s)).sum() + wt[t])) / (wc.sum() + wt[t])
            logs = logs + np.log(eps) + (eps - 1) * math.log(max(p, 1e-12))
            last = float(np.logaddexp.reduce(logs) - math.log(len(eps)))
        out[t] = last
    return out


BASELINES = ["BBSD", "CUSUM", "PH", "DDM", "ADWIN", "RiskTrack", "Kernel2ST", "Clf2ST", "CTM", "WATCH"]
NATIVE = {  # each method's own alarm rule on the trace (None = no guarantee / heuristic only)
    "BBSD": None, "CUSUM": None, "PH": None, "DDM": 3.0, "ADWIN": 1.0,
    "RiskTrack": math.log(20), "Kernel2ST": math.log(20), "Clf2ST": math.log(20),
    "CTM": math.log(20), "WATCH": math.log(20),
}


class BaselineContext:
    def __init__(self, src, x_cal, y_cal, seed=0):
        self.src = src
        rng = np.random.default_rng(seed)
        self.p_cal = src.proba(x_cal)
        self.x_cal = x_cal
        self.cal_scores = -np.log(np.maximum(self.p_cal[np.arange(len(y_cal)), y_cal], 1e-12))
        self.r0 = float((self.p_cal.argmax(1) != y_cal).mean())
        self.grid = prior_grid(src.K, src.pi, rng)


def run_baselines(stream, ctx: BaselineContext, seed=0):
    src = ctx.src
    xt, yt, xr, r = stream["x_tgt"], stream["y_tgt"], stream["x_ref"], stream["r"]
    pt = src.proba(xt)
    err = (pt.argmax(1) != yt).astype(float)
    sc = -np.log(np.maximum(pt[np.arange(len(yt)), yt], 1e-12))
    rng = np.random.default_rng(seed + 777)
    return {
        "BBSD": bbsd(pt, ctx.p_cal),
        "CUSUM": cusum(pt, src.pi, ctx.grid),
        "PH": page_hinkley(pt, yt, r),
        "DDM": ddm(err, r),
        "ADWIN": adwin(err, r),
        "RiskTrack": risk_tracking(err, r, ctx.r0),
        "Kernel2ST": kernel_2st(xt, xr, src.bw),
        "Clf2ST": classifier_2st(xt, xr),
        "CTM": conformal_tm(sc, r, ctx.cal_scores[:500], rng),
        "WATCH": watch_tm(sc, r, ctx.cal_scores[:500], xt, ctx.x_cal[:500], rng),
    }
