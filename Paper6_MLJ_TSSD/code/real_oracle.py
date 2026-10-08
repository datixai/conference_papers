"""Offline full-label 'oracle' characterisation of real shifts (not available to the monitor).

For each target domain/segment, with a random 50/50 split A/B of fully labelled target data:
  LL_frozen   : log loss of the frozen source posterior on B
  LL_prior    : log loss after the best label-shift correction (q, beta) fitted on A (the C-null family)
  LL_refit    : log loss of a logistic model refitted on A
  gap         : LL_prior - LL_refit  (posterior change NOT explained by a prior correction)
  AUC_X       : domain-classifier AUC (source reference vs target inputs), fit on A, scored on B
Oracle action type: 'concept' if gap > 0.05 nats with a bootstrap lower bound > 0, else
'label' if LL_frozen - LL_prior > 0.02, else 'covariate' if AUC_X > 0.6, else 'none'.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tssd_core import LogReg, FLOOR  # noqa: E402
import tssd_streams as S  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"


def fit_prior(lp, y, log_pi, betas=(0.7, 1.0, 1.4)):
    K = lp.shape[1]; Y = np.eye(K)[y]; best = (-np.inf, None, None)
    for b in betas:
        base = b * lp - log_pi
        def f(u1):
            u = np.concatenate([[0.0], u1]); Z = base + u; lse = logsumexp(Z, 1)
            P = np.exp(Z - lse[:, None])
            return -((Z * Y).sum() - lse.sum()), -(Y - P).sum(0)[1:]
        r = minimize(f, np.zeros(K - 1), jac=True, method="L-BFGS-B", bounds=[(-8, 8)] * (K - 1))
        if -r.fun > best[0]:
            best = (-r.fun, b, np.concatenate([[0.0], r.x]))
    return best[1], best[2]


def ll_of(P, y):
    return -np.log(np.maximum(P[np.arange(len(y)), y], 1e-12))


def auc(s_pos, s_neg):
    s = np.concatenate([s_pos, s_neg]); r = s.argsort().argsort() + 1
    return (r[:len(s_pos)].sum() - len(s_pos) * (len(s_pos) + 1) / 2) / (len(s_pos) * len(s_neg))


def characterise(src, x, y, x_src, l2, rng):
    n = len(y); perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
    lp = np.log(np.maximum(src.proba(x), FLOOR))
    b, u = fit_prior(lp[A], y[A], src.log_pi)
    Pprior = np.exp(b * lp[B] - src.log_pi + u); Pprior /= Pprior.sum(1, keepdims=True)
    refit = LogReg(l2).fit(x[A], y[A], src.K)
    l_f, l_p, l_r = ll_of(np.exp(lp[B]), y[B]), ll_of(Pprior, y[B]), ll_of(refit.predict_proba(x[B]), y[B])
    d = l_p - l_r
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]
    ms = rng.choice(len(x_src), n, replace=len(x_src) < n)
    xs = x_src[ms]
    dom = LogReg(1e-3).fit(np.vstack([xs[A], x[A]]), np.r_[np.zeros(len(A), int), np.ones(len(A), int)], 2)
    a = auc(dom.predict_proba(x[B])[:, 1], dom.predict_proba(xs[B])[:, 1])
    gap, lo = float(d.mean()), float(np.quantile(boots, 0.025))
    typ = ("concept" if (gap > 0.05 and lo > 0) else "label" if (l_f.mean() - l_p.mean() > 0.02)
           else "covariate" if a > 0.6 else "none")
    return dict(n=int(n), LL_frozen=float(l_f.mean()), LL_prior=float(l_p.mean()), LL_refit=float(l_r.mean()),
                gap=gap, gap_lo=lo, beta=float(b), AUC_X=float(a), acc_frozen=float((np.exp(lp[B]).argmax(1) == y[B]).mean()),
                oracle=typ)


def main():
    rng = np.random.default_rng(123)
    out = {}
    C = S.CovertypeSetup()
    for dom, idx in C.domains.items():
        ii = rng.choice(idx, min(6000, len(idx)), replace=False)
        out[f"covertype/{dom}"] = characterise(C.src, C.Z[ii], C.y[ii], C.x_ref, 1e-3, rng)
        print(dom, out[f"covertype/{dom}"], flush=True)
    E = S.ElectricitySetup()
    xs = E.Z[E.ref]
    ii = E.src_tg
    out["electricity/src"] = characterise(E.src, E.Z[ii], E.y[ii], xs, 1e-3, rng)
    for s0 in range(8000, 44000, 2000):
        ii = np.arange(s0, s0 + 1000)
        out[f"electricity/seg{s0}"] = characterise(E.src, E.Z[ii], E.y[ii], xs, 1e-3, rng)
        print(s0, out[f"electricity/seg{s0}"], flush=True)
    (RES / "real_oracle.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
