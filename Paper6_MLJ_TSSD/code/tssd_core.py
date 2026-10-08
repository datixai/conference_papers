"""TSSD-2: typed shift sequential diagnosis (core monitor).

Pure numpy/scipy implementation (no scikit-learn dependency).

Witnesses
---------
X  : paired, source-referenced, class-direction-orthogonalised kernel witness (covariate).
L  : paired, source-referenced label/prevalence witness with a predictable covariate gate.
C  : posterior-validity witness. A universal-inference e-process whose null is the
     composite family {eta_{q,beta}(y|x) ∝ (q_y/pi_y) p_g(y|x)^beta}. It is valid under
     ANY covariate shift and ANY label shift (given a calibrated source posterior, up to
     the beta-tempering family), so a C crossing is a certified claim that prior
     correction is insufficient.

Two evidence modes are produced from the same increments
  * test mode  : one test martingale per witness from time 1 (Ville; "ever alarm" control);
  * SR mode    : Shiryaev-Roberts e-detector R_t = sum_j prod_{i=j..t} f_i, which restarts
                 implicitly and gives P(false alarm within m steps) <= m/A and ARL >= A
                 per witness (Shin, Ramdas & Rinaldo).
Ablation traces (raw MMD witness, ungated label witness, paired concept witness of TSSD-1)
are computed in the same loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

FLOOR = 1e-12


# ----------------------------------------------------------------------------- models
def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class LogReg:
    """Multinomial logistic regression with L2 penalty (L-BFGS)."""

    def __init__(self, l2=1e-4):
        self.l2 = l2

    def fit(self, X, y, K):
        X = np.asarray(X, float)
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])
        Y = np.eye(K)[y]
        l2 = self.l2

        def f(w):
            W = w.reshape(d + 1, K)
            Z = Xb @ W
            lse = logsumexp(Z, axis=1)
            nll = (lse - (Z * Y).sum(1)).mean() + 0.5 * l2 * (W[:-1] ** 2).sum()
            P = np.exp(Z - lse[:, None])
            G = Xb.T @ (P - Y) / n
            G[:-1] += l2 * W[:-1]
            return nll, G.ravel()

        res = minimize(f, np.zeros((d + 1) * K), jac=True, method="L-BFGS-B",
                       options={"maxiter": 5000, "gtol": 1e-9})
        self.W = res.x.reshape(d + 1, K)
        self.K = K
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        return softmax(X @ self.W[:-1] + self.W[-1])


def gauss_kernel(a, b, bw):
    a2 = (a * a).sum(1)[:, None]
    b2 = (b * b).sum(1)[None, :]
    return np.exp(-np.maximum(a2 + b2 - 2 * a @ b.T, 0.0) / (2 * bw * bw))


def median_bandwidth(x, rng=None, m=1000):
    rng = rng or np.random.default_rng(0)
    if len(x) > m:
        x = x[rng.choice(len(x), m, replace=False)]
    a2 = (x * x).sum(1)
    d2 = np.maximum(a2[:, None] + a2[None, :] - 2 * x @ x.T, 0)
    return float(np.sqrt(np.median(d2[np.triu_indices(len(x), 1)])))


class Source:
    """Frozen source model and every source quantity the monitor needs."""

    def __init__(self, x_fit, y_fit, K, l2=1e-4, n_kernel=2000, seed=2026):
        rng = np.random.default_rng(seed)
        self.K = K
        self.model = LogReg(l2).fit(x_fit, y_fit, K)
        self.pi = np.bincount(y_fit, minlength=K) / len(y_fit)
        self.pi = np.maximum(self.pi, 1e-3)
        self.pi /= self.pi.sum()
        self.log_pi = np.log(self.pi)
        xk = x_fit if len(x_fit) <= n_kernel else x_fit[rng.choice(len(x_fit), n_kernel, replace=False)]
        self.x_kern = np.asarray(xk, float)
        self.bw = median_bandwidth(self.x_kern, rng)
        p = self.model.predict_proba(self.x_kern)
        self.v_kern = self.vfeat(p)
        G = self.v_kern.T @ self.v_kern / len(self.x_kern) + 1e-8 * np.eye(K - 1)
        self.g_inv = np.linalg.pinv(G)
        self.mean_v = self.v_kern.mean(0)

    def vfeat(self, p):
        return p[:, : self.K - 1] / self.pi[: self.K - 1] - 1.0

    def proba(self, x):
        return self.model.predict_proba(x)

    def kernel_source_means(self, z, chunk=1500):
        """mean_i k(x_i, z) and mean_i k(x_i, z) v_i for every row z."""
        km = np.empty(len(z))
        kv = np.empty((len(z), self.K - 1))
        for s in range(0, len(z), chunk):
            Kz = gauss_kernel(self.x_kern, z[s:s + chunk], self.bw)  # n0 x c
            km[s:s + chunk] = Kz.mean(0)
            kv[s:s + chunk] = Kz.T @ self.v_kern / len(self.x_kern)
        return km, kv


# ----------------------------------------------------------------------------- config
@dataclass
class Config:
    alpha: float = 0.05
    sr_horizon: int = 400            # m in the finite-horizon SR bound 3m/A <= alpha
    window: int = 50                 # covariate kernel window
    em_window: int = 200             # label-share EM window
    em_steps: int = 10
    gate_test: float = 2.0           # gate on past max log test-evidence of X
    gate_sr: float = 3.0             # gate on log(R^X_{t-1}/(t-1)) in SR mode
    lam_X: tuple = (0.5, 1.0)
    lam_L: tuple = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
    lam_Cp: tuple = (1.0,)
    m_min: int = 5
    decay: float = 0.85
    alt: str = "mixture"             # C alternative: "mixture" (3 experts, fixed share) or "basic"
    alt_mix: float = 0.5             # weight of eta-hat in the "basic" C alternative
    share: float = 0.01              # fixed-share rate of the expert mixture
    ms_l2: float = 1.0               # ridge (towards the frozen logits) of the refit expert
    ms_window: int = 50              # most recent labels used by the refit expert
    betas: tuple = (0.7, 1.0, 1.4)   # tempering grid of the composite posterior null
    u_box: float = 8.0               # |log q ratios| <= u_box in the composite null
    max_starts: int = 40             # SR starts kept for C (dropping starts keeps validity)
    orth: bool = True
    clips: dict = field(default_factory=dict)  # keys X, Xraw, L, Cp

    @property
    def log_thr_test(self):
        return math.log(3.0 / self.alpha)

    @property
    def log_thr_sr(self):
        return math.log(3.0 * self.sr_horizon / self.alpha)


# ----------------------------------------------------------------------------- helpers
def _log1p_vec(z):
    out = np.full(z.shape, -np.inf)
    ok = z > -1.0
    out[ok] = np.log1p(z[ok])
    return out


class PairedEvidence:
    """Two-sided capital mixture in test mode and SR mode for a bounded increment."""

    def __init__(self, lams):
        lam = np.asarray(lams, float)
        self.sl = np.concatenate([lam, -lam])  # signed bets
        self.log_m = np.zeros(len(self.sl))
        self.log_r = np.full(len(self.sl), -np.inf)
        self.nc = len(self.sl)

    def step(self, d, update_test=True, update_sr=True):
        f = _log1p_vec(self.sl * d)
        if update_test:
            self.log_m += f
        if update_sr:
            self.log_r = f + np.logaddexp(self.log_r, 0.0)
        else:
            self.log_r = np.logaddexp(self.log_r, 0.0)

    @property
    def test(self):
        return float(logsumexp(self.log_m) - math.log(self.nc))

    @property
    def sr(self):
        return float(logsumexp(self.log_r) - math.log(self.nc))


def clipped(raw, c):
    return float(np.clip(raw, -c, c) / c)


def paired_d(a, b, c):
    return float(np.clip((clipped(a, c) - clipped(b, c)) / 2.0, -1, 1))


def em_q(q, p_hist, r_hist, y_hist, pi, steps):
    if len(p_hist) == 0:
        return q
    unl = r_hist == 0
    K = len(pi)
    labc = np.bincount(y_hist[~unl], minlength=K).astype(float)
    pu = p_hist[unl]
    for _ in range(steps):
        cnt = labc.copy()
        if len(pu):
            w = pu * (q / pi)[None, :]
            w /= np.maximum(w.sum(1, keepdims=True), FLOOR)
            cnt += w.sum(0)
        q = np.maximum(cnt / len(p_hist), 1e-3)
        q /= q.sum()
    return q


def eta(p, q, pi, beta=1.0):
    w = (q / pi) * np.power(np.maximum(p, FLOOR), beta)
    return w / w.sum(-1, keepdims=True)


def remap(p, conf):
    R = conf / np.maximum(conf.sum(1, keepdims=True), FLOOR)
    a = R.T @ p
    return a / a.sum()


# ----------------------------------------------------------------------------- C witness
def suffix_sup_certified(base, Y, U, box, iters=40, tol=1e-6):
    """Certified upper bounds on sup_u sum_{i>=c} log softmax(base_i + u)_{y_i} over the box
    |u_k| <= box (u_0 = 0), simultaneously for every suffix start c.

    Active-set projected Newton (Bertsekas) is run from warm starts U (n, K-1). The returned
    value is f(u) + max_{v in box} grad f(u).(v - u), an upper bound on the supremum by
    concavity, so the universal-inference denominator is valid even if the optimiser stops early.
    """
    n, K = base.shape
    M = np.triu(np.ones((n, n)))  # M[c, i] = 1 if i >= c
    Y1 = np.eye(K)[Y]
    I = np.eye(K - 1)[None]

    def evaluate(Uc):
        u = np.concatenate([np.zeros((n, 1)), Uc], 1)
        Z = base[None, :, :] + u[:, None, :]            # (c, i, k)
        lse = logsumexp(Z, axis=2)
        P = np.exp(Z - lse[:, :, None])
        f = (M * (Z[:, np.arange(n), Y] - lse)).sum(1)
        G = np.einsum("ci,cik->ck", M, Y1[None, :, :] - P)[:, 1:]
        return f, G, P

    def gap_of(G, U):
        return np.where(G > 0, G * (box - U), G * (-box - U)).sum(1)

    U = np.clip(U, -box, box)
    f, G, P = evaluate(U)
    for _ in range(iters):
        gap = gap_of(G, U)
        if gap.max() < tol:
            break
        act = ((U <= -box + 1e-10) & (G < 0)) | ((U >= box - 1e-10) & (G > 0))
        Pk = P[:, :, 1:]
        Hd = np.einsum("ci,cik->ck", M, Pk)
        H = Hd[:, :, None] * I - np.einsum("ci,cik,cij->ckj", M, Pk, Pk)
        free = (~act).astype(float)
        H = H * free[:, :, None] * free[:, None, :] + (1 - free)[:, :, None] * I + 1e-10 * I
        d = np.linalg.solve(H, (G * free)[:, :, None])[:, :, 0]
        step = np.ones(n)
        done = gap < tol
        improved = False
        for _ in range(10):
            cand = np.clip(U + step[:, None] * d, -box, box)
            fc, Gc, Pc = evaluate(cand)
            ok = (fc > f + 1e-12) & ~done
            U[ok], f[ok], G[ok], P[ok] = cand[ok], fc[ok], Gc[ok], Pc[ok]
            improved |= bool(ok.any())
            done |= ok
            if done.all():
                break
            step = np.where(done, step, step * 0.5)
        if not improved:
            break
    gap = gap_of(G, U)
    # exact fallback for problems the batched Newton did not close (e.g. near-zero-probability classes)
    for c in np.flatnonzero(gap > 1e-4):
        b, yc = base[c:], Y[c:]
        Yc = Y1[c:]

        def fc(u1):
            Z = b + np.concatenate([[0.0], u1])[None, :]
            lse = logsumexp(Z, axis=1)
            P = np.exp(Z - lse[:, None])
            return -((Z * Yc).sum() - lse.sum()), -(Yc - P).sum(0)[1:]

        r = minimize(fc, U[c], jac=True, method="L-BFGS-B", bounds=[(-box, box)] * (K - 1),
                     options={"maxiter": 1000, "gtol": 1e-10, "ftol": 1e-15})
        fv, gv = fc(r.x)
        gc = -gv
        bound = -fv + max(float(np.where(gc > 0, gc * (box - r.x), gc * (-box - r.x)).sum()), 0.0)
        cur = f[c] + max(gap[c], 0.0)
        if bound < cur:
            U[c], f[c], gap[c] = r.x, -fv, bound + fv
    return f + np.maximum(gap, 0.0), U


class PosteriorValidity:
    """Universal-inference e-process for the composite posterior-validity null

        H0^C: P_t(Y|X) = eta_{theta_t}(.|X), theta_t = (q_t, beta_t) piecewise constant in t
              with at most ONE change point, eta_theta(y|x) proportional to (q_y/pi_y) p_g(y|x)^beta.

    This contains the source regime, any covariate shift (posterior preserved), and a label
    shift at an unknown time. Numerator: predictable alternative a_{t-1}(y|x). Denominator for
    start m (labelled points m..n): max over a split c of sup_theta L(m..c-1) * sup_theta L(c..n),
    computed exactly from cached suffix suprema S(c, n). Test mode = start 0; SR mode sums starts.
    """

    def __init__(self, cfg: Config, src: Source):
        self.cfg, self.src = cfg, src
        K = src.K
        self.lp = []      # log p_g rows of labelled points
        self.y = []
        self.t_lab = []   # 1-based times of labelled points
        self.cum_la = []  # per start: sum of log alternative
        self.u_warm = []  # per suffix start c: dict beta -> warm start
        self.S = []       # S[c][n - c] = sup log-lik over labelled points c..n
        self.warm = {}    # beta -> warm-start matrix (one row per suffix)
        self.log_e = []   # per start current log e-value
        self.prior_c = 1.0 if cfg.alt == "basic" else 1.0 / K
        self.conf = np.full((K, K), self.prior_c)
        self.active = []  # active start indices (SR)
        self.test_log = 0.0
        self.w = np.full(3, 1.0 / 3)              # expert weights (fixed share)
        self.ms = None                            # refit-expert params (lazy init)
        self.xs = []                              # inputs of labelled points

    def _feat(self, x, p):
        return np.concatenate([np.asarray(x, float), np.log(np.maximum(p, 1e-4)), [1.0]])

    def _init_ms(self, dim):
        K = self.src.K
        W0 = np.zeros((dim, K))
        d = dim - K - 1
        W0[d:d + K] = np.eye(K)  # start at the frozen model's logits
        return W0

    def _experts(self, p, q_hat, x):
        e1 = eta(p, q_hat, self.src.pi)
        e2 = remap(p, self.conf)
        f = self._feat(x, p)
        if self.ms is None:
            self.ms = self._init_ms(len(f))
        e3 = softmax(f @ self.ms)
        return np.vstack([e1, e2, e3])

    def alternative(self, p, q_hat, x=None):
        cfg = self.cfg
        if cfg.alt == "basic":
            e_hat = eta(p, q_hat, self.src.pi)
            if len(self.y) < cfg.m_min:
                return e_hat
            return cfg.alt_mix * e_hat + (1 - cfg.alt_mix) * remap(p, self.conf)
        self._E = self._experts(p, q_hat, x)
        return self.w @ self._E

    def _fit_refit_expert(self):
        """Ridge multinomial refit on [x, log p_g(x), 1], shrunk towards the frozen logits."""
        K = self.src.K
        F = np.asarray(self.xs[-self.cfg.ms_window:])
        Y = np.eye(K)[np.asarray(self.y[-self.cfg.ms_window:])]
        W0 = self._init_ms(F.shape[1])
        lam = self.cfg.ms_l2

        def f(w):
            W = w.reshape(W0.shape)
            A = F @ W
            lse = logsumexp(A, axis=1)
            P = np.exp(A - lse[:, None])
            D = W - W0
            return (lse - (A * Y).sum(1)).sum() + 0.5 * lam * (D * D).sum(), (F.T @ (P - Y) + lam * D).ravel()

        r = minimize(f, self.ms.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 200})
        self.ms = r.x.reshape(W0.shape)

    def observe(self, t1, p, y, q_hat, x=None):
        """Called on a revealed label at 1-based time t1 (alternative formed before y)."""
        a = self.alternative(p, q_hat, x)
        la = math.log(max(float(a[y]), FLOOR))
        self.lp.append(np.log(np.maximum(p, FLOOR)))
        self.y.append(int(y))
        self.t_lab.append(t1)
        n = len(self.y) - 1
        self.cum_la.append(0.0)
        self.u_warm.append({})
        self.S.append([])
        self.log_e.append(0.0)
        self.active.append(n)
        if len(self.active) > self.cfg.max_starts:
            self.active.pop(1)  # drop the oldest SR start; start 0 (test mode) is always kept
        LP = np.asarray(self.lp)
        Y = np.asarray(self.y)
        # certified suffix suprema S(c, n) for every c, maximised over the beta grid
        K = self.src.K
        suf = np.full(n + 1, -np.inf)
        for b in self.cfg.betas:
            W = self.warm.get(b)
            W = np.zeros((n + 1, K - 1)) if W is None else np.vstack([W, np.zeros((1, K - 1))])
            val, self.warm[b] = suffix_sup_certified(b * LP - self.src.log_pi[None, :], Y, W, self.cfg.u_box)
            suf = np.maximum(suf, val)
        for c in range(n + 1):
            self.S[c].append(float(suf[c]))
        for m in self.active:
            self.cum_la[m] += la
            den = suf[m]
            if n > m:
                pre = np.array([self.S[m][c - 1 - m] for c in range(m + 1, n + 1)])  # S(m, c-1)
                den = max(den, float(np.max(pre + suf[m + 1:n + 1])))
            self.log_e[m] = self.cum_la[m] - den
        self.test_log = self.log_e[0]
        # alternative-state updates (after the bet)
        if self.cfg.alt != "basic":
            lik = np.maximum(self._E[:, int(y)], FLOOR)
            w = self.w * lik
            w /= w.sum()
            self.w = (1 - self.cfg.share) * w + self.cfg.share / 3.0
        pc = self.prior_c
        self.conf = self.cfg.decay * (self.conf - pc) + pc
        self.conf[:, int(y)] += p
        if self.cfg.alt != "basic":
            self.xs.append(self._feat(x, p))
            self._fit_refit_expert()

    def sr_log(self, t1):
        """log R_t at 1-based time t1 (after any observation at t1)."""
        terms = []
        mults = np.diff([0] + self.t_lab)
        for m in self.active:
            terms.append(self.log_e[m] + math.log(mults[m]))
        last = self.t_lab[-1] if self.t_lab else 0
        if t1 - last > 0:
            terms.append(math.log(t1 - last))
        return float(logsumexp(terms)) if terms else -np.inf


# ----------------------------------------------------------------------------- monitor
TRACE_KEYS = ["X", "Xraw", "L", "Lng", "C", "Cp", "X_sr", "L_sr", "C_sr", "Cp_sr"]


def run_monitor(stream, src: Source, cfg: Config, collect_raw=False):
    """Run TSSD-2 on a stream dict with keys x_tgt, y_tgt, x_ref, y_ref, r.

    Returns dict of float32 traces (log evidence after each step) or raw-score lists.
    """
    xt, yt, xr, yr, r = (stream[k] for k in ("x_tgt", "y_tgt", "x_ref", "y_ref", "r"))
    T = len(yt)
    K = src.K
    pt, pr = src.proba(xt), src.proba(xr)
    vt, vr = src.vfeat(pt), src.vfeat(pr)
    km, kv = src.kernel_source_means(np.vstack([xt, xr]))
    km_t, km_r, kv_t, kv_r = km[:T], km[T:], kv[:T], kv[T:]
    clips = cfg.clips
    ev = {k: PairedEvidence(l) for k, l in
          (("X", cfg.lam_X), ("Xraw", cfg.lam_X), ("L", cfg.lam_L), ("Lng", cfg.lam_L), ("Lsr", cfg.lam_L), ("Cp", cfg.lam_Cp))}
    pv = PosteriorValidity(cfg, src)
    q = src.pi.copy()
    conf_p = np.ones((K, K))
    n_lab = 0
    max_x = 0.0
    traces = {k: np.zeros(T, np.float32) for k in TRACE_KEYS}
    raw = {k: [] for k in ("X", "Xraw", "L", "Cp")}
    for t in range(T):
        # ---------------- covariate scores (state from steps < t)
        if t == 0:
            hx_t = hx_r = hr_t = hr_r = 0.0
        else:
            s = max(0, t - cfg.window)
            Wt, Wr = xt[s:t], xr[s:t]
            pts = np.vstack([xt[t], xr[t]])
            kt = gauss_kernel(pts, Wt, src.bw).mean(1)
            kr = gauss_kernel(pts, Wr, src.bw).mean(1)
            h = kt - kr
            hr_t, hr_r = float(h[0]), float(h[1])
            c_t = kv_t[s:t].mean(0) - kv_r[s:t].mean(0)
            mh = km_t[s:t].mean() - km_r[s:t].mean()
            b_t = mh - c_t @ src.g_inv @ src.mean_v
            hx_t = hr_t - float(c_t @ src.g_inv @ vt[t]) - b_t
            hx_r = hr_r - float(c_t @ src.g_inv @ vr[t]) - b_t
            if not cfg.orth:
                hx_t, hx_r = hr_t, hr_r
        # ---------------- label scores
        rt = int(r[t])
        if rt:
            lt = math.log(max(q[yt[t]] / src.pi[yt[t]], 1e-8))
            lr = math.log(max(q[yr[t]] / src.pi[yr[t]], 1e-8))
        else:
            lt = math.log(max(float((q / src.pi) @ pt[t]), 1e-8))
            lr = math.log(max(float((q / src.pi) @ pr[t]), 1e-8))
        # ---------------- paired concept (TSSD-1 ablation)
        if rt and n_lab >= cfg.m_min:
            def ck(p, y):
                a = remap(p, conf_p)
                e = eta(p, q, src.pi)
                return math.log(max(a[y], 1e-8)) - math.log(max(e[y], 1e-8))
            ct, cr = ck(pt[t], yt[t]), ck(pr[t], yr[t])
        else:
            ct = cr = 0.0
        if collect_raw:
            if t > 0:
                raw["X"] += [abs(hx_t), abs(hx_r)]
                raw["Xraw"] += [abs(hr_t), abs(hr_r)]
            raw["L"] += [abs(lt), abs(lr)]
            if rt and n_lab >= cfg.m_min:
                raw["Cp"] += [abs(ct), abs(cr)]
        else:
            dX = paired_d(hx_t, hx_r, clips["X"])
            dXr = paired_d(hr_t, hr_r, clips["Xraw"])
            dL = paired_d(lt, lr, clips["L"])
            dC = paired_d(ct, cr, clips["Cp"])
            # gates are computed from evidence before step t
            gate_test = max_x < cfg.gate_test
            gate_sr = (t == 0) or (ev["X"].sr - math.log(t) < cfg.gate_sr)
            ev["X"].step(dX)
            ev["Xraw"].step(dXr)
            max_x = max(max_x, ev["X"].test)
            ev["Lng"].step(dL)
            ev["L"].step(dL, update_test=gate_test, update_sr=False)
            ev["Lsr"].step(dL, update_test=False, update_sr=gate_sr)
            ev["Cp"].step(dC if rt else 0.0)
        # ---------------- posterior-validity witness (labels only)
        if rt:
            if not collect_raw:
                pv.observe(t + 1, pt[t], int(yt[t]), q, xt[t])
            n_lab += 1
            conf_p = cfg.decay * (conf_p - 1.0) + 1.0
            conf_p[:, int(yt[t])] += pt[t]
        if not collect_raw:
            traces["X"][t] = ev["X"].test
            traces["Xraw"][t] = ev["Xraw"].test
            traces["L"][t] = ev["L"].test
            traces["Lng"][t] = ev["Lng"].test
            traces["Cp"][t] = ev["Cp"].test
            traces["C"][t] = pv.test_log
            traces["X_sr"][t] = ev["X"].sr
            traces["L_sr"][t] = ev["Lsr"].sr
            traces["Cp_sr"][t] = ev["Cp"].sr
            traces["C_sr"][t] = pv.sr_log(t + 1)
        # ---------------- EM update of q (windowed), stored for t+1
        s = max(0, t + 1 - cfg.em_window)
        q = em_q(q, pt[s:t + 1], r[s:t + 1], yt[s:t + 1], src.pi, cfg.em_steps)
    if collect_raw:
        return raw
    traces["n_lab"] = int(np.sum(r))
    return traces


def calibrate_clips(streams, src, cfg, quantile=0.95):
    pool = {k: [] for k in ("X", "Xraw", "L", "Cp")}
    for st in streams:
        raw = run_monitor(st, src, cfg, collect_raw=True)
        for k in pool:
            pool[k] += raw[k]
    return {k: float(np.quantile(np.asarray(v), quantile)) for k, v in pool.items()}, {k: len(v) for k, v in pool.items()}
