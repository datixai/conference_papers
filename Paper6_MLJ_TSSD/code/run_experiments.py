"""Run all TSSD-2 experiments. Usage: python run_experiments.py <experiment> [...]

Experiments: gauss_main, gauss_reveal, gauss_magnitude, gauss_changetime, gauss_ablation,
             digits, covertype, electricity
Each writes results/traces_<name>.npz (traces) and results/meta_<name>.json.
All seeds are fresh (>= 30000) and disjoint from the clip-calibration seeds (9000-9019).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tssd_core import Config, calibrate_clips, run_monitor, TRACE_KEYS  # noqa: E402
import tssd_streams as S  # noqa: E402
from tssd_baselines import BaselineContext, run_baselines, BASELINES  # noqa: E402

RES = HERE.parent / "results"
RES.mkdir(exist_ok=True)
NPROC = max(1, (os.cpu_count() or 2) - 1)

_W = {}


# ----------------------------------------------------------------------------- dataset setups
def setup(dataset):
    """Return (src, stream_fn(seed, cond, **kw), clip_streams, ctx, sr_horizon)."""
    if dataset == "gauss":
        means, src = S.g_source()
        fn = lambda seed, cond, **kw: S.g_stream(seed, cond, means, **kw)  # noqa: E731
        clip_streams = [S.g_stream(9000 + i, "null", means) for i in range(20)]
        xc, yc = S.g_draw(np.random.default_rng(777), 2000, means)
        return src, fn, clip_streams, BaselineContext(src, xc, yc), 400
    if dataset == "digits":
        D = S.DigitsSetup()
        clip_streams = [D.stream(9000 + i, "null") for i in range(10)]
        return D.src, D.stream, clip_streams, BaselineContext(D.src, D.x_ref, D.y_ref), 300
    if dataset == "covertype":
        C = S.CovertypeSetup()
        clip_streams = [C.stream(9000 + i, "area0_heldout") for i in range(10)]
        return C.src, C.stream, clip_streams, BaselineContext(C.src, C.x_ref[:4000], C.y_ref[:4000]), 600
    if dataset == "electricity":
        E = S.ElectricitySetup()
        # source-period null streams: targets from a random half of the reference period in random
        # order, references from the disjoint other half (both without replacement)
        def src_stream(seed):
            g = np.random.default_rng(seed)
            it = g.choice(E.src_tg, 1000, replace=False)  # random order: an exchangeable null
            ir = g.choice(E.ref, 1000, replace=False)
            return dict(x_tgt=E.Z[it], y_tgt=E.y[it], x_ref=E.Z[ir], y_ref=E.y[ir],
                        r=g.binomial(1, 0.05, 1000).astype(int), t0=0, cond="src")
        clip_streams = [src_stream(9000 + i) for i in range(10)]

        def fn(seed, cond, **kw):
            if cond == "src":
                return src_stream(seed)
            return E.stream(seed, int(cond[3:]), **kw)
        return E.src, fn, clip_streams, BaselineContext(E.src, E.Z[E.ref][:2000], E.y[E.ref][:2000]), 1000
    raise ValueError(dataset)


def _init(dataset, cfg_dict, cfg_overrides):
    src, fn, _, ctx, _ = setup(dataset)
    _W.update(src=src, fn=fn, ctx=ctx, cfg=Config(**cfg_dict), over=cfg_overrides)


def _job(job):
    seed, cond, kw, tag, with_base = job
    cfg = _W["cfg"]
    if tag in _W["over"]:
        cfg = replace(cfg, **_W["over"][tag])
    st = _W["fn"](seed, cond, **kw)
    tr = run_monitor(st, _W["src"], cfg)
    out = {k: tr[k] for k in TRACE_KEYS}
    if with_base:
        out.update(run_baselines(st, _W["ctx"], seed))
    t0 = st.get("t0", 0) or 0
    meta = dict(seed=seed, cond=cond, tag=tag, t0=int(t0), T=len(st["y_tgt"]),
                rate=float(kw.get("rate", 0.05)), n_lab=int(st["r"].sum()),
                n_post=int(st["r"][t0:].sum()), **{k: v for k, v in kw.items() if k not in ("rate", "T", "t0")})
    return meta, {k: np.asarray(v, np.float32) for k, v in out.items()}


def run(name, dataset, jobs, cfg_overrides=None):
    t = time.time()
    src, fn, clip_streams, ctx, horizon = setup(dataset)
    cfg = Config(sr_horizon=horizon)
    cfg.clips, nclip = calibrate_clips(clip_streams, src, cfg)
    cfg_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    metas, traces = [], {}
    with ProcessPoolExecutor(NPROC, initializer=_init, initargs=(dataset, cfg_dict, cfg_overrides or {})) as ex:
        for i, (meta, tr) in enumerate(ex.map(_job, jobs, chunksize=4)):
            metas.append(meta)
            for k, v in tr.items():
                traces.setdefault(k, []).append(v)
    # pad to common length
    L = max(len(v) for v in traces["X"])
    arr = {}
    for k, vs in traces.items():
        a = np.full((len(vs), L), np.nan, np.float32)
        for i, v in enumerate(vs):
            a[i, :len(v)] = v
        arr[k] = a
    np.savez_compressed(RES / f"traces_{name}.npz", **arr)
    info = dict(dataset=dataset, clips=cfg.clips, n_clip_scores=nclip, bandwidth=src.bw,
                log_thr_test=cfg.log_thr_test, log_thr_sr=cfg.log_thr_sr, sr_horizon=cfg.sr_horizon,
                n_jobs=len(jobs), seconds=round(time.time() - t, 1), meta=metas,
                overrides=cfg_overrides or {})
    (RES / f"meta_{name}.json").write_text(json.dumps(info, default=float))
    print(f"{name}: {len(jobs)} streams in {time.time() - t:.0f}s; clips={cfg.clips}", flush=True)


SHIFTS = ["covariate", "label", "concept", "covariate_label", "covariate_concept", "label_concept"]


def main(which):
    if which == "gauss_main":
        jobs = [(30000 + i, "null", {}, "cal", True) for i in range(1000)]
        jobs += [(40000 + i, "null", {}, "eval", True) for i in range(1000)]
        for ci, c in enumerate(SHIFTS):
            jobs += [(50000 + 1000 * ci + i, c, {}, "main", True) for i in range(200)]
        run(which, "gauss", jobs)
    elif which == "gauss_reveal":
        jobs = []
        for ri, rate in enumerate([0.02, 0.10, 0.20]):
            jobs += [(60000 + 1000 * ri + i, "null", {"rate": rate}, "rev", True) for i in range(200)]
            for ci, c in enumerate(["covariate", "label", "concept", "label_concept"]):
                jobs += [(61000 + 10000 * ri + 1000 * ci + i, c, {"rate": rate}, "rev", True) for i in range(200)]
        run(which, "gauss", jobs)
    elif which == "gauss_magnitude":
        jobs = []
        for v in (0.1, 0.2, 0.35, 0.5):
            jobs += [(70000 + int(v * 1000) * 7 + i, "covariate", {"cov_scale": v}, "mag", True) for i in range(200)]
        for v in (0.25, 0.5, 0.75):
            jobs += [(80000 + int(v * 1000) * 7 + i, "label", {"label_gamma": v}, "mag", True) for i in range(200)]
        for v in (0.1, 0.25, 0.5):
            jobs += [(90000 + int(v * 1000) * 7 + i, "concept", {"concept_frac": v}, "mag", True) for i in range(200)]
        run(which, "gauss", jobs)
    elif which == "gauss_changetime":
        jobs = []
        for t0 in (100, 500, 1000, 2000):
            for ci, c in enumerate(["covariate", "label", "concept"]):
                jobs += [(100000 + 10 * t0 + 1000 * ci * 7 + i, c, {"t0": t0, "T": t0 + 300}, "ct", False)
                         for i in range(100)]
        jobs += [(130000 + i, "null", {"t0": 2300, "T": 2300}, "ct", False) for i in range(200)]
        run(which, "gauss", jobs)
    elif which == "gauss_ablation":
        over = {
            "win25": {"window": 25}, "win100": {"window": 100},
            "decay70": {"decay": 0.70}, "decay95": {"decay": 0.95},
            "gate1": {"gate_test": 1.0, "gate_sr": 2.0}, "gate3": {"gate_test": 3.0, "gate_sr": 4.0},
            "beta1": {"betas": (1.0,)}, "betawide": {"betas": (0.5, 0.7, 1.0, 1.4, 2.0)},
            "mix25": {"alt_mix": 0.25}, "mix75": {"alt_mix": 0.75},
            "emfull": {"em_window": 10000},
            "altbasic": {"alt": "basic"}, "box4": {"u_box": 4.0},
        }
        jobs = []
        for tag in over:
            jobs += [(140000 + i, "null", {}, tag, False) for i in range(300)]
            for ci, c in enumerate(["covariate", "label", "concept", "covariate_label", "label_concept"]):
                jobs += [(150000 + 1000 * ci + i, c, {}, tag, False) for i in range(100)]
        run(which, "gauss", jobs, over)
    elif which == "digits":
        jobs = [(30000 + i, "null", {}, "cal", True) for i in range(300)]
        jobs += [(40000 + i, "null", {}, "eval", True) for i in range(300)]
        for ci, c in enumerate(SHIFTS):
            jobs += [(50000 + 1000 * ci + i, c, {}, "main", True) for i in range(100)]
        jobs += [(60000 + i, "null", {"rate": 0.2}, "null20", True) for i in range(100)]
        for ci, c in enumerate(SHIFTS):
            jobs += [(61000 + 1000 * ci + i, c, {"rate": 0.2}, "main20", True) for i in range(100)]
        run(which, "digits", jobs)
    elif which == "digits_cl":  # covariate+label arm (re-run after a generator fix)
        jobs = [(30000 + i, "null", {}, "cal", True) for i in range(300)]
        jobs += [(50000 + 1000 * 3 + i, "covariate_label", {}, "main", True) for i in range(100)]
        jobs += [(61000 + 1000 * 3 + i, "covariate_label", {"rate": 0.2}, "main20", True) for i in range(100)]
        run(which, "digits", jobs)
    elif which == "covertype":
        jobs = [(30000 + i, "area0_heldout", {}, "cal", True) for i in range(300)]
        jobs += [(40000 + i, "area0_heldout", {}, "eval", True) for i in range(300)]
        for ai, a in enumerate(["area1", "area2", "area3"]):
            jobs += [(50000 + 1000 * ai + i, a, {}, "main", True) for i in range(100)]
        for ai, a in enumerate(["area0_heldout", "area1", "area2", "area3"]):
            jobs += [(60000 + 1000 * ai + i, a, {"rate": 0.2}, "main20", True) for i in range(30)]
        run(which, "covertype", jobs)
    elif which == "electricity":
        jobs = [(30000 + i, "src", {}, "cal", True) for i in range(300)]
        segs = list(range(8000, 44000, 2000))
        for si, s0 in enumerate(segs):
            jobs += [(50000 + 100 * si + i, f"seg{s0}", {}, "main", True) for i in range(20)]
        run(which, "electricity", jobs)
    else:
        raise ValueError(which)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        main(w)
