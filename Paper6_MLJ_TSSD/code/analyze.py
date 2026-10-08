"""Turn traces into tables (LaTeX + CSV) and figure data. Usage: python analyze.py"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"
TAB = RES / "tables"
TAB.mkdir(exist_ok=True)
ALPHA = 0.05
NAT = {"DDM": 3.0, "ADWIN": 1.0, "RiskTrack": math.log(20), "Kernel2ST": math.log(20),
       "Clf2ST": math.log(20), "CTM": math.log(20), "WATCH": math.log(20)}
BASE = ["BBSD", "CUSUM", "PH", "DDM", "ADWIN", "RiskTrack", "Kernel2ST", "Clf2ST", "CTM", "WATCH"]
BASE_NAME = {"BBSD": "BBSD-KS [Rabanser19]", "CUSUM": "Classifier-CUSUM [EvansGSell24]",
             "PH": "Page--Hinkley", "DDM": "DDM [Gama04]", "ADWIN": "ADWIN [Bifet07]",
             "RiskTrack": "Risk tracking [PodkopaevRamdas22]", "Kernel2ST": "Kernel 2ST betting [ShekharRamdas23]",
             "Clf2ST": "Seq. classifier 2ST [Jang22]", "CTM": "Conformal TM [Vovk21]",
             "WATCH": "WATCH-style WCTM [Prinster25]"}
ORACLE = json.loads((RES / "real_oracle.json").read_text()) if (RES / "real_oracle.json").exists() else {}
ALLOWED = {"null": {"none"}, "covariate": {"covariate"}, "label": {"label"}, "concept": {"concept"},
           "covariate_label": {"covariate", "label"}, "covariate_concept": {"concept"},
           "label_concept": {"concept"}, "area0_heldout": {"none"}}
for _k, _v in ORACLE.items():
    if _k.startswith("covertype/area") and _k != "covertype/area0_heldout":
        ALLOWED[_k.split("/")[1]] = {_v["oracle"]}
C_NULL_HOLDS = {"null", "covariate", "label", "covariate_label", "area0_heldout"}


def load(name):
    z = np.load(RES / f"traces_{name}.npz")
    meta = json.loads((RES / f"meta_{name}.json").read_text())
    return {k: z[k] for k in z.files}, meta


def first_cross(a, thr):
    """a: (n,L) traces; returns 0-based first index with a>=thr, or -1."""
    hit = np.nan_to_num(a, nan=-np.inf) >= thr
    idx = hit.argmax(1)
    idx[~hit.any(1)] = -1
    return idx


def strict_cross(a, thr):
    hit = np.nan_to_num(a, nan=-np.inf) > thr
    idx = hit.argmax(1)
    idx[~hit.any(1)] = -1
    return idx


def tssd_eval(tr, meta_list, mode="test", thr=None, delta=0.0, ckey="C"):
    keys = {"test": ("X", "L", ckey), "sr": ("X_sr", "L_sr", ckey + "_sr")}[mode]
    th = thr + delta
    cross = {w: first_cross(tr[k], th) for w, k in zip("XLC", keys)}
    out = []
    for i, m in enumerate(meta_list):
        T, t0 = m["T"], m["t0"]
        times = {w: cross[w][i] for w in "XLC" if cross[w][i] >= 0}
        alarm = min(times.values()) if times else -1
        pre = alarm >= 0 and alarm < t0 and m["cond"] not in ("null", "area0_heldout", "src") and t0 > 0
        active = set(times)
        if "C" in active:
            typ = "concept"
        elif "L" in active:
            typ = "label"
        elif "X" in active:
            typ = "covariate"
        else:
            typ = "none"
        out.append(dict(alarm=alarm, pre=pre, det=(alarm >= 0 and not pre), delay=(alarm - t0) if alarm >= 0 else None,
                        typ=typ, active="".join(sorted(active)) or "-", c_cross=("C" in active),
                        first="".join(sorted(w for w, v in times.items() if v == alarm))))
    return out


def base_eval(tr, meta_list, key, thr, strict=True):
    c = (strict_cross if strict else first_cross)(tr[key], thr)
    out = []
    for i, m in enumerate(meta_list):
        a = c[i]; t0 = m["t0"]
        pre = a >= 0 and a < t0 and m["cond"] not in ("null", "area0_heldout", "src") and t0 > 0
        out.append(dict(alarm=a, pre=pre, det=(a >= 0 and not pre), delay=(a - t0) if a >= 0 else None))
    return out


def ensemble_eval(tr, meta_list, thr3):
    """Bonferroni specialist ensemble: BBSD, CUSUM, PH each at alpha/3 (matched); typed by detector."""
    cr = {k: strict_cross(tr[k], thr3[k]) for k in ("BBSD", "CUSUM", "PH")}
    out = []
    for i, m in enumerate(meta_list):
        t0 = m["t0"]
        times = {k: cr[k][i] for k in cr if cr[k][i] >= 0}
        a = min(times.values()) if times else -1
        pre = a >= 0 and a < t0 and m["cond"] not in ("null", "area0_heldout", "src") and t0 > 0
        typ = "concept" if "PH" in times else "label" if "CUSUM" in times else "covariate" if "BBSD" in times else "none"
        out.append(dict(alarm=a, pre=pre, det=(a >= 0 and not pre), delay=(a - t0) if a >= 0 else None, typ=typ))
    return out


def calib_quantile(vals, q):
    return float(np.quantile(np.asarray(vals), q, method="higher"))


def calibrate(tr, idx_cal, thr_test, thr_sr):
    """Matched 5% horizon thresholds from calibration nulls."""
    sel = lambda k: np.nan_to_num(tr[k][idx_cal], nan=-np.inf)  # noqa: E731
    out = {}
    mx = np.max(np.stack([sel("X").max(1), sel("L").max(1), sel("C").max(1)]), 0) - thr_test
    out["tssd_test_delta"] = calib_quantile(mx, 1 - ALPHA)
    mx = np.max(np.stack([sel("X_sr").max(1), sel("L_sr").max(1), sel("C_sr").max(1)]), 0) - thr_sr
    out["tssd_sr_delta"] = calib_quantile(mx, 1 - ALPHA)
    for b in BASE:
        if b in tr:
            out[b] = calib_quantile(sel(b).max(1), 1 - ALPHA)
            out[b + "_a3"] = calib_quantile(sel(b).max(1), 1 - ALPHA / 3)
    return out


def summarize(ev, idx):
    sub = [ev[i] for i in idx]
    n = len(sub)
    det = [e for e in sub if e["det"]]
    d = [e["delay"] for e in det]
    res = dict(n=n, pre=sum(e["pre"] for e in sub), det=len(det),
               delay=float(np.median(d)) if d else float("nan"))
    if sub and "typ" in sub[0]:
        res["types"] = Counter(e["typ"] for e in det)
        res["c_cross"] = sum(e.get("c_cross", False) for e in sub)
    return res


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0, c - h), min(1, c + h)


def fmt_delay(x):
    return "--" if x != x else f"{x:.0f}"


# ============================================================================ main study
def study(name, conds, label):
    tr, meta = load(name)
    ml = meta["meta"]
    thr_t, thr_s = meta["log_thr_test"], meta["log_thr_sr"]
    idx = defaultdict(list)
    for i, m in enumerate(ml):
        idx[(m["tag"], m["cond"])].append(i)
    cal_tag_cond = [k for k in idx if k[0] == "cal"][0]
    eval_keys = [k for k in idx if k[0] == "eval"]
    cal = calibrate(tr, idx[cal_tag_cond], thr_t, thr_s)
    ev = {
        "TSSD-2 (test)": tssd_eval(tr, ml, "test", thr_t),
        "TSSD-2 (SR)": tssd_eval(tr, ml, "sr", thr_s),
        "TSSD-1 concept arm": tssd_eval(tr, ml, "test", thr_t, ckey="Cp"),
        "TSSD-2 (test, matched)": tssd_eval(tr, ml, "test", thr_t, cal["tssd_test_delta"]),
        "TSSD-2 (SR, matched)": tssd_eval(tr, ml, "sr", thr_s, cal["tssd_sr_delta"]),
        "Specialist ensemble (matched)": ensemble_eval(tr, ml, {k: cal[k + "_a3"] for k in ("BBSD", "CUSUM", "PH")}),
    }
    for b in BASE:
        if b in tr:
            ev[b + " (matched)"] = base_eval(tr, ml, b, cal[b])
            if b in NAT:
                ev[b + " (native)"] = base_eval(tr, ml, b, NAT[b], strict=False)
    rows = []
    for meth, e in ev.items():
        row = dict(method=meth)
        for k in eval_keys:
            s = summarize(e, idx[k]); row["null_far"] = s["det"] + s["pre"]; row["null_n"] = s["n"]
        for c in conds:
            key = ("main", c)
            if key not in idx:
                continue
            s = summarize(e, idx[key])
            row[c] = s
        rows.append(row)
    return tr, ml, idx, cal, ev, rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main_tables(name, conds, cond_names, prefix):
    tr, ml, idx, cal, ev, rows = study(name, conds, prefix)
    # --------- detection table
    lines = []
    flat = []
    for r in rows:
        if r["method"].startswith("TSSD-1"):
            continue
        far = r.get("null_far", 0); n = r.get("null_n", 1)
        cells = [r["method"], f"{100 * far / n:.1f}"]
        fr = dict(method=r["method"], null_far=far, null_n=n)
        for c in conds:
            s = r.get(c)
            if s is None:
                cells.append("--"); continue
            cells.append(f"{s['det']}/{s['n']} ({fmt_delay(s['delay'])})")
            fr[c + "_det"] = s["det"]; fr[c + "_n"] = s["n"]; fr[c + "_pre"] = s["pre"]; fr[c + "_delay"] = s["delay"]
        flat.append(fr)
        lines.append(" & ".join(cells) + r" \\")
    (TAB / f"{prefix}_detection.tex").write_text("\n".join(lines))
    write_csv(RES / f"{prefix}_detection.csv", flat, list(flat[0].keys()) + [k for c in conds for k in
                                                                               (c + "_det", c + "_n", c + "_pre", c + "_delay") if k not in flat[0]])
    # --------- typing table (TSSD test, SR, ensemble)
    tl = []
    trows = []
    for meth in ["TSSD-2 (test)", "TSSD-2 (SR)", "TSSD-2 (test, matched)", "Specialist ensemble (matched)"]:
        e = ev[meth]
        cells = [meth]
        for c in conds:
            if ("main", c) not in idx:
                continue
            sub = [e[i] for i in idx[("main", c)]]
            det = [x for x in sub if x["det"]]
            ok = sum(x["typ"] in ALLOWED[c] for x in det)
            cells.append(f"{ok}/{len(sub)}")
            trows.append(dict(method=meth, cond=c, n=len(sub), detected=len(det), correct=ok,
                              types=dict(Counter(x["typ"] for x in det))))
        tl.append(" & ".join(cells) + r" \\")
    (TAB / f"{prefix}_typing.tex").write_text("\n".join(tl))
    (RES / f"{prefix}_typing.json").write_text(json.dumps(trows, indent=1))
    # --------- certified concept claim: false C crossings where H0^C holds
    cc = []
    for meth, ck in [("TSSD-2 C (test)", "C"), ("TSSD-2 C (SR)", "C_sr"), ("TSSD-1 paired C", "Cp")]:
        thr = (json.loads((RES / f"meta_{name}.json").read_text())["log_thr_sr"] if ck.endswith("_sr")
               else json.loads((RES / f"meta_{name}.json").read_text())["log_thr_test"])
        crs = first_cross(tr[ck], thr)
        row = dict(method=meth)
        for (tag, c), ii in idx.items():
            if tag == "cal":
                continue
            row[c] = (int((crs[ii] >= 0).sum()), len(ii))
        cc.append(row)
    (RES / f"{prefix}_concept_claims.json").write_text(json.dumps(cc, indent=1))
    (RES / f"{prefix}_calibration.json").write_text(json.dumps(cal, indent=1))
    return tr, ml, idx, cal, ev, rows


# ============================================================================ sweeps
def sweep_tables():
    out = {}
    # reveal rates (native thresholds; TSSD test & SR; strongest native baselines)
    tr, meta = load("gauss_reveal")
    ml = meta["meta"]
    e_t = tssd_eval(tr, ml, "test", meta["log_thr_test"])
    e_s = tssd_eval(tr, ml, "sr", meta["log_thr_sr"])
    e_p = tssd_eval(tr, ml, "test", meta["log_thr_test"], ckey="Cp")
    bases = {b: base_eval(tr, ml, b, NAT[b], strict=False) for b in ("CTM", "WATCH", "RiskTrack")}
    rows = []
    for rate in (0.02, 0.05, 0.10, 0.20):
        for c in ("null", "covariate", "label", "concept", "label_concept"):
            ii = [i for i, m in enumerate(ml) if abs(m["rate"] - rate) < 1e-9 and m["cond"] == c]
            if not ii:
                continue
            r = dict(rate=rate, cond=c, n=len(ii))
            for nm, e in (("test", e_t), ("sr", e_s), ("tssd1", e_p)):
                s = summarize(e, ii)
                r[nm + "_det"] = s["det"] + (s["pre"] if c == "null" else 0)
                r[nm + "_ok"] = sum(e[i]["det"] and e[i]["typ"] in ALLOWED[c] for i in ii)
                r[nm + "_delay"] = s["delay"]
            for b, e in bases.items():
                s = summarize(e, ii); r[b + "_det"] = s["det"] + (s["pre"] if c == "null" else 0)
            r["post_labels"] = float(np.median([ml[i]["n_post"] for i in ii]))
            rows.append(r)
    write_csv(RES / "gauss_reveal_summary.csv", rows, list(rows[0].keys()))
    out["reveal"] = rows
    # magnitude sweeps with matched thresholds from the main calibration
    trm, metam = load("gauss_main")
    idx_cal = [i for i, m in enumerate(metam["meta"]) if m["tag"] == "cal"]
    cal = calibrate(trm, idx_cal, metam["log_thr_test"], metam["log_thr_sr"])
    tr, meta = load("gauss_magnitude")
    ml = meta["meta"]
    evs = {"TSSD-2 (test)": tssd_eval(tr, ml, "test", meta["log_thr_test"]),
           "TSSD-2 (SR)": tssd_eval(tr, ml, "sr", meta["log_thr_sr"]),
           "Ensemble": ensemble_eval(tr, ml, {k: cal[k + "_a3"] for k in ("BBSD", "CUSUM", "PH")})}
    for b in ("BBSD", "CUSUM", "PH", "Kernel2ST", "Clf2ST", "CTM", "WATCH"):
        evs[b] = base_eval(tr, ml, b, cal[b])
    rows = []
    groups = defaultdict(list)
    for i, m in enumerate(ml):
        lev = m.get("cov_scale", m.get("label_gamma", m.get("concept_frac")))
        groups[(m["cond"], lev)].append(i)
    # add full-strength points from main study
    trm_ev = None
    for (c, lev), ii in sorted(groups.items()):
        r = dict(cond=c, level=lev, n=len(ii))
        for nm, e in evs.items():
            r[nm] = sum(e[i]["det"] for i in ii)
            if nm.startswith("TSSD") or nm == "Ensemble":
                r[nm + "_typed"] = sum(e[i]["det"] and e[i]["typ"] in ALLOWED[c] for i in ii)
        rows.append(r)
    write_csv(RES / "gauss_magnitude_summary.csv", rows, list(rows[0].keys()))
    out["magnitude"] = rows
    # change time
    tr, meta = load("gauss_changetime")
    ml = meta["meta"]
    e_t = tssd_eval(tr, ml, "test", meta["log_thr_test"])
    e_s = tssd_eval(tr, ml, "sr", meta["log_thr_sr"])
    rows = []
    groups = defaultdict(list)
    for i, m in enumerate(ml):
        groups[(m["cond"], m["t0"])].append(i)
    for (c, t0), ii in sorted(groups.items()):
        r = dict(cond=c, t0=t0, n=len(ii))
        for nm, e in (("test", e_t), ("sr", e_s)):
            s = summarize(e, ii)
            r[nm + "_det"] = s["det"]; r[nm + "_pre"] = s["pre"]; r[nm + "_delay"] = s["delay"]
            r[nm + "_typed"] = sum(e[i]["det"] and e[i]["typ"] in ALLOWED[c] for i in ii)
            if c == "null":
                r[nm + "_det"] = sum(e[i]["alarm"] >= 0 for i in ii)
                # false alarms within the first 400 steps (horizon of the SR bound)
                r[nm + "_fa400"] = sum(0 <= e[i]["alarm"] < 400 for i in ii)
        rows.append(r)
    write_csv(RES / "gauss_changetime_summary.csv", rows, sorted({k for r in rows for k in r}, key=lambda k: (k not in ("cond", "t0", "n"), k)))
    out["changetime"] = rows
    # ablation
    tr, meta = load("gauss_ablation")
    ml = meta["meta"]
    e_t = tssd_eval(tr, ml, "test", meta["log_thr_test"])
    e_s = tssd_eval(tr, ml, "sr", meta["log_thr_sr"])
    rows = []
    groups = defaultdict(list)
    for i, m in enumerate(ml):
        groups[(m["tag"], m["cond"])].append(i)
    for tag in dict.fromkeys(m["tag"] for m in ml):
        r = dict(config=tag)
        for c in ("null", "covariate", "label", "concept", "covariate_label", "label_concept"):
            ii = groups[(tag, c)]
            for nm, e in (("test", e_t), ("sr", e_s)):
                if c == "null":
                    r[f"{nm}_null_fa"] = sum(e[i]["alarm"] >= 0 for i in ii)
                    r[f"{nm}_null_n"] = len(ii)
                else:
                    r[f"{nm}_{c}_typed"] = sum(e[i]["det"] and e[i]["typ"] in ALLOWED[c] for i in ii)
            r[f"{c}_n"] = len(ii)
        rows.append(r)
    write_csv(RES / "gauss_ablation_summary.csv", rows, list(rows[0].keys()))
    out["ablation"] = rows
    return out


def rate20_table(name, prefix):
    """Native-threshold TSSD (test, SR) and native anytime-valid baselines on the 20%-label arms."""
    tr, meta = load(name)
    ml = meta["meta"]
    e_t = tssd_eval(tr, ml, "test", meta["log_thr_test"])
    e_s = tssd_eval(tr, ml, "sr", meta["log_thr_sr"])
    bases = {b: base_eval(tr, ml, b, NAT[b], strict=False) for b in ("CTM", "WATCH", "RiskTrack", "Kernel2ST")}
    groups = defaultdict(list)
    for i, m in enumerate(ml):
        if m["tag"] in ("main20", "null20"):
            groups[m["cond"]].append(i)
    rows = []
    for c, ii in groups.items():
        r = dict(cond=c, n=len(ii), post_labels=float(np.median([ml[i]["n_post"] for i in ii])))
        for nm, e in (("test", e_t), ("sr", e_s)):
            null_like = c in ("null", "area0_heldout")
            r[nm + "_det"] = sum((e[i]["alarm"] >= 0) if null_like else e[i]["det"] for i in ii)
            r[nm + "_typed"] = sum(e[i]["det"] and e[i]["typ"] in ALLOWED.get(c, set()) for i in ii)
            r[nm + "_concept"] = sum(e[i]["c_cross"] for i in ii)
            r[nm + "_types"] = dict(Counter(e[i]["typ"] for i in ii if e[i]["det"]))
        for b, e in bases.items():
            r[b + "_det"] = sum(e[i]["alarm"] >= 0 if c in ("null", "area0_heldout") else e[i]["det"] for i in ii)
        rows.append(r)
    (RES / f"{prefix}_rate20.json").write_text(json.dumps(rows, indent=1, default=int))
    return rows


def electricity_table():
    tr, meta = load("electricity")
    ml = meta["meta"]
    idx = defaultdict(list)
    for i, m in enumerate(ml):
        idx[m["cond"]].append(i)
    cal = calibrate(tr, idx["src"], meta["log_thr_test"], meta["log_thr_sr"])
    e_t = tssd_eval(tr, ml, "test", meta["log_thr_test"])
    e_s = tssd_eval(tr, ml, "sr", meta["log_thr_sr"])
    bases = {b: base_eval(tr, ml, b, cal[b]) for b in ("BBSD", "CUSUM", "PH", "Kernel2ST", "CTM", "WATCH")}
    rows = []
    null = dict(segment="source period (null)", oracle="none", gap=float("nan"), auc=float("nan"), n=len(idx["src"]))
    for nm, e in (("test", e_t), ("sr", e_s)):
        null[nm + "_alarm"] = sum(e[i]["alarm"] >= 0 for i in idx["src"])
        null[nm + "_concept"] = sum(e[i]["c_cross"] for i in idx["src"])
    rows.append(null)
    for c, ii in idx.items():
        if c == "src":
            continue
        o = ORACLE.get(f"electricity/{c}", {})
        r = dict(segment=int(c[3:]), oracle=o.get("oracle"), gap=o.get("gap"), auc=o.get("AUC_X"),
                 ll_frozen=o.get("LL_frozen"), ll_prior=o.get("LL_prior"), ll_refit=o.get("LL_refit"), n=len(ii))
        for nm, e in (("test", e_t), ("sr", e_s)):
            al = [e[i]["alarm"] for i in ii if e[i]["alarm"] >= 0]
            r[nm + "_alarm"] = len(al)
            r[nm + "_delay"] = float(np.median(al)) if al else float("nan")
            r[nm + "_concept"] = sum(e[i]["typ"] == "concept" for i in ii)
            r[nm + "_label"] = sum(e[i]["typ"] == "label" for i in ii)
            r[nm + "_covariate"] = sum(e[i]["typ"] == "covariate" for i in ii)
            r[nm + "_agree"] = sum(e[i]["typ"] == o.get("oracle") for i in ii)
        for b, e in bases.items():
            r[b + "_alarm"] = sum(e[i]["alarm"] >= 0 for i in ii)
        rows.append(r)
    for b, e in bases.items():
        rows[0][b + "_alarm"] = sum(e[i]["alarm"] >= 0 for i in idx["src"])
    write_csv(RES / "electricity_summary.csv", rows, sorted({k for r in rows for k in r}, key=lambda k: (k != "segment", k)))
    return rows


FIG = RES.parent / "figures"


def _write_dat(path, header, rows):
    with open(path, "w") as f:
        f.write(" ".join(header) + chr(10))
        for r in rows:
            f.write(" ".join("nan" if (isinstance(v, float) and v != v) else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + chr(10))


def figure_data():
    FIG.mkdir(exist_ok=True)
    # ---- magnitude curves (typed-correct fraction for TSSD/ensemble, detection fraction for baselines)
    trm, metam = load("gauss_main")
    mlm = metam["meta"]
    idx_cal = [i for i, m in enumerate(mlm) if m["tag"] == "cal"]
    cal = calibrate(trm, idx_cal, metam["log_thr_test"], metam["log_thr_sr"])
    full = {
        "TSSD-SR": tssd_eval(trm, mlm, "sr", metam["log_thr_sr"]),
        "TSSD-test": tssd_eval(trm, mlm, "test", metam["log_thr_test"]),
        "Ensemble": ensemble_eval(trm, mlm, {k: cal[k + "_a3"] for k in ("BBSD", "CUSUM", "PH")}),
    }
    for b in ("BBSD", "CUSUM", "PH", "Kernel2ST", "CTM", "WATCH"):
        full[b] = base_eval(trm, mlm, b, cal[b])
    tr, meta = load("gauss_magnitude")
    ml = meta["meta"]
    ev = {"TSSD-SR": tssd_eval(tr, ml, "sr", meta["log_thr_sr"]),
          "TSSD-test": tssd_eval(tr, ml, "test", meta["log_thr_test"]),
          "Ensemble": ensemble_eval(tr, ml, {k: cal[k + "_a3"] for k in ("BBSD", "CUSUM", "PH")})}
    for b in ("BBSD", "CUSUM", "PH", "Kernel2ST", "CTM", "WATCH"):
        ev[b] = base_eval(tr, ml, b, cal[b])
    names = list(ev)
    for cond, key in (("covariate", "cov_scale"), ("label", "label_gamma"), ("concept", "concept_frac")):
        groups = defaultdict(list)
        for i, m in enumerate(ml):
            if m["cond"] == cond:
                groups[m[key]].append(i)
        rows = []
        for lev in sorted(groups):
            ii = groups[lev]
            rows.append([lev] + [sum((e[i]["det"] and e[i]["typ"] in ALLOWED[cond]) if n in ("TSSD-SR", "TSSD-test", "Ensemble")
                                     else e[i]["det"] for i in ii) / len(ii) for n, e in ev.items()])
        ii = [i for i, m in enumerate(mlm) if m["tag"] == "main" and m["cond"] == cond]
        rows.append([1.0] + [sum((full[n][i]["det"] and full[n][i]["typ"] in ALLOWED[cond]) if n in ("TSSD-SR", "TSSD-test", "Ensemble")
                                 else full[n][i]["det"] for i in ii) / len(ii) for n in names])
        _write_dat(FIG / f"mag_{cond}.dat", ["level"] + [n.replace("-", "") for n in names], rows)
    # ---- reveal rate (merge 5% from main)
    trr, metar = load("gauss_reveal")
    mlr = metar["meta"]
    evr = {"t": tssd_eval(trr, mlr, "test", metar["log_thr_test"]), "s": tssd_eval(trr, mlr, "sr", metar["log_thr_sr"]),
           "p": tssd_eval(trr, mlr, "test", metar["log_thr_test"], ckey="Cp"),
           "ctm": base_eval(trr, mlr, "CTM", NAT["CTM"], strict=False)}
    evm = {"t": full["TSSD-test"], "s": full["TSSD-SR"], "p": tssd_eval(trm, mlm, "test", metam["log_thr_test"], ckey="Cp"),
           "ctm": base_eval(trm, mlm, "CTM", NAT["CTM"], strict=False)}
    rows = []
    for rate in (0.02, 0.05, 0.10, 0.20):
        r = [rate]
        for cond in ("label", "concept", "label_concept"):
            if rate == 0.05:
                src_ml, e = mlm, evm
                ii = [i for i, m in enumerate(mlm) if m["tag"] == "main" and m["cond"] == cond]
            else:
                src_ml, e = mlr, evr
                ii = [i for i, m in enumerate(mlr) if m["cond"] == cond and abs(m["rate"] - rate) < 1e-9]
            for k in ("t", "s", "p"):
                r.append(sum(e[k][i]["det"] and e[k][i]["typ"] in ALLOWED[cond] for i in ii) / len(ii))
            r.append(sum(e["ctm"][i]["det"] for i in ii) / len(ii))
        rows.append(r)
    _write_dat(FIG / "reveal.dat", ["rate", "lab_t", "lab_s", "lab_p", "lab_ctm", "con_t", "con_s", "con_p", "con_ctm",
                                    "lc_t", "lc_s", "lc_p", "lc_ctm"], rows)
    # ---- example traces: first main label_concept stream
    i = [j for j, m in enumerate(mlm) if m["tag"] == "main" and m["cond"] == "label_concept"][0]
    T = mlm[i]["T"]
    rows = [[t + 1] + [float(trm[k][i, t]) for k in ("X", "L", "C", "X_sr", "L_sr", "C_sr")] for t in range(T)]
    _write_dat(FIG / "trace.dat", ["t", "X", "L", "C", "Xsr", "Lsr", "Csr"], rows)

    # ---- change time
    if not (RES / "meta_gauss_changetime.json").exists():
        print("changetime not available yet")
        return
    tr, meta = load("gauss_changetime")
    ml = meta["meta"]
    e_t = tssd_eval(tr, ml, "test", meta["log_thr_test"])
    e_s = tssd_eval(tr, ml, "sr", meta["log_thr_sr"])
    rows = []
    for t0 in (100, 500, 1000, 2000):
        r = [t0]
        for cond in ("covariate", "label", "concept"):
            ii = [i for i, m in enumerate(ml) if m["cond"] == cond and m["t0"] == t0]
            for e in (e_t, e_s):
                d = [e[i]["delay"] for i in ii if e[i]["det"]]
                r += [float(np.median(d)) if d else float("nan"), sum(e[i]["det"] for i in ii) / len(ii)]
        rows.append(r)
    _write_dat(FIG / "changetime.dat", ["t0", "cov_test_delay", "cov_test_det", "cov_sr_delay", "cov_sr_det",
                                        "lab_test_delay", "lab_test_det", "lab_sr_delay", "lab_sr_det",
                                        "con_test_delay", "con_test_det", "con_sr_delay", "con_sr_det"], rows)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["gauss"]
    if "gauss" in which:
        main_tables("gauss_main", ["covariate", "label", "concept", "covariate_label", "covariate_concept", "label_concept"], None, "gauss")
        sweep_tables()
    if "figures" in which:
        figure_data()
    if "digits" in which:
        main_tables("digits", ["covariate", "label", "concept", "covariate_label", "covariate_concept", "label_concept"], None, "digits")
        rate20_table("digits", "digits")
    if "covertype" in which:
        main_tables("covertype", ["area1", "area2", "area3"], None, "covertype")
        rate20_table("covertype", "covertype")
    if "electricity" in which:
        electricity_table()
    print("done")


def reveal_rows():
    """LaTeX rows for the label-budget table (5% taken from the main study)."""
    trm, metam = load("gauss_main"); mlm = metam["meta"]
    trr, metar = load("gauss_reveal"); mlr = metar["meta"]
    out = []
    names = {"null": "null (any alarm)", "covariate": "Cov", "label": "Lab", "concept": "Con", "label_concept": "L+C"}
    for rate in (0.02, 0.05, 0.10, 0.20):
        if rate == 0.05:
            tr, meta, ml = trm, metam, mlm
            sel = lambda m, c: m["cond"] == c and m["tag"] in (("eval",) if c == "null" else ("main",))  # noqa: E731
        else:
            tr, meta, ml = trr, metar, mlr
            sel = lambda m, c, r=rate: m["cond"] == c and abs(m["rate"] - r) < 1e-9  # noqa: E731
        ev = [tssd_eval(tr, ml, "sr", meta["log_thr_sr"]), tssd_eval(tr, ml, "test", meta["log_thr_test"]),
              tssd_eval(tr, ml, "test", meta["log_thr_test"], ckey="Cp"), base_eval(tr, ml, "CTM", NAT["CTM"], strict=False)]
        first = True
        for c in ("null", "covariate", "label", "concept", "label_concept"):
            ii = [i for i, m in enumerate(ml) if sel(m, c)]
            cells = []
            for k, e in enumerate(ev):
                if c == "null":
                    v = sum(e[i]["alarm"] >= 0 for i in ii)
                elif k == 3:
                    v = sum(e[i]["det"] for i in ii)
                else:
                    v = sum(e[i]["det"] and e[i]["typ"] in ALLOWED[c] for i in ii)
                cells.append(f"{v}/{len(ii)}")
            out.append((f"{int(rate * 100)}\%" if first else "") + f" & {names[c]} & " + " & ".join(cells) + r" \\")
            first = False
        out.append(r"\addlinespace")
    (TAB / "reveal_rows.tex").write_text("\n".join(out[:-1]))


def merge_digits_cl():
    """Replace the Digits covariate+label arm by the re-run after the generator fix."""
    tr, meta = load("digits")
    tc, mc = load("digits_cl")
    keep = [i for i, m in enumerate(meta["meta"]) if m["cond"] != "covariate_label"]
    add = [i for i, m in enumerate(mc["meta"]) if m["tag"] in ("main", "main20")]
    out = {}
    for k in tr:
        a, b = tr[k][keep], tc[k][add]
        L = max(a.shape[1], b.shape[1])
        out[k] = np.vstack([np.pad(a, ((0, 0), (0, L - a.shape[1])), constant_values=np.nan),
                            np.pad(b, ((0, 0), (0, L - b.shape[1])), constant_values=np.nan)])
    meta["meta"] = [meta["meta"][i] for i in keep] + [mc["meta"][i] for i in add]
    meta["merged_from"] = "digits + digits_cl (covariate_label arm re-run after generator fix)"
    np.savez_compressed(RES / "traces_digits.npz", **out)
    (RES / "meta_digits.json").write_text(json.dumps(meta, default=float))


def real_rows():
    # ---------------- digits
    tr, meta = load("digits"); ml = meta["meta"]
    idx = defaultdict(list)
    for i, m in enumerate(ml):
        idx[(m["tag"], m["cond"])].append(i)
    cal = calibrate(tr, idx[("cal", "null")], meta["log_thr_test"], meta["log_thr_sr"])
    conds = ["covariate", "label", "concept", "covariate_label", "covariate_concept", "label_concept"]
    meths05 = [("TSSD, SR mode", tssd_eval(tr, ml, "sr", meta["log_thr_sr"]), True),
               ("TSSD, test mode", tssd_eval(tr, ml, "test", meta["log_thr_test"]), True),
               ("Specialist ensemble", ensemble_eval(tr, ml, {k: cal[k + "_a3"] for k in ("BBSD", "CUSUM", "PH")}), True),
               ("Classifier-CUSUM", base_eval(tr, ml, "CUSUM", cal["CUSUM"]), False),
               ("BBSD-KS", base_eval(tr, ml, "BBSD", cal["BBSD"]), False),
               ("Kernel 2ST betting", base_eval(tr, ml, "Kernel2ST", cal["Kernel2ST"]), False),
               ("Page--Hinkley", base_eval(tr, ml, "PH", cal["PH"]), False),
               ("Conformal TM (native)", base_eval(tr, ml, "CTM", NAT["CTM"], strict=False), False),
               ("WATCH-style (native)", base_eval(tr, ml, "WATCH", NAT["WATCH"], strict=False), False)]

    def cell(e, ii, typed, c):
        return sum((e[i]["det"] and e[i]["typ"] in ALLOWED[c]) if typed else e[i]["det"] for i in ii)

    lines = []
    for nm, e, typed in meths05:
        lines.append(nm + " & " + " & ".join(str(cell(e, idx[("main", c)], typed, c)) for c in conds) + r" \\")
    (TAB / "digits_rows05.tex").write_text("\n".join(lines))
    lines = []
    for nm, e, typed in [meths05[0], meths05[1], meths05[7], meths05[8]]:
        lines.append(nm + " & " + " & ".join(str(cell(e, idx[("main20", c)], typed, c)) for c in conds) + r" \\")
    (TAB / "digits_rows20.tex").write_text("\n".join(lines))
    # ---------------- oracle
    lines = []
    names = {"covertype/area0_heldout": "Covertype area 0 (held-out)", "covertype/area1": "Covertype area 1",
             "covertype/area2": "Covertype area 2", "covertype/area3": "Covertype area 3",
             "electricity/src": "Elec2 source period"}
    for k, o in ORACLE.items():
        if k.startswith("electricity/seg"):
            s0 = int(k.split("seg")[1])
            if s0 not in (8000, 12000, 16000, 18000, 24000, 34000):
                continue
            nm = f"Elec2 segment {s0}"
        else:
            nm = names[k]
        lines.append(f"{nm} & {o['acc_frozen']:.2f} & {o['LL_frozen']:.2f} & {o['LL_prior']:.2f} & {o['LL_refit']:.2f} & "
                     f"{o['gap']:.2f} & {o['AUC_X']:.2f} & {o['oracle']} \\\\")
    (TAB / "oracle_rows.tex").write_text("\n".join(lines))


def ablation_rows():
    tr, meta = load("gauss_ablation"); ml = meta["meta"]
    e_t = tssd_eval(tr, ml, "test", meta["log_thr_test"]); e_s = tssd_eval(tr, ml, "sr", meta["log_thr_sr"])
    g = defaultdict(list)
    for i, m in enumerate(ml):
        g[(m["tag"], m["cond"])].append(i)
    label = {"win25": "kernel window 25", "win100": "kernel window 100", "decay70": "confusion discount 0.70",
             "decay95": "confusion discount 0.95", "gate1": "gates 1 / 2 (test / SR)", "gate3": "gates 3 / 4 (test / SR)",
             "beta1": r"tempering grid $B=\{1\}$", "betawide": r"$B=\{0.5,0.7,1,1.4,2\}$", "emfull": "EM on full history",
             "altbasic": "basic two-component alternative", "box4": r"null box $|u_k|\le4$"}
    lines = []
    for tag, nm in label.items():
        cells = []
        for c in ("null", "covariate", "label", "concept", "covariate_label", "label_concept"):
            ii = g[(tag, c)]
            if c == "null":
                cells.append(f"{sum(e_s[i]['alarm'] >= 0 for i in ii)} / {sum(e_t[i]['alarm'] >= 0 for i in ii)}")
            else:
                cells.append(f"{sum(e_s[i]['det'] and e_s[i]['typ'] in ALLOWED[c] for i in ii)} / "
                             f"{sum(e_t[i]['det'] and e_t[i]['typ'] in ALLOWED[c] for i in ii)}")
        fc = sum(e_s[i]["c_cross"] or e_t[i]["c_cross"] for c in ("null", "covariate", "label", "covariate_label") for i in g[(tag, c)])
        lines.append(nm + " & " + " & ".join(cells) + f" & {fc} \\\\")
    (TAB / "ablation_rows.tex").write_text("\n".join(lines))


def elec_rows():
    rows = electricity_table()
    lines = []
    for r in rows:
        if r["segment"] == "source period (null)":
            lines.append(r"Source period (null) & -- & -- & " + f"{r['sr_alarm']}/{r['n']} & {r['test_alarm']}/{r['n']} & "
                         f"0 / 0 / -- & {r['PH_alarm']} & {r['CTM_alarm']} & {r['WATCH_alarm']} & {r['BBSD_alarm']} \\\\")
            continue
        lines.append(f"{r['segment']} & {r['oracle']} & {r['gap']:.2f} & {r['sr_alarm']}/{r['n']} & {r['test_alarm']}/{r['n']} & "
                     f"{r['sr_concept']} / {r['sr_label']} / {r['sr_covariate']} & {r['PH_alarm']} & {r['CTM_alarm']} & "
                     f"{r['WATCH_alarm']} & {r['BBSD_alarm']} \\\\")
    (TAB / "elec_rows.tex").write_text("\n".join(lines))
    return rows


def changetime_rows():
    tr, meta = load("gauss_changetime"); ml = meta["meta"]
    e_t = tssd_eval(tr, ml, "test", meta["log_thr_test"]); e_s = tssd_eval(tr, ml, "sr", meta["log_thr_sr"])
    lines = []
    for t0 in (100, 500, 1000, 2000):
        cells = [str(t0)]
        for c in ("covariate", "label", "concept"):
            ii = [i for i, m in enumerate(ml) if m["cond"] == c and m["t0"] == t0]
            for e in (e_t, e_s):
                ok = [i for i in ii if e[i]["det"] and e[i]["typ"] in ALLOWED[c]]
                d = [e[i]["delay"] for i in ii if e[i]["det"]]
                cells.append(f"{len(ok)} ({fmt_delay(float(np.median(d)) if d else float('nan'))})")
        lines.append(" & ".join(cells) + r" \\")
    (TAB / "changetime_rows.tex").write_text("\n".join(lines))
    ii = [i for i, m in enumerate(ml) if m["cond"] == "null"]
    info = dict(n=len(ii), test_any=sum(e_t[i]["alarm"] >= 0 for i in ii), sr_any=sum(e_s[i]["alarm"] >= 0 for i in ii),
                sr_400=sum(0 <= e_s[i]["alarm"] < 400 for i in ii), test_400=sum(0 <= e_t[i]["alarm"] < 400 for i in ii),
                c_claims=sum(e_s[i]["c_cross"] or e_t[i]["c_cross"] for i in ii),
                c_claims_shift=sum(e_s[i]["c_cross"] or e_t[i]["c_cross"] for i, m in enumerate(ml) if m["cond"] in ("covariate", "label")))
    (RES / "changetime_null.json").write_text(json.dumps(info, default=int))
    return info


def verify_concept_claims():
    """Count concept claims (either mode, any time) on every stream whose posterior is prior-correctable."""
    tot = {"streams": 0, "claims": 0}
    per = {}
    safe = {"null", "covariate", "label", "covariate_label", "area0_heldout", "src"}
    for name in ("gauss_main", "gauss_reveal", "gauss_magnitude", "gauss_ablation", "gauss_changetime", "digits",
                 "covertype", "electricity"):
        if not (RES / f"meta_{name}.json").exists():
            continue
        tr, meta = load(name); ml = meta["meta"]
        ct = first_cross(tr["C"], meta["log_thr_test"]); cs = first_cross(tr["C_sr"], meta["log_thr_sr"])
        ii = [i for i, m in enumerate(ml) if m["cond"] in safe]
        if name == "digits":  # the original covariate_label arm was a null stream; merged file is used
            pass
        n, c = len(ii), int(sum((ct[i] >= 0) or (cs[i] >= 0) for i in ii))
        per[name] = (c, n); tot["streams"] += n; tot["claims"] += c
    (RES / "concept_claim_audit.json").write_text(json.dumps({"per_experiment": per, "total": tot}, indent=1))
    return per, tot
