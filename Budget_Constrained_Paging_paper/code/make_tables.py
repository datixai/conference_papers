"""Write LaTeX result tables from results/*.json and print numbers quoted in the text."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper"

ROWS = [
    ("Unconstrained", "degree", "Degree"),
    ("", "iforest", "Isolation Forest"),
    ("", "lexical", "Verifier only"),
    ("", "mlp_struct", "Structure MLP"),
    ("", "mlp_full", "Full MLP"),
    ("", "mlp_full+lex", r"Full MLP $\wedge$ verifier"),
    ("Train-calibrated", "calib_full", "Full MLP"),
    ("", "calib_struct", "Structure MLP"),
    ("Lagrangian", "lag_full", "Full MLP"),
    ("", "lag_struct", "Structure MLP"),
    ("", "lag_full+lex", r"Full MLP $\wedge$ verifier"),
    ("Hard mask", "mask_full", "Full MLP"),
    ("", "mask_struct", "Structure MLP"),
    ("", "mask_full+lex", r"Full MLP $\wedge$ verifier"),
    ("", "mask_full+lex@0.35", r"Full MLP $\wedge$ verifier, $\theta_0=0.35$"),
    ("Hybrid", "hyb_full", "Full MLP"),
    ("", "hyb_full+lex", r"Full MLP $\wedge$ verifier"),
    ("Rising edge", "lag_full+lex+edge", r"Lagrangian, full MLP $\wedge$ verifier"),
]
UNSW_ROWS = ["iforest", "mlp_full", "calib_full", "lag_full", "lag_full+lex", "mask_full", "mask_full+lex", "hyb_full", "hyb_full+lex"]
NAMES = {k: (g + " " + n if g and k in UNSW_ROWS else n) for g, k, n in ROWS}


def pm(v):
    return f"\\pmv{{{v['mean']:.3f}}}{{{v['std']:.3f}}}"


def main_table(m: dict) -> str:
    lines = []
    for grp, key, name in ROWS:
        if grp and lines:
            lines.append(r"\midrule")
        v = m["main"][key]
        lines.append(f"{grp} & {name} & {pm(v['tpr'])} & {pm(v['far'])} & {pm(v['interrupt'])} & "
                     f"{v['precision']['mean']:.3f} & {v['inc_recall']['mean']:.3f} & {v['seeds_over_cap']} \\\\")
    return "\n".join(lines) + "\n"


def unsw_table(m: dict) -> str:
    label = {"iforest": "Isolation Forest (unconstr.)", "mlp_full": "Full MLP (unconstr.)", "calib_full": "Train-calibrated",
             "lag_full": "Lagrangian", "lag_full+lex": r"Lagrangian $\wedge$ verifier", "mask_full": "Hard mask",
             "mask_full+lex": r"Hard mask $\wedge$ verifier", "hyb_full": "Hybrid", "hyb_full+lex": r"Hybrid $\wedge$ verifier"}
    lines = []
    for k in UNSW_ROWS:
        v = m["main"][k]
        lines.append(f"{label[k]} & {pm(v['tpr'])} & {pm(v['interrupt'])} & {v['precision']['mean']:.3f} & "
                     f"{v['inc_recall']['mean']:.3f} & {v['seeds_over_cap']} \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    cic = json.loads((RES / "revision_metrics.json").read_text(encoding="utf-8"))
    u20 = json.loads((RES / "revision_metrics_unsw_20.json").read_text(encoding="utf-8"))
    u10 = json.loads((RES / "revision_metrics_unsw_10.json").read_text(encoding="utf-8"))
    (OUT / "tab_main_rows.tex").write_text(main_table(cic), encoding="utf-8")
    (OUT / "tab_unsw_rows.tex").write_text(unsw_table(u20), encoding="utf-8")
    for name, m in (("CIC", cic), ("UNSW20", u20), ("UNSW10", u10)):
        print("==", name, json.dumps(m["data"]))
        print("AUROC", {k: round(v["mean"], 3) for k, v in m["auroc"].items()})
        print("boot", m["bootstrap_seed0_tpr_diff"])
        for f, s in m["sweep"].items():
            print("Fmax", f, {k: (round(v["tpr"]["mean"], 3), round(v["interrupt"]["mean"], 3), v["seeds_over_cap"]) for k, v in s.items()})
        for d, fo in m["day_folds"].items():
            print("fold", d, fo["n_test"], fo["test_pos"], {k: (round(v["tpr"]["mean"], 3), round(v["interrupt"]["mean"], 3), v["seeds_over_cap"]) for k, v in fo["res"].items()})


if __name__ == "__main__":
    main()
