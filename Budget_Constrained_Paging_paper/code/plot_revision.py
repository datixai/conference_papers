"""Budget sweep figure from results/revision_metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SERIES = [
    ("calib_full", "Train-calibrated threshold", "#E69F00", "s"),
    ("lag_full", "Lagrangian", "#56B4E9", "o"),
    ("lag_full+lex", "Lagrangian + verifier", "#009E73", "^"),
    ("mask_full+lex", "Hard mask + verifier", "#CC79A7", "D"),
    ("hyb_full+lex", "Hybrid + verifier", "#D55E00", "v"),
]
PANELS = [("interrupt", "Realised interrupt rate"), ("tpr", "Window TPR"), ("inc_recall", "Incident recall")]


def main() -> None:
    m = json.loads((ROOT / "results" / "revision_metrics.json").read_text(encoding="utf-8"))
    fs = sorted(float(f) for f in m["sweep"])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), constrained_layout=True)
    for ax, (key, title) in zip(axes, PANELS):
        for name, label, color, marker in SERIES:
            mu = [m["sweep"][str(f)][name][key]["mean"] for f in fs]
            sd = [m["sweep"][str(f)][name][key]["std"] for f in fs]
            ax.errorbar(fs, mu, yerr=sd, color=color, marker=marker, ms=3.5, lw=1.2, capsize=2, label=label)
        if key == "interrupt":
            ax.plot(fs, fs, ls="--", color="black", lw=0.8, label=r"$F_{\max}$")
        ax.set_xlabel(r"Declared budget $F_{\max}$", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(0, 1.02 if key != "interrupt" else 0.32)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, fontsize=6.5, frameon=False)
    out = ROOT / "paper" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_sweep.pdf")
    fig.savefig(out / "fig_sweep.png", dpi=200)


if __name__ == "__main__":
    main()
