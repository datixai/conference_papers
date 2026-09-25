"""Figures for Paper 5. No titles inside the images; captions live in the paper."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "Paper5_ICAIIT2027_Intent" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c55a11", "#7f7f7f"

# ---------------------------------------------------------------- fig1 pipeline
fig, ax = plt.subplots(figsize=(7.0, 1.8), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
W = 16
boxes = [(1, "Customer\nmessage"), (22, "Sentence encoder\n(MiniLM, frozen)"),
         (43, "Intent head\n+ aspect head"),
         (64, "Handover gate\nconf $\\geq$ t\naspect $\\neq$ other\nheads agree")]
for x, txt in boxes:
    ax.add_patch(FancyBboxPatch((x, 7), W, 17, boxstyle="round,pad=0.4", fc="#eef3f8", ec=C1, lw=0.9))
    ax.text(x + W / 2, 15.5, txt, ha="center", va="center", fontsize=7.2, linespacing=1.3)
for x in (1, 22, 43):
    ax.annotate("", xy=(x + W + 4.2, 15.5), xytext=(x + W + 0.8, 15.5), arrowprops=dict(arrowstyle="->", lw=0.9))
ax.add_patch(FancyBboxPatch((86, 18), 13, 9, boxstyle="round,pad=0.4", fc="#e8f3e8", ec="#2e7d32", lw=0.9))
ax.text(92.5, 22.5, "Auto-route\n{intent, aspect}", ha="center", va="center", fontsize=7)
ax.add_patch(FancyBboxPatch((86, 3), 13, 9, boxstyle="round,pad=0.4", fc="#fbeee6", ec=C2, lw=0.9))
ax.text(92.5, 7.5, "Human queue", ha="center", va="center", fontsize=7)
ax.annotate("", xy=(85.5, 22.5), xytext=(80.8, 18), arrowprops=dict(arrowstyle="->", lw=0.9))
ax.annotate("", xy=(85.5, 7.5), xytext=(80.8, 13), arrowprops=dict(arrowstyle="->", lw=0.9))
ax.text(83.2, 21.8, "pass", fontsize=6.3, ha="right"); ax.text(83.2, 8.3, "fail", fontsize=6.3, ha="right")
ax.text(72, 1.2, "t tuned by leaving known intents out", ha="center", fontsize=6.2, color=C3)
fig.savefig(OUT / "fig1_handover_pipeline.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig2 coverage vs false auto-route
fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
for key, col, lab in (("tfidf_n20", C3, "TF-IDF + LR"), ("minilm_n20", C1, "MiniLM + LR")):
    cur = R["curves"][key]
    ax.plot([c[1] * 100 for c in cur], [c[2] * 100 for c in cur], color=col, lw=1.3, label=lab)
    run = R["runs"][key]
    for pol, mk in (("abstain", "o"), ("abstain_open", "s")):
        s = run[pol]
        ax.plot(s["coverage"] * 100, s["false_auto_rate_of_all"] * 100, mk, color=col, ms=5,
                mfc="white" if pol == "abstain" else col)
ax.plot([], [], "o", color="k", mfc="white", ms=5, label="known-only threshold")
ax.plot([], [], "s", color="k", ms=5, label="leave-intents-out threshold")
ax.axvline(2700 / 4050 * 100, color=C2, lw=0.8, ls="--")
ax.text(2700 / 4050 * 100 + 1, 1, "eligible\nceiling", color=C2, fontsize=6.5)
ax.set_xlabel("Coverage: messages auto-routed (%)")
ax.set_ylabel("False auto-routes (% of all messages)")
ax.set_xlim(0, 101); ax.set_ylim(0, 19)
ax.legend(fontsize=6.3, frameon=False, loc="upper left")
fig.savefig(OUT / "fig2_coverage_risk.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig3 unknown leakage
fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=300)
keys = [("tfidf_n20", "TF-IDF\n20/intent"), ("minilm_n20", "MiniLM\n20/intent"),
        ("tfidf_n100", "TF-IDF\n100/intent"), ("minilm_n100", "MiniLM\n100/intent")]
w = 0.26
for i, (k, _) in enumerate(keys):
    run = R["runs"][k]
    vals = [run["no_abstain"]["unknown_auto_rate"], run["abstain"]["unknown_auto_rate"], run["abstain_open"]["unknown_auto_rate"]]
    for j, (v, col) in enumerate(zip(vals, (C3, C2, C1))):
        ax.bar(i + (j - 1) * w, v * 100, w, color=col)
        ax.text(i + (j - 1) * w, v * 100 + 1.5, f"{v*100:.0f}", ha="center", fontsize=6)
ax.set_xticks(range(len(keys))); ax.set_xticklabels([k[1] for k in keys], fontsize=7)
ax.set_ylabel("Unseen-intent messages\nauto-routed (%)")
ax.set_ylim(0, 112)
for lab, col in (("no abstain", C3), ("known-only threshold", C2), ("leave-intents-out", C1)):
    ax.bar(0, 0, color=col, label=lab)
ax.legend(fontsize=6.3, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.savefig(OUT / "fig3_unknown_leakage.png", bbox_inches="tight"); plt.close(fig)
print("ok", sorted(p.name for p in OUT.iterdir()))
