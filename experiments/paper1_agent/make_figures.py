"""Figures for Paper 1. No titles inside the images; captions live in the paper."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "Paper1_ICAIIT2027_Agent" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, GREEN, RED = "#1f4e79", "#c55a11", "#7f7f7f", "#2e7d32", "#b71c1c"


def box(ax, x, y, w, h, txt, fc="#eef3f8", ec=BLUE, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, txt=None, dx=0, dy=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9))
    if txt:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, txt, ha="center", fontsize=6.3, color=GREY)


# ---------------------------------------------------------------- fig1 architecture
fig, ax = plt.subplots(figsize=(7.0, 2.6), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
box(ax, 0.5, 16, 12, 9, "Customer\nticket")
box(ax, 17, 16, 15, 9, "LLM planner\n(untrusted)\nJSON tool call")
box(ax, 37, 12, 19, 17, "Validator\n1 parse, tool name\n2 schema, types\n3 evidence span\n4 record ownership\n5 business policy", fs=6.8)
box(ax, 62, 29, 16, 7, "read: auto-run", fc="#e8f3e8", ec=GREEN)
box(ax, 62, 19.5, 16, 7, "update: confirm\nwith diff", fc="#fff6e5", ec=ORANGE)
box(ax, 62, 10, 16, 7, "money / e-mail:\ntyped reason", fc="#fbeee6", ec=RED)
box(ax, 62, 0.5, 16, 7, "blocked:\nescalate to human", fc="#f2f2f2", ec=GREY)
box(ax, 84, 16, 15, 9, "Business API\n+ audit log")
arrow(ax, 13.3, 20.5, 16.6, 20.5)
arrow(ax, 32.6, 20.5, 36.6, 20.5)
for y in (32.5, 23, 13.5):
    arrow(ax, 56.6, 20.5, 61.6, y)
arrow(ax, 56.6, 17, 61.6, 4, "fail", dx=-2.5, dy=-2.5)
for y in (32.5, 23, 13.5):
    arrow(ax, 78.6, y, 83.6, 20.5)
ax.text(47, 36.5, "risk tier assigned only after all checks pass", ha="center", fontsize=6.3, color=GREY)
fig.savefig(OUT / "fig1_architecture.png", bbox_inches="tight"); plt.close(fig)


def outcome_figures(res):
    tags = [t for t in ("phi35", "phi3mini") if t in res]
    names = {"phi35": "Phi-3.5-mini", "phi3mini": "Phi-3-mini"}
    cats = [("correct", "correct action", GREEN), ("wrong_execution", "wrong state change", RED),
            ("failed_call", "failed / illegal call", ORANGE), ("overblock", "correct call blocked", "#e0a96d"),
            ("correct_block", "wrong call blocked", BLUE), ("blocked_other", "unparseable, escalated", "#cfd8dc")]
    fig, ax = plt.subplots(figsize=(3.4, 2.7), dpi=300)
    labels, y = [], 0
    for tag in tags:
        for cond, cname in (("A", "unconstrained"), ("B", "strict validator"), ("B_lenient", "tolerant validator")):
            counts = res[tag][cond]
            left = 0
            for key, _, col in cats:
                v = counts.get(key, 0)
                if v:
                    ax.barh(y, v, left=left, color=col, height=0.7)
                    if v >= 5:
                        ax.text(left + v / 2, y, str(v), ha="center", va="center", fontsize=5.8,
                                color="white" if col not in ("#cfd8dc", "#e0a96d") else "black")
                left += v
            labels.append(f"{names[tag]}\n{cname}")
            y += 1
        y += 0.4
    pos = [i + (i // 3) * 0.4 for i in range(len(labels))]
    ax.set_yticks(pos); ax.set_yticklabels(labels, fontsize=6.3)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Tickets (of 100)")
    for key, name, col in cats:
        if any(res[t][c].get(key) for t in tags for c in ("A", "B", "B_lenient")):
            ax.barh(0, 0, color=col, label=name)
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.42, -0.2))
    fig.savefig(OUT / "fig2_outcomes.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    rp = HERE / "results.json"
    if rp.exists():
        outcome_figures(json.loads(rp.read_text(encoding="utf-8")))
    print(sorted(p.name for p in OUT.iterdir()))
