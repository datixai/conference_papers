"""ICAIIT figures. No titles on the images. Captions live only in the paper."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#14293F"
TEAL = "#1A5674"
RED = "#7A2E2E"
GREEN = "#1F4F38"
GOLD = "#7A5E16"
GRAY = "#334155"
LIGHT = "#F4F7FB"
WHITE = "#FFFFFF"
LINE = "#1B3348"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": NAVY,
    "figure.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "savefig.dpi": 300,
})


def rounded(ax, x, y, w, h, fc=LIGHT, ec=LINE):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.05",
        facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=2,
    ))


def box(ax, x, y, w, h, title, body="", fc=LIGHT, ec=LINE, tc=NAVY):
    rounded(ax, x, y, w, h, fc=fc, ec=ec)
    cx = x + w / 2
    if body:
        ax.text(cx, y + h * 0.68, title, ha="center", va="center",
                fontsize=11.5, color=tc, fontweight="bold", zorder=3)
        ax.text(cx, y + h * 0.32, body, ha="center", va="center",
                fontsize=10.2, color=tc, zorder=3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=11.5, color=tc, fontweight="bold", zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.5, color=LINE, zorder=1, shrinkA=0, shrinkB=0,
    ))


def fig1():
    fig, ax = plt.subplots(figsize=(11.6, 3.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    steps = [
        (0.12, "Query", "desk question"),
        (2.10, "BM25", "top-3 things"),
        (4.08, "Slot extract", "days, fee, scope"),
        (6.06, "Conflict test", "values disagree?"),
        (8.04, "Answer / refuse", "current wins"),
    ]
    for x, title, body in steps:
        box(ax, x, 2.2, 1.82, 5.7, title, body)
    for x in (1.94, 3.92, 5.90, 7.88):
        arrow(ax, x, 5.0, x + 0.16, 5.0)
    fig.savefig(OUT / "fig1_refusal_pipeline.png", bbox_inches="tight", pad_inches=0.14)
    plt.close()


def fig2():
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    labels = ["Always\nrank-1", "Conflict\nrefuse", "Current\noverride"]
    wrong = [4, 0, 0]
    bars = ax.bar(labels, wrong, color=[RED, GOLD, GREEN], width=0.55, edgecolor=WHITE)
    ax.set_ylabel("Wrong answers out of 6 tasks", fontsize=12, color=GRAY)
    ax.set_ylim(0, 6.4)
    ax.tick_params(colors=GRAY, labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#C5CDD6")
    for b, v in zip(bars, wrong):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v),
                ha="center", va="bottom", fontsize=12, color=NAVY, fontweight="bold")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.16)
    fig.savefig(OUT / "fig2_wrong_answers.png", bbox_inches="tight", pad_inches=0.12)
    plt.close()


def fig3():
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    labels = ["T1", "T2", "T3", "T5", "T6"]
    base = [30, 30, 15, 14, 14]
    gold = [14, 14, 8, 14, 14]
    x = range(len(labels))
    w = 0.36
    ax.bar([i - w / 2 for i in x], base, width=w, color=RED, label="Rank-1 extract")
    ax.bar([i + w / 2 for i in x], gold, width=w, color=TEAL, label="Governing value")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Extracted slot value", fontsize=12, color=GRAY)
    ax.tick_params(colors=GRAY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(0, 36)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.14)
    fig.savefig(OUT / "fig3_slot_values.png", bbox_inches="tight", pad_inches=0.12)
    plt.close()


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("OK", sorted(p.name for p in OUT.glob("*.png")))
