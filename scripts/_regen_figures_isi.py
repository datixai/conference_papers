"""Publication figures for ISI 2027. No titles on images. Captions live only in the paper."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "Paper3_ISI2027_Informed" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#14293F"
TEAL = "#1A5674"
GOLD = "#7A5E16"
RED = "#7A2E2E"
GREEN = "#1F4F38"
GRAY = "#334155"
LIGHT = "#F4F7FB"
WHITE = "#FFFFFF"
LINE = "#1B3348"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": NAVY,
    "axes.edgecolor": NAVY,
    "figure.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "savefig.dpi": 300,
})


def rounded(ax, x, y, w, h, fc=LIGHT, ec=LINE, lw=1.7):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.05",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    )
    ax.add_patch(patch)
    return patch


def box(ax, x, y, w, h, title, body="", fc=LIGHT, ec=LINE, tc=NAVY, title_size=12.5, body_size=11.0):
    rounded(ax, x, y, w, h, fc=fc, ec=ec)
    cx = x + w / 2
    if body:
        ax.text(cx, y + h * 0.68, title, ha="center", va="center",
                fontsize=title_size, color=tc, fontweight="bold", zorder=3)
        ax.text(cx, y + h * 0.32, body, ha="center", va="center",
                fontsize=body_size, color=tc, linespacing=1.35, zorder=3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, color=tc, fontweight="bold",
                linespacing=1.35, zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=13, linewidth=1.6,
        color=LINE, zorder=1, shrinkA=0, shrinkB=0,
    ))


def canvas(w, h, pad=(0.18, 0.18, 0.18, 0.18)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def fig1():
    fig, ax = canvas(12.2, 3.35)
    steps = [
        (0.15, "1. Theory", "Buckland, Wilson\nKuhlthau, Lloyd"),
        (2.12, "2. Model", "Informing\nepisode"),
        (4.09, "3. Instrument", "18-document corpus\nand six tasks"),
        (6.06, "4. Validation", "BM25 retrieval\nwalkthrough"),
        (8.03, "5. Protocol", "Think-aloud script\nfor later reuse"),
    ]
    for x, title, body in steps:
        box(ax, x, 1.7, 1.82, 6.6, title, body, title_size=13, body_size=11)
    for x in (1.97, 3.94, 5.91, 7.88):
        arrow(ax, x, 5.0, x + 0.15, 5.0)
    save(fig, "fig1_research_design.png")


def fig2():
    fig, ax = canvas(12.2, 4.55)
    nodes = [
        (0.12, "Workplace\ntask"),
        (1.78, "Query to\nthe RAG box"),
        (3.44, "Retrieved\nthings"),
        (5.10, "Generated\nutterance"),
        (6.76, "Worker\nuptake"),
        (8.42, "Informing\nothers"),
    ]
    for x, title in nodes:
        box(ax, x, 5.35, 1.46, 3.9, title, title_size=12.2)
    for x in (1.58, 3.24, 4.90, 6.56, 8.22):
        arrow(ax, x, 7.3, x + 0.20, 7.3)
    outcomes = [
        (0.45, "Informed", "#DCEFE4", GREEN),
        (2.95, "Misinformed", "#F6E4E4", RED),
        (5.45, "Uninformed", "#F3EBD4", GOLD),
        (7.95, "Overtrust / repair", LIGHT, NAVY),
    ]
    for x, title, fc, ec in outcomes:
        box(ax, x, 0.55, 1.70, 2.9, title, fc=fc, ec=ec, tc=ec, title_size=12.4)
    save(fig, "fig2_informing_episode.png")


def fig3():
    fig, ax = canvas(12.2, 4.55)
    box(ax, 0.18, 5.55, 2.20, 3.55, "Worker query", title_size=13)
    box(ax, 2.72, 5.55, 2.35, 3.55, "BM25 retriever", "lexical, inspectable", title_size=12.6, body_size=11)
    box(ax, 5.40, 5.55, 2.35, 3.55, "Top-k snippets", "dates and versions", title_size=12.6, body_size=11)
    box(ax, 8.08, 5.55, 1.74, 3.55, "Optional\nutterance", title_size=12.6)
    arrow(ax, 2.38, 7.32, 2.72, 7.32)
    arrow(ax, 5.07, 7.32, 5.40, 7.32)
    arrow(ax, 7.75, 7.32, 8.08, 7.32)
    box(
        ax, 1.55, 0.45, 6.90, 3.85,
        "Northline corpus  (18 records)",
        "current SOPs   |   superseded SOPs   |   marketing copy\nprice lists   |   tickets   |   memos   |   changelog",
        title_size=13, body_size=11.2,
    )
    arrow(ax, 3.90, 5.55, 3.90, 4.30)
    save(fig, "fig3_rag_architecture.png")


def fig4():
    fig, ax = canvas(12.2, 4.65)
    items = [
        (0.15, 5.35, "T1  Factoid", "One current SOP\nshould suffice", GREEN),
        (2.60, 5.35, "T2  Conflict", "14-day SOP versus\n30-day web copy", RED),
        (5.05, 5.35, "T3  Stale", "8% fee now versus\n15% superseded", GOLD),
        (7.50, 5.35, "T4  Missing", "No international\nwarranty record", TEAL),
        (1.40, 0.45, "T5  Inform others", "Customer sentence\nmust not hide conflict", NAVY),
        (5.75, 0.45, "T6  Repair", "Challenge a fluent\n30-day answer", NAVY),
    ]
    for x, y, title, body, ec in items:
        box(ax, x, y, 2.25, 4.20, title, body, fc=WHITE, ec=ec, title_size=12.8, body_size=11.1)
    save(fig, "fig4_task_battery.png")


def fig5():
    fig, ax = plt.subplots(figsize=(11.4, 4.7))
    labels = ["T1\nFactoid", "T2\nConflict", "T3\nStale", "T4\nMissing", "T5\nInform", "T6\nRepair"]
    vals = [10.05, 6.34, 10.56, 9.66, 3.93, 7.78]
    colors = [TEAL, RED, GOLD, NAVY, TEAL, GREEN]
    bars = ax.bar(labels, vals, color=colors, width=0.58, edgecolor=WHITE, linewidth=0.6)
    ax.set_ylabel("BM25 score", fontsize=13, color=GRAY, labelpad=8)
    ax.tick_params(colors=GRAY, labelsize=12)
    ax.set_ylim(0, 13.5)
    ax.set_xlim(-0.6, 5.6)
    for spine in ax.spines.values():
        spine.set_color("#C5CDD6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.28, f"{v:.2f}",
            ha="center", va="bottom", fontsize=12, color=NAVY, fontweight="bold",
        )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.16)
    fig.savefig(OUT / "fig5_retrieval_scores.png", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print("OK", sorted(p.name for p in OUT.glob("*.png")))
