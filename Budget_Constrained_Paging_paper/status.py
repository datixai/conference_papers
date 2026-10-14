"""Progress check for the LLM_SOC_TGN_CMDP revision.

Run:  python status.py
Each task is checked against the files on disk, not a hand-kept list.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
TEX = WORK / "paper" / "main.tex"
PDF = WORK / "paper" / "main.pdf"
RES = WORK / "results" / "revision_metrics.json"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


tex = read(TEX)
res = json.loads(read(RES)) if RES.exists() else {}
STALE = ["code/graph_tgn.py", "code/synth_stream.py", "code/run_experiments.py", "results/metrics.json", "paper/figs/fig_bars.pdf"]

TASKS = [
    ("Setup", "Project folder, original paper extracted", (ROOT / "original").exists()),
    ("Setup", "Expert review written", (ROOT / "review" / "Expert_Review.md").exists()),
    ("Setup", "Virtual environment with libraries", (ROOT / ".venv" / "Scripts" / "python.exe").exists()),
    ("Experiments", "12-hour clock bug repaired (M7)", "Timedelta(hours=12)" in read(WORK / "code" / "cic_windows.py")),
    ("Experiments", "Isolation Forest leakage fixed (minor 1)", "train-only" in read(WORK / "code" / "risk_models.py")),
    ("Experiments", "Incident-level metrics and burst lengths (M5)", "test_incident_lengths" in res.get("data", {})),
    ("Experiments", "Calibrated-threshold baselines (M1)", "calib_full" in res.get("main", {})),
    ("Experiments", "2x2 mask/Lagrangian x verifier ablation (M6)", "mask_full+lex@0.35" in res.get("main", {})),
    ("Experiments", "F_max sweep (M9)", len(res.get("sweep", {})) >= 5),
    ("Experiments", "Rolling-origin day folds (M4)", "Fri" in res.get("day_folds", {})),
    ("Experiments", "Block-bootstrap intervals (M4)", "bootstrap_seed0_tpr_diff" in res),
    ("Experiments", "Budget sweep figure", (WORK / "paper" / "figs" / "fig_sweep.pdf").exists()),
    ("Cleanup", "Stale synthetic files removed", not any((WORK / s).exists() for s in STALE)),
    ("Paper", "New title (no TGN/LLM over-claim) (M3)", "Budget-Constrained" in tex),
    ("Paper", "CMDP reframed as online rate control (M2)", "rate-constrained" in tex.lower()),
    ("Paper", "Mask incident proposition with proof (M10)", "\\begin{proof}" in tex),
    ("Paper", "Results rewritten with repaired clock", "fig_sweep" in tex and "incident recall" in tex.lower()),
    ("Paper", "References expanded and fixed (>= 40)", tex.count("\\bibitem") >= 40),
    ("Paper", "No em/en dashes in text", bool(tex) and not re.search("[\u2013\u2014]", tex) and "---" not in tex),
    ("Paper", "PDF compiled after last edit", PDF.exists() and TEX.exists() and PDF.stat().st_mtime >= TEX.stat().st_mtime),
    ("Data", "CIC parquet downloaded, clock repair verified on raw data (M7)", (WORK / "results" / "cic_check.json").exists()),
    ("Data", "Second dataset with IPs: UNSW-NB15 raw flows (M8)", (WORK / "results" / "revision_metrics_unsw_20.json").exists()),
    ("Experiments", "Hybrid controller (dual inside mask) evaluated", "hyb_full+lex" in res.get("main", {})),
    ("Paper", "UNSW-NB15 results and table in paper", "tab:unsw" in tex),
    ("Optional", "Real TGN encoder and 4-bit LLM verifier (GPU)", (WORK / "code" / "tgn_encoder.py").exists()),
]


def main() -> None:
    done = sum(ok for _, _, ok in TASKS)
    group = None
    for g, name, ok in TASKS:
        if g != group:
            print(f"\n[{g}]")
            group = g
        print(f"  {'[x]' if ok else '[ ]'} {name}")
    pct = 100 * done / len(TASKS)
    bar = "#" * int(pct // 5) + "." * (20 - int(pct // 5))
    print(f"\nProgress: {done}/{len(TASKS)} tasks  [{bar}] {pct:.0f}%")
    left = [n for _, n, ok in TASKS if not ok]
    if left:
        print("Remaining:")
        for n in left:
            print("  -", n)


if __name__ == "__main__":
    main()
