# Budget-Constrained Analyst Paging on Minute-Level IP Interaction Graphs

Online rate control, verifier gating and incident-level evaluation on CIC-IDS2017 and UNSW-NB15.
Author: Asher Mehfooz (University of Kotli Azad Jammu and Kashmir). Target: journal (IEEEtran format).

## Contents

- `paper/main.pdf`, `paper/main.tex`: the manuscript (7 pages). Tables are generated into `paper/tab_*.tex`.
- `paper/figs/fig_sweep.pdf`: budget sweep figure.
- `code/`: window builders, experiments, verification, plotting and table scripts.
- `results/`: cached window features (`*.npz`), metrics for every seed (`revision_metrics*.json`), run logs and the CIC clock check (`cic_check.json`).
- `review/Expert_Review.md`: the reviewer-style critique of the first draft that this revision answers.
- `status.py`: checklist of revision tasks (run `python status.py`).

## Main findings

1. The common CIC-IDS2017 flow export stores afternoon times on a 12-hour clock without AM/PM, so a naive parse puts each afternoon before its morning. `code/cic_windows.py` repairs it; `code/check_cic.py` verifies it against the published attack schedule.
2. Thresholds calibrated on training data exceed a 10% analyst budget by 1.8 to 2.8 times on both datasets; the plain Lagrangian controller also violates it.
3. A hybrid controller (dual threshold driven by the requested paging rate, inside a hard running-rate mask) never exceeds the budget in any seed, budget level or dataset, with window TPR 0.917 on CIC-IDS2017 and 0.818 (all incidents detected) on UNSW-NB15.
4. A port-based verifier helps on scan and flood traffic (CIC-IDS2017) but hurts on exploit-heavy traffic (UNSW-NB15).

## Reproduce

Data are not included (CIC file is 370 MB). Download into `data/Network-Flows/`:

- `CICIDS_Flow.parquet` from Hugging Face `rdpahalavan/CIC-IDS2017` (SHA-256 `e64f6adfbcb7f96b018cf36c731be64ef4de3353ccd7340b190590734a0dda35`)
- `UNSW_Flow.parquet` from Hugging Face `rdpahalavan/UNSW-NB15`

The cached windows in `results/` are enough to rerun every experiment without the raw data.

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
python code/cic_windows.py          # optional, rebuilds results/cic_windows.npz from raw data
python code/unsw_windows.py         # optional, rebuilds results/unsw_windows.npz
python code/check_cic.py            # optional, needs raw CIC data
python code/run_revision.py cic
python code/run_revision.py unsw 0.2
python code/run_revision.py unsw 0.1
python code/make_tables.py
python code/plot_revision.py
```

Then compile `paper/main.tex` with PDFLaTeX twice. All runs take about 3 minutes on a CPU.
