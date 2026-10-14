# Windowed interval type-2 streaming TOPSIS (C-MAPSS FD001)

Title: *A Windowed Interval Type-2 Fuzzy Multi-Criteria Decision Framework for Streaming Multivariate Analytics*

Map-reduce-compatible schedule; measured on one host. Not a Spark/Flink cluster study.

IEEE-style manuscript with a measured single-host experiment. Not a Spark/Flink cluster study.

## Files
- `paper/main.pdf` — compiled paper (5 pages)
- `paper/main.tex` — source
- `code/it2_stream_mcdm.py` — FD001 ranking, baselines, process-pool timing
- `code/extra_splits.py` — early/late split and vectorised scale-up
- `code/make_figs.py` — figures
- `code/fou_flags.py` — per-channel FOU widths and telemetry flags (`results/fou_flags.json`)
- `data/train_FD001.txt` — NASA C-MAPSS FD001 train (public)
- `results/metrics.json`, `results/extra.json` — measured numbers used in the tables

## Reproduce
```
pip install numpy scikit-learn matplotlib
cd code
python it2_stream_mcdm.py
python extra_splits.py
python make_figs.py
cd ../paper && pdflatex main.tex && pdflatex main.tex
```

## What the numbers are
All table entries come from the scripts above. IT2-TOPSIS does not beat mean |z| or Isolation Forest on the 18-batch average. Process pools are slower than one process on 96 engines. Late-life batches (mean RUL ≤ 80) are the only setting with strong rank correlation.
