# Windowed IT2-TOPSIS for fleet condition monitoring (C-MAPSS FD001-FD004)

Title (v3): *Windowed Interval Type-2 TOPSIS for Fleet Condition Monitoring: Corrected Memberships, Incremental Statistics and an Evaluation on Four C-MAPSS Subsets*

## Files
- `paper/main.tex`, `paper/main.pdf`: current paper (v3, 7 pages)
- `paper/main_v2.tex`: round 1 revision (v2); `paper/main_v1_original.tex`, `.pdf`: first submission
- `code/it2_stream_v3.py`: every number, table and figure in v3 (writes `results/metrics_v3.json` and `paper/figs/*_v3.*`, about 6 min on one core). It imports the model from `it2_stream_v2.py`.
- `code/it2_stream_v2.py`: v2 experiment (FD001 only, `results/metrics_v2.json`)
- `code/it2_stream_mcdm.py`, `extra_splits.py`, `fou_flags.py`, `make_figs.py`: v1 scripts
- `data/`: NASA C-MAPSS FD001-FD004 train, test and RUL files (public, from the NASA PCoE data repository)
- `review/Peer_Review_Panel.md`, `review/Peer_Review_Round2.md`: panel reviews of v1 and v2

## Reproduce
```
pip install numpy scipy scikit-learn matplotlib
cd code
python it2_stream_v3.py
cd ../paper && pdflatex main.tex && pdflatex main.tex
```

## What changed in v3 (answers to round 2)
- FD002, FD003, FD004 added (FD002/FD004 normalised per operating condition with k-means on the settings)
- Fault study: AUC, partial AUC (FPR <= 5%), detection at 1% realised false alarms; new detectors: upper membership, mean plus spread, detrended spread ratio; new zero-mean variance fault
- Bootstrap with 2000 replicates and Holm correction; pooled-reference row included
- Test set: age-only baseline, R-precision instead of saturated Precision@10
- Delay: Proposition 4's own bound evaluated; ratios corrected
- Fig. 1 legend and caption made consistent; primary map stated as pre-specified

## Honest summary of results
- Per-engine reference (+0.10 to +0.16 Spearman) and the corrected membership (+0.03 to +0.06 on FD002-FD004) are significant on every subset.
- IT2-TOPSIS: Spearman 0.78-0.85 (train) and 0.78-0.88 (test). It is within 0.05 of the best unsupervised ranker on every subset but never the best; the supervised ridge index is clearly better on FD002-FD004.
- The FOU width is not a better fault detector than a plain detrended spread ratio (mean AUC 0.78 vs 0.94). Its role is an interpretable display of spread on the urgency scale.
