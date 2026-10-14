# Windowed IT2-TOPSIS for fleet condition monitoring (C-MAPSS FD001)

Title (v2): *Windowed Interval Type-2 TOPSIS for Fleet Condition Monitoring: Corrected Urgency Memberships, Incremental Window Statistics and Uncertainty Flags*

## Files
- `paper/main.tex`, `paper/main.pdf`: revised paper (v2, 6 pages)
- `paper/main_v1_original.tex`, `paper/main_v1_original.pdf`: the first submission, kept for comparison
- `code/it2_stream_v2.py`: every number, table and figure in v2 (writes `results/metrics_v2.json` and `paper/figs/*_v2.*`, about 35 s on one core)
- `code/it2_stream_mcdm.py`, `extra_splits.py`, `fou_flags.py`, `make_figs.py`: v1 scripts (results in `results/metrics.json`, `extra.json`, `fou_flags.json`)
- `data/`: NASA C-MAPSS FD001 train, test and RUL files (public)
- `review/Peer_Review_Panel.md`: three-reviewer panel review of v1

## Reproduce
```
pip install numpy scipy scikit-learn matplotlib
cd code
python it2_stream_v2.py
cd ../paper && pdflatex main.tex && pdflatex main.tex
```

## What changed in v2 (answers to the review)
- Lower membership fixed (distance from 0 to the z-interval); v1 formula was wrong for 34% of cells
- Per-engine healthy reference, 14 sensors (BPR restored), FOU from detrended window spread
- Baselines added: interval TOPSIS, Hotelling T2, PCA-SPE, refitted Isolation Forest, supervised ridge health index
- Held-out test set (100 test engines), engine-block bootstrap, chance level for Precision@10, early/late split for all methods
- Sensitivity sweep (W, H, z*, k), fault injection with calibrated thresholds, packet loss, numerical check of the delay bound
- O(m)-per-record sliding statistics with a record-by-record replay; process-pool table removed
- Proofs repaired, trivial theorem demoted, references updated, ORCID added for A. Ali

## Honest summary of results
IT2-TOPSIS reaches Spearman 0.846 (train) and 0.859 (test). It is significantly better than type-1 TOPSIS and level with interval TOPSIS, Hotelling T2 and Isolation Forest, but mean |z| is still best on train (0.893). FD002-FD004 and N-CMAPSS were not run (data not on this PC).
