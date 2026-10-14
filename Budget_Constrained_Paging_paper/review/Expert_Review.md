# Expert Review

**Manuscript:** Operator-Budgeted Threat Correlation on Streamed IP Interaction Graphs
**Author:** A. Mehfooz (single author)
**Format:** IEEEtran journal, about 2,400 words, 2 tables, 2 figures, 30 references
**Reviewed material:** main.tex, main.pdf, code/, results/ (cic_metrics.json, per-seed JSON)
**Date:** 15 October 2026

---

## 1. Summary of the submission

The paper turns CIC-IDS2017 flows into 60-second IP interaction graphs (2,454 windows), scores each window with simple detectors (max out-degree, Isolation Forest, two small MLPs, a port-based "lexical" rule) and then decides whether to page a human analyst under a declared budget F_max = 0.10 (at most 10% of windows may interrupt). Two budget policies are compared: an online Lagrangian threshold and a hard running-rate mask. On a 70/30 time split with 10 seeds, the Lagrangian policy reaches interrupt rate 0.095, TPR 0.883; the hard mask holds 0.092 but TPR falls to 0.377. An LLM verifier is "specified" but not run.

The framing (alert fatigue as a constraint, not a metric afterthought) is valuable and well motivated, and the author is unusually honest about what was not done. However, in its present form the manuscript does not deliver the methods its title, keywords and file names imply, the core empirical claim is undermined by the paper's own Table I, and the evaluation is too small to support journal-level conclusions.

## 2. Recommendation

**Major revision** (for Computers & Security, IEEE TNSM, JISA or IEEE Access).
**Reject and resubmit** if targeted at IEEE TIFS / TDSC / TNNLS level venues.

---

## 3. Major concerns

### M1. The best policy under the cap is not the proposed one
Table I shows the **structure-only MLP, with no budget control at all**, already meets the cap (interrupt 0.098) with **TPR 0.963**, higher than the Lagrangian policy (0.883) at almost identical FAR (0.061 vs 0.062). The paper notes this but does not draw the consequence: on this data the CMDP layer adds nothing over choosing a good feature set. The dual policy must be shown to help over (a) a fixed threshold calibrated on the training split to hit F_max, (b) a top-k-per-hour rule, and (c) the Lagrangian applied to the structure MLP. Without these baselines the main contribution is not demonstrated.

### M2. The CMDP is in name only
No reward J(pi) is defined, no transition model is used, nothing is optimised over occupancy measures, and the "policy" is a scalar threshold `0.5 + 0.08*lambda`. In the code `lambda` is clipped at 0, so the threshold can only rise above 0.5; this is online threshold control (a PI-style rate controller), not CMDP solution. Proposition 3 then states that KKT does not apply to what was measured. Either (a) reframe honestly as "online budgeted thresholding" and cite the relevant literature (online learning with budgets, selective classification, learning to defer, rate-constrained classification such as Cotter et al.), or (b) actually define reward (e.g. +1 per detected incident, -c per false page), solve the CMDP, and compare against CPO/RCPO-style baselines.

### M3. Title, keywords and artefact over-promise
- "Threat correlation": nothing is correlated across windows or alerts.
- Keyword "Temporal graph" and the archive name "TGN": the only temporal state is an EMA of one scalar; no TGN, GNN or node memory is used. The "graph" is reduced to aggregate counts (edges, unique sources, max out-degree), which can be computed without constructing a graph.
- "LLM SOC": no LLM is run; the verifier is a two-quantile port rule.
A journal reader will judge the paper against these labels. Either implement the TGN encoder and the 4-bit LLM verifier, or retitle (e.g. "Budget-Constrained Analyst Paging on Windowed Flow Graphs: An Empirical Study") and remove TGN/LLM from keywords and contributions.

### M4. The evaluation is statistically too thin
- 54 positive windows in total; about **30 in the test split**. One missed window moves TPR by 0.033, so all TPR differences in Table I below about 0.07 are within 2 windows.
- Ten seeds vary only model initialisation on **one fixed split**. They measure optimiser noise, not generalisation (the paper admits this in Limits).
- Required: bootstrap or block-bootstrap CIs over test windows; multiple temporal folds (e.g. leave-one-day-out over the five CIC days); a paired test (McNemar or permutation) for the key comparisons.

### M5. The hard-mask explanation is likely wrong, and the metric is not SOC-relevant
The paper says the mask "spends early interrupts on stub-true benign minutes." A more probable cause is that attack-dominant minutes in CIC-IDS2017 arrive in **contiguous bursts** (a DDoS or PortScan lasts tens of minutes). A running-rate cap of 10% can page at most about one in ten minutes of a burst, so per-window TPR collapses even when every incident is paged. Please:
1. Report the run lengths of positive windows in the test split.
2. Add **incident-level metrics**: incidents detected, time to first page, pages per incident. For a SOC, paging once in minute 1 of a 20-minute DDoS is a full success, yet per-window TPR scores it 0.05.
3. Re-evaluate both policies with incident-level recall. This could reverse the paper's conclusion about the mask.

### M6. Unfair comparison between the two budget policies
In `cmdp_policies.py`, `hard_mask` uses **MLP >= 0.35 AND stub**, while `lagrangian_threshold` uses **MLP alone** starting at 0.5. The text describes the mask as "the MLP wants the page" (implying 0.5). The two policies therefore differ in three things at once (rule, prefilter, verifier). Add the 2x2 ablation: {mask, Lagrangian} x {with stub, without stub}, at the same base threshold, and fix the text to state 0.35.

### M7. Data validity of CIC-IDS2017
- **Timestamp parsing.** `_parse_ts` tries `%m/%d/%Y` first and falls back to day-first; the paper says `dd/mm/YYYY`. CIC dates (3 to 7 July 2017) are ambiguous under both orders. Some CIC CSVs also store afternoon times in 12-hour format without AM/PM. Either error would reorder windows and silently break the "time-ordered" split and the causal claim (Proposition 1). Please show a per-day window histogram and a sanity table of attack types by day/time against the official CIC schedule.
- **Known label errors.** Engelen et al. (2021) and Lanvin et al. (2022) documented mislabelled and duplicated flows in CIC-IDS2017. Use the corrected release or justify not doing so.
- **Provenance.** The parquet comes from a Hugging Face mirror (`Network-Flows/CICIDS_Flow.parquet`). Give the exact repository id, revision hash and row-count reconciliation with the official 2,830,743 flows.
- **Attack mix per split.** With a time split, the test portion is mostly Thursday afternoon / Friday (PortScan, DDoS, Bot). Provide a per-attack-type breakdown; a degree rule catching "every" attack is unsurprising if the test set is almost all volumetric.

### M8. Single dataset, and the stated reason for excluding UNSW-NB15 is incorrect
The Limits section says UNSW-NB15 has no IP addresses. That applies only to the training/testing partition CSVs; the four raw files `UNSW-NB15_1.csv` to `_4.csv` contain `srcip` and `dstip`. A second dataset is expected for a journal paper. Candidates with IPs and timestamps: UNSW-NB15 raw, CSE-CIC-IDS2018, CTU-13, LANL 2015 (host/auth graph), DARPA OpTC.

### M9. No budget trade-off curve
F_max = 0.10 is the only operating point. Provide a TPR (and incident recall) vs F_max curve for all methods over F_max in {0.02, 0.05, 0.10, 0.15, 0.20}, plus sensitivity to `dual_lr`, `lambda_max` and the 0.08 slope. Also report per-seed constraint satisfaction: with mean 0.095 and s.d. 0.023, several seeds exceed the cap, which must be stated in the results, not only in the "not claimed" list.

### M10. Theoretical content is not at journal level
- Proposition 1 is a design statement, not a result.
- Proposition 2 is true by construction (one line).
- Proposition 3 restates textbook CMDP duality and then disclaims it.
- Contribution 4 promises "a token/feature bound for the verifier"; no such proposition exists.
Either add a substantive result (e.g. a regret or constraint-violation bound for the online dual update with bursty arrivals, or an incident-recall bound for the mask as a function of burst length and F_max) or remove the "Invariants" section and the theory contribution.

---

## 4. Minor concerns

1. **Leakage in Isolation Forest normalisation.** `iforest_risk` sets min/max percentiles on `x_all` (train + test). Use training scores only, otherwise Proposition 1 is violated.
2. `lagrangian_threshold` takes `y` as an argument although it is unused; remove it to make the absence of label leakage obvious.
3. Timing: `ms_per_window_trainscore` mixes training and inference time and is not reported. Report inference latency per window separately (a key SOC metric), and later the LLM latency column you mention.
4. MLP: full-batch, 40 epochs, no validation set, no early stopping, `pos_weight` about 70 makes the 0.5 threshold arbitrary. Report calibration (reliability plot or ECE) and PR-AUC; AUROC at 2 to 4% prevalence is optimistic.
5. Report precision and alerts per hour alongside TPR/FAR; analysts experience precision.
6. Isolation Forest `contamination=0.2` is unexplained; its threshold 0.5 on a rescaled score is arbitrary.
7. EMA alpha = 0.7, the 50% labelling rule and the 60 s window size are not ablated. Show at least 30 s / 60 s / 300 s and 25% / 50% / 75%.
8. The EMA carries across overnight gaps between capture days; state this or reset per day.
9. The "discarded any-flow trial" (AUROC 0.56 to 0.60) is reported without a table; either include it or remove it.
10. Figure 1 (TikZ chain of six boxes) carries little information; replace it with an architecture figure showing tiers, state and the budget controller. Figure 2 is fine but add the TPR vs F_max curve (M9).
11. Algorithm 1 runs the Lagrangian and the mask in the same loop and updates `n_int` from both; separate them.
12. Writing style is very terse and telegraphic (many one-clause sentences, numbers packed into the abstract). Journals expect a fuller introduction (problem, gap, approach, findings), a proper discussion and a threats-to-validity section. Shorten the abstract to about 200 words with 3 to 4 headline numbers.
13. "No Llama weights and no BETH rows were used" in the abstract reads as a defensive note; move to Limitations.
14. Section names "Measurements" and "Specified LLM slot" are unconventional; use "Experimental Setup", "Results", "Discussion".
15. Units: "six minutes per hour" equals 0.10 only if one page costs a full window; state the assumed handling time per page.

## 5. References

1. Only about 30 references, few after 2022. Missing key related work:
   - Alert fatigue / SOC studies: Alahmadi et al., "99% False Positives" (USENIX Security 2022); Kokulu et al. (CCS 2019) on SOC issues.
   - Graph-based NIDS / temporal graphs on flows: E-GraphSAGE (Lo et al., NOMS 2022), Euler (King and Huang, NDSS 2023), Jbeil (Khoury et al., 2024), Argus, Kairos (S&P 2024), FLASH (S&P 2024).
   - Budgeted / selective decisions: Cotter et al. rate constraints (JMLR 2019), learning to defer (Mozannar and Sontag 2020), conformal risk control.
   - LLMs in SOC triage (2024 to 2026 work), since the LLM slot is part of the pitch.
   - CIC-IDS2017 label corrections: Engelen et al. (2021), Lanvin et al. (2022).
2. `refs.bib` has 42 entries but the paper uses an inline `thebibliography`; keep one source of truth.
3. `highnam2021beth` is listed but never cited. IEEE requires every reference to be cited.
4. Key `sokalert2022` points to Husák et al. 2020 (situational awareness SoK), which is not an alert-fatigue source; fix key, year and relevance.
5. Incomplete entries: JODIE, EvolveGCN, DeepLog, Drain, NoDoze, HOLMES, SLEUTH, LoRA, QLoRA have truncated titles; Kuhn and Tucker 1951 lacks venue (Proc. 2nd Berkeley Symposium). Add DOIs.

## 6. Reproducibility package

- Strong point: code, per-seed JSON and the window cache are provided.
- Remove stale artefacts from an earlier synthetic version: `graph_tgn.py`, `synth_stream.py`, `run_experiments.py`, `results/metrics.json` (F_max 0.12, 36 test events, synthetic), `results/seeds/`, `figs/fig_bars.*`. Reviewers who open `metrics.json` see numbers that contradict the paper.
- `data/UNSW_NB15_training-set.csv` is shipped but unused.
- Pin package versions in `requirements.txt`, add a single `make all` / script that rebuilds every table and figure, and add a data-availability statement.

---

## 7. What would make this publishable (priority order)

1. Add incident-level metrics and burst analysis (M5); this may be the paper's real finding.
2. Add calibrated-threshold, top-k and structure-MLP+dual baselines (M1) and the 2x2 ablation (M6).
3. Verify timestamps and labels, use a corrected CIC release, report per-attack results (M7).
4. Leave-one-day-out folds and bootstrap CIs (M4); F_max sweep (M9).
5. Add a second dataset with IPs (M8).
6. Either implement TGN + LLM verifier, or retitle and reframe as an empirical budgeted-paging study (M2, M3).
7. Replace trivial propositions with one real bound, or drop the theory claim (M10).
8. Rewrite to journal style, expand related work to 50+ references (Sections 4, 5).

## 8. Scores (1 = poor, 5 = excellent)

| Criterion | Score |
|---|---|
| Relevance and problem framing | 4 |
| Novelty | 2 |
| Technical soundness | 2 |
| Experimental rigour | 2 |
| Reproducibility | 4 |
| Clarity and presentation | 2 |
| References | 2 |
| **Overall** | **Major revision** |

## 9. Confidential note to the editor (draft)

The problem is relevant and the author's transparency about limitations is commendable. However, the manuscript's own results show a baseline (structure-only MLP) that satisfies the budget with higher recall than the proposed policy, the CMDP and TGN/LLM components are nominal, and the test set contains about 30 positive windows on a single split. I recommend major revision, contingent on incident-level evaluation, stronger baselines, multi-fold validation and a second dataset.
