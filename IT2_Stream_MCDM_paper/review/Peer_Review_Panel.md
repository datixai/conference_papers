# Peer Review Panel Report

**Manuscript:** A Windowed Interval Type-2 Fuzzy Multi-Criteria Decision Framework for Streaming Multivariate Analytics
**Authors:** M. S. Raja, A. Mehfooz, N. Yaqoob, A. Ali
**Format:** IEEEtran journal, 5 pages, code and data supplied
**Assumed venue:** an IEEE fuzzy systems / computational intelligence journal (e.g. IEEE Trans. Fuzzy Systems, IEEE TETCI, or a broader venue such as IEEE Access)
**Review date:** 15 October 2026

---

## 1. The panel

| Reviewer | Expertise | Focus in this review |
|---|---|---|
| **R1** | Type-2 fuzzy sets, fuzzy MCDM (TOPSIS, VIKOR, AHP) | Is the model really IT2? Are the propositions correct and new? |
| **R2** | Prognostics and health management (PHM), C-MAPSS, predictive maintenance | Is the experiment sound, fair and meaningful for maintenance practice? |
| **R3** | Stream processing systems and ML evaluation methodology | Is the "streaming / map-reduce" contribution real? Are the statistics valid? |

All three reviewers read the PDF and LaTeX source. R3 also re-ran parts of the supplied code (`it2_stream_mcdm.py`) to check two specific concerns; those checks are reported in Section 3.

---

## 2. Summary of the submission

The paper defines, for each engine and sensor, a window-based interval `[mean - std, mean + std]`, standardises it against a pooled healthy prefix, maps `|z|` through a linear ramp saturating at `z* = 3`, and treats the resulting `[LMF, UMF]` pair as an interval type-2 membership. Engines are ranked by an IT2 TOPSIS closeness coefficient (after Chen and Lee, 2010). The authors give four propositions (work/depth, collapse to type-1, Lipschitz closeness, delay bound) and a correctness theorem, and argue the computation is "map-reduce compatible" because the ideal solutions are min/max reductions. On C-MAPSS FD001 (training set, 18 cycle batches) IT2-TOPSIS reaches Spearman rho = 0.683, equal to type-1 TOPSIS (0.682) and below mean |z| (0.709) and Isolation Forest (0.708). The authors argue the value lies in the FOU interval, which they use to raise channel and engine "telemetry flags".

The paper is unusually candid about its limits (no cluster run, baselines are as good or better, early-life ranking is poor). The panel valued that honesty. The question is whether what remains is a journal-level contribution.

---

## 3. Panel discussion (transcript of the reviewer meeting)

**Chair (R3):** Let's start with the headline. The method does not beat its baselines, there is no distributed run, and the dataset is one simulated subset. So the contribution has to stand on the model and the theory. R1, does it?

**R1:** I have three problems with the model itself.

First, calling this "interval type-2" is a stretch. What they have is an interval-valued membership grade per cell: two numbers, `mean - std` and `mean + std`, pushed through a ramp. There is no IT2 fuzzy set over a domain, no LMF and UMF functions, no type reduction, no Karnik-Mendel. Chen and Lee's IT2 TOPSIS works on trapezoidal IT2 fuzzy numbers with ranking values; this paper uses only the endpoints. Mathematically it is crisp-interval TOPSIS, which is exactly Jahanshahloo et al. (2006) and Dymova et al. (2013), both of which the authors cite and then do not compare against or distinguish from.

Second, the membership map has a real error. The lower membership is defined as `min(|z_lo|, |z_hi|) / z*`. That is only the minimum of `|z|` over the interval when the interval does not contain zero. If `z_lo < 0 < z_hi`, the minimum of `|z|` is 0, so the LMF should be 0. As written, an engine that is sitting right on its healthy mean but has some spread gets a positive lower urgency.

**R3:** I checked how often that happens. On the 18 evaluation batches, 5,573 of 16,406 cells (34%) have a z-interval that straddles zero. So the bug is not a corner case: a third of the LMF entries are wrong, and so are the FOU widths and therefore the flags built on them.

**R1:** That alone requires the experiments to be redone. Third, the propositions. Proposition 2 (collapse to type-1) and parts (i)-(iii) of the theorem are one-line facts: min and max are associative, distances are non-negative, the flags are functions of the widths. Labelling them a "Theorem of correctness" oversells them. Proposition 3 has the right final constant but the proof is incomplete: it ignores that the ideals `v*` and `v°` also move under the perturbation (each residual can move by 2 delta, not delta), it applies a gradient bound that needs `d* + d° >= gamma` along the whole segment, not only at the unperturbed point, and it drops the epsilon. These are fixable, but a fuzzy-systems journal will expect a clean proof.

**R2:** On Proposition 4 I have a modelling objection. It assumes each raw sensor path is L-Lipschitz in time. C-MAPSS sensors are dominated by measurement noise; consecutive samples jump by several standard deviations of noise, so any realistic L makes the bound vacuous. Also, a reordering of samples inside a window does not change the mean or the std at all, so the "out-of-order packets" story in Remark 1 is trivially true and needs no Lipschitz argument; the only non-trivial case is a whole-window shift, which is just "the score at t minus tau", not a watermark property.

**R3:** And the bound is never evaluated numerically. They have the data: they could estimate L, sigma_min and gamma on FD001 and show whether the bound is tight or useless. Without that it is a decoration.

**Chair:** R2, the experiment.

**R2:** Several issues, roughly in order of severity.

1. **No method advantage, and the claimed advantage is not tested.** The paper says the value is the FOU interval. But nothing validates the interval. FD001 has no injected sensor faults, so there is no ground truth for "noisy or thin-sample channel". The flag thresholds (0.50 and 0.65) are set at roughly the 90th percentile of the same run, so a ~15% flag rate is produced by construction. The paper shows that flags exist, not that they mean anything.

2. **The FOU conflates noise with degradation.** The window std contains the within-window trend. Late in life, when a sensor is drifting fast, the std grows because of the drift, not because the channel is unreliable. So "wide FOU = noisy channel" is not established and may be backwards near failure.

3. **"Mean plus or minus one std" is not an uncertainty statement.** It is not a confidence interval for the window mean (that would scale with std / sqrt(W)), nor a calibrated prediction interval. If the selling point is uncertainty, the uncertainty should have a stated meaning and be checked for coverage.

4. **Other methods do give intervals.** The paper repeatedly says a z-score or an Isolation Forest score "does not return" an interval. A bootstrap over the window, a conformal interval, or a Gaussian-process health index all return intervals cheaply. The claimed unique selling point is therefore not unique, and those are the baselines the FOU should be compared with.

5. **Weak baselines for PHM.** Mean |z|, Isolation Forest and type-1 TOPSIS are reasonable sanity checks, but a PHM reader will ask for Mahalanobis distance or Hotelling T-squared on the healthy envelope, PCA-based SPE, a simple linear health index, and at least one RUL model used as a ranker (an RUL regressor induces a ranking directly). Without these, the reader cannot place the method.

6. **Sensor choice.** The code comment says seven sensors are constant on FD001, which leaves 14 informative sensors, but only 13 are used; sensor 15 (BPR) is dropped without explanation. Also, `|z|` throws away direction: several FD001 sensors decrease with degradation, and a move in the "healthy" direction is scored as urgent.

7. **Pooled healthy reference.** Engines in C-MAPSS start with different initial wear and manufacturing variation. A pooled healthy mean and std across engines mixes between-engine offset into "urgency". A per-engine baseline is the standard choice and should at least be tested.

8. **Only FD001, only the training file.** FD002 and FD004 (six operating conditions) are where a fixed healthy envelope would break, and they are exactly the cases a streaming, drift-aware framework claims to handle. N-CMAPSS (2021) is the more realistic modern benchmark. One single-condition simulated subset is not enough for a journal paper that claims a "framework for streaming multivariate analytics".

9. **No sensitivity analysis.** W, H, z*, the weights and the two thresholds are all fixed. The UMF saturates at 1 for many cells (several of the reported example intervals end at 1.000), which suggests z* = 3 is limiting discrimination. A sweep is needed.

**R3:** Let me add the evaluation-methodology problems.

1. **The batches are not independent samples.** Windows of length 20 at a step of 10 overlap by half, the same engines appear in consecutive batches, and the panel shrinks from 100 to 17 engines through survivorship. Reporting mean plus or minus s.d. over 18 batches and calling IT2 and type-1 "statistically indistinguishable" has no test behind it. Use a paired test or a block bootstrap over engines.

2. **Precision@10 is partly undefined and needs a chance baseline.** I computed the share of engines with RUL <= 30 in each batch. At t = 80 and t = 90 no engine has RUL <= 30, so Precision@10 is 0 for every possible method; those batches pull the pooled mean down and make the "early" split look worse than it is. Across the late batches the chance level is about 0.4 to 0.57, so the reported late Precision@10 of 0.90 should be presented against that. The early/late split is also only reported for IT2; the same split for every baseline is needed, otherwise Table II says nothing comparative.

3. **NDCG mismatch.** The paper states the graded gain is 1/(1+R); the code uses an exponential gain 2^rel - 1 with rel = 1/(1+R). Minor, but the text and code must agree.

4. **The noise-robustness test is too weak.** Noise of 0.05 sigma is added to window means only (the std, which drives the FOU, is untouched), on one batch, three seeds. A Kendall tau of 0.98 under that is expected and says little. A meaningful test would perturb raw samples, include dropouts and missing values (the paper's own motivation), and cover all batches.

5. **The streaming contribution is thin.** The "map-reduce compatible schedule" is the observation that column-wise min and max are associative. That holds for every TOPSIS variant ever published; it is not new and does not need a section. There is no incremental update between windows (each window is recomputed from scratch), no handling of event time, no state management, no stream engine. The title and abstract promise "streaming analytics"; the experiment is an offline replay of a static file.

6. **Table III should be removed.** The latency table measures process-pool start-up: `ProcessPoolExecutor` is created inside `rank_batch` on every call, so the 300 ms is mostly spawning processes on Windows. The paper itself spends a paragraph saying the table means nothing. A table that the authors tell readers not to interpret should not be in a journal paper. Fig. 3 (scaling of a vectorised O(nm) kernel on random matrices) is also expected and does not need a figure; at most one sentence.

**Chair:** Presentation?

**R1:** The writing is clear sentence by sentence but defensive. "Not a Spark cluster" or equivalent appears five or six times; the word "authorised" is used in an unusual way ("authorised parameter", "authorised width of disagreement") and will confuse readers. The abstract is too long and spends space on what the paper does not do. The intro motivates cost/safety/throughput trade-offs, but the 13 "criteria" are just 13 sensors with equal weights, so the MCDM framing (and the AHP/VIKOR discussion) is not exercised by the experiment.

**R2:** References are dated. The newest is 2018. For a 2026 submission there is nothing on recent IT2/IVIF MCDM, on N-CMAPSS, on transformer or attention-based RUL models, or on uncertainty quantification in PHM (conformal, Bayesian deep learning). One citation is also misused: Liao, Xu and Zeng (2015) is a hesitant fuzzy linguistic VIKOR paper, but it is cited as evidence about "published stream MCDM". It has nothing to do with streams.

**R1:** Small things: the author block uses `\IEEEmembership{Corresponding Author,}`, which prints as a membership grade; use a footnote instead. The bibliography key `wu2007` points to a 2002 paper (the entry itself is correct). ORCID iDs are missing for all authors.

**Chair:** Time to decide. Positions?

**R1:** Reject. The core IT2 claim does not hold up (it is interval TOPSIS), the LMF has a bug affecting a third of the cells, and the propositions are either trivial or need repair. I would encourage resubmission, but this is not a revision-sized gap.

**R2:** I lean reject as well. For a PHM audience the paper gives no reason to use the method: no accuracy gain, and the claimed benefit (the FOU) is not validated against anything. If the authors could show that the FOU detects injected sensor faults better than a bootstrap interval on mean |z|, I would see a paper. That is a new experiment, not a revision.

**R3:** I was initially at major revision because the code is open and the honesty is rare. But after checking the code, the membership bug changes every IT2 number in the paper, the streaming part is not implemented, and the statistics need redoing. I agree with reject, with a clear invitation to resubmit.

**Chair:** Consensus: **Reject in current form, encourage resubmission as a new manuscript.** For a lower-tier or broad venue (e.g. IEEE Access, Applied Soft Computing) the panel would rate it **Major Revision** at best, conditional on items M1 to M6 below.

---

## 4. Decision

| | R1 (Fuzzy/IT2) | R2 (PHM) | R3 (Streams/Eval) | **Panel** |
|---|---|---|---|---|
| Recommendation | Reject | Reject | Reject (was Major Rev.) | **Reject, resubmission encouraged** |
| Novelty | 2/5 | 2/5 | 1/5 | **2/5** |
| Technical soundness | 2/5 | 2/5 | 2/5 | **2/5** |
| Experimental rigour | 2/5 | 2/5 | 2/5 | **2/5** |
| Clarity | 3/5 | 3/5 | 3/5 | **3/5** |
| Reproducibility | 4/5 | 4/5 | 4/5 | **4/5** |
| Confidence | High | High | High | |

**Why reject:**
1. The model is crisp-interval TOPSIS presented as interval type-2; the distinction from Jahanshahloo/Dymova is not made, so the fuzzy contribution is not new.
2. The lower-membership formula is wrong whenever the z-interval contains zero (34% of cells in the authors' own run), which invalidates the reported IT2 results and FOU flags.
3. The method does not improve ranking, and the claimed benefit (the FOU interval and flags) is never validated against ground truth or against other interval-producing baselines.
4. The streaming / map-reduce contribution amounts to the associativity of min and max; no incremental computation or stream system is implemented.
5. Evaluation on a single simulated subset with non-independent batches, no significance testing and no sensitivity analysis.

**What the panel liked:** honest reporting of negative results, public data plus runnable code, a clear and well-motivated operator view (ranking plus an explanation per channel), and a clean algorithm description.

---

## 5. Consolidated comments to the authors

### Major (must be addressed in any resubmission)

**M1. Fix the membership map.** For `z_lo < 0 < z_hi`, set `LMF = 0`. In general use `LMF = clip(dist(0, [z_lo, z_hi]) / z*, 0, 1)` and `UMF = clip(max(|z_lo|, |z_hi|) / z*, 0, 1)`. Re-run every experiment and update Tables I-II, the flag statistics and all worked examples.

**M2. Position the model honestly against interval and IT2 TOPSIS.** Either (a) call it interval-valued fuzzy / interval TOPSIS and compare against Jahanshahloo et al. and Dymova et al., or (b) build a genuine IT2 model (e.g. IT2 membership functions over the z-domain with an FOU derived from data, type reduction via Karnik-Mendel or Nie-Tan, Chen-Lee ranking values) and show what it adds.

**M3. Validate the FOU and the flags.** Inject controlled sensor faults (noise bursts, bias, stuck-at, dropouts) into FD001 or use a dataset with labelled sensor faults, then report detection rate and false-alarm rate of the FOU flags versus (i) a bootstrap interval on mean |z|, (ii) a conformal or quantile interval on a health index, (iii) a simple rolling-variance detector. Separate within-window trend from noise (e.g. use the std of detrended residuals) so the FOU means what the paper says it means. Choose thresholds on held-out engines, not on the evaluation run.

**M4. Strengthen the experimental design.**
- Add FD002-FD004 and preferably N-CMAPSS.
- Add PHM baselines: Mahalanobis / Hotelling T-squared, PCA-SPE, a linear health index, and one RUL regressor used as a ranker.
- Use per-engine and pooled healthy baselines and report both.
- Report the early/late split for all methods, add a random-ranker baseline for Precision@10, and drop or flag batches where no engine has RUL <= 30.
- Use paired significance tests or a block bootstrap over engines; do not treat overlapping batches as independent.
- Sensitivity analysis over W, H, z*, the weights and the thresholds.
- Explain the sensor set (why sensor 15 is excluded) and handle sensor direction instead of `|z|` only.

**M5. Repair or reduce the theory.** Rewrite Proposition 3's proof to account for the moving ideals, the epsilon, and the gamma condition on both the perturbed and unperturbed matrices. Replace the Lipschitz-path assumption in Proposition 4 with something that tolerates noise (e.g. a bound on the drift of the window mean and std, or a probabilistic bound), and evaluate the bound numerically on FD001. Demote trivial statements (Prop. 2, Theorem parts i-iii) to remarks.

**M6. Either implement streaming or drop the claim.** A real streaming contribution would include incremental (O(m) per arrival) updates of window mean and std (Welford or sliding sums), incremental maintenance of the ideals under window expiry (min/max under deletion is not trivial and is a genuine problem), event-time handling, and a run on Flink, Spark Structured Streaming or a comparable engine with throughput and latency measurements. Otherwise retitle the paper as an interval-valued fuzzy ranking method for condition monitoring and remove the map-reduce section and Table III.

### Minor

1. Remove Table III (pool start-up cost) and reduce Fig. 3 to a sentence.
2. Shorten the abstract; state what is done, not what is not done. Reduce the repeated "not a Spark cluster" disclaimers to one sentence in Limitations.
3. Replace the unusual use of "authorised" with "user-specified" or "fixed".
4. Align the NDCG gain in the text with the code.
5. Strengthen the noise test (perturb raw samples, include missing values, all batches).
6. Either use unequal, elicited weights with real cost/safety criteria, or tone down the multi-criteria motivation.
7. Fix the Liao et al. (2015) citation, which is not about streaming MCDM; rename the `wu2007` key.
8. Update the literature to 2020-2026 (IT2/IVIF MCDM, N-CMAPSS, attention/transformer RUL models, uncertainty quantification in PHM).
9. Fix the author block: use a footnote for the corresponding author, and add ORCID iDs (e.g. Ahmed Ali, 0009-0000-1885-9225).
10. Add a data and code availability statement with a permanent link (Zenodo / GitHub DOI).

---

## 6. Short letter to the editor

> The panel recommends **rejection with encouragement to resubmit**. The manuscript applies interval TOPSIS to windowed sensor data and reports results honestly, but (i) the "interval type-2" model reduces to crisp-interval TOPSIS that is already in the literature, (ii) the lower-membership formula is incorrect for about a third of the matrix entries in the authors' own experiment, (iii) the method does not outperform simple baselines and its claimed benefit (the FOU interval) is not validated, and (iv) the streaming/map-reduce contribution is not implemented beyond the associativity of min and max. The code is available and the work could become a useful contribution after the changes listed in Section 5, but these exceed a normal revision.
