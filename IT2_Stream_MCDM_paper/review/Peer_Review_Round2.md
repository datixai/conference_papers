# Peer Review Panel Report: Round 2 (revised manuscript)

**Manuscript:** Windowed Interval Type-2 TOPSIS for Fleet Condition Monitoring: Corrected Urgency Memberships, Incremental Window Statistics and Uncertainty Flags (v2)
**Authors:** M. S. Raja, A. Mehfooz, N. Yaqoob, A. Ali
**Previous decision:** Reject, resubmission encouraged (round 1, see `Peer_Review_Panel.md`)
**Review date:** 15 October 2026

Same panel as round 1: **R1** (type-2 fuzzy sets and fuzzy MCDM), **R2** (prognostics and C-MAPSS), **R3** (stream processing and evaluation methodology). R3 re-ran `code/it2_stream_v2.py` (35 s) and checked the reported numbers against `results/metrics_v2.json`.

---

## 1. How the round 1 points were handled

| Round 1 item | Status | Panel comment |
|---|---|---|
| M1 lower-membership bug | **Resolved** | Distance-to-zero definition, 34% of cells quantified, old formula kept as an ablation. |
| M2 IT2 versus interval TOPSIS | **Resolved** | Interval-valued view stated openly (Bustince et al.), Jahanshahloo interval TOPSIS added as a baseline and found equivalent. |
| M3 validate FOU flags | **Partly resolved** | Fault injection with thresholds set on separate engines. Concerns about matched false-alarm rates and what the width measures remain (Section 3). |
| M4 experimental design | **Mostly resolved** | Held-out test set, 11 rankers plus a supervised reference, block bootstrap, chance levels, early/late split for all methods, sensitivity, per-engine reference, BPR restored. FD002 to FD004 and N-CMAPSS still missing. |
| M5 theory | **Resolved** | Proposition 3 proof now handles moving ideals and epsilon; Proposition 4 rests on measurable window-statistic drift; trivial items demoted to a remark; bound checked numerically. |
| M6 streaming claim | **Resolved by scoping** | O(m) sliding update with an exact replay; claims toned down; no distributed engine claimed. |
| Minor items 1 to 10 | **Resolved** | Pool table removed, abstract rewritten, "authorised" gone, NDCG text matches code, Liao citation fixed, references updated, ORCID added. |

---

## 2. Panel discussion (transcript)

**Chair (R3):** This is a different paper. The honesty is still there, but now it rests on a correct model and a real evaluation. The question is no longer "is it sound" but "is it enough".

**R1:** On the fuzzy side I am satisfied with soundness. The membership is now correct, the interval-valued framing is honest, and the Lipschitz proof is right: the product-form identity for C minus C' gives the bound with only S' needing a lower bound, which is cleaner than what I asked for. My remaining concern is novelty. Strip away the careful engineering and the ranking core is interval TOPSIS with a specific membership ramp, and Table I says it is statistically indistinguishable from Jahanshahloo's interval TOPSIS. The paper admits this. So the fuzzy-methodology contribution is the membership definition plus the detrended FOU. For IEEE Transactions on Fuzzy Systems that is too thin. For an applied venue it is acceptable if the FOU part is convincing.

**R2:** Which brings us to the FOU. The fault-injection study is a real improvement, but I have three problems with how it is read.

First, the headline "detects noise bursts twice as often as a rolling variance ratio (0.37 versus 0.17)" compares detectors at thresholds that were matched on the calibration engines, but on the evaluation engines the FOU's false-alarm rate is 0.013 against 0.007 for the variance ratio. Roughly double the false alarms for double the detections is not clearly a better operating point. They need ROC curves or AUC, or at least detection at equal realised false-alarm rate on the evaluation engines.

Second, the paper explains the advantage by the shape of the membership: the lower membership drops to zero when the interval reaches the healthy value, and the upper one saturates at 1. That means the FOU width depends on the mean deviation as well as the spread. So the FOU flag is partly an anomaly score, not purely a "this channel is unreliable" signal. The fair comparison is a detector that also combines mean and spread, for example the upper membership alone, or `|z| + k r` on the raw scale. Without that, we cannot tell whether the IT2 construction adds anything or whether any mean-plus-spread statistic would do the same.

Third, a noise burst on the last 10 readings also shifts the window mean, which helps exactly the detector that mixes mean and spread. An experiment with zero-mean faults that do not move the mean (for example a variance increase over the whole window) would separate the two effects.

**R3:** Agreed, and I'd add that the bias-step result cuts against the paper's own design choice: the detrended FOU, which the paper uses for ranking and noise flags, detects 11% of bias steps, while the raw-spread FOU detects 46%. The paper handles this honestly by recommending both widths, but then the "detrended FOU" is a ranking choice, not a flagging improvement, and the abstract should say so.

**R2:** On the prognostics side, two more points. The test-set evaluation is good to have, but Precision@10 is 1.0 for every method, so it carries no information there. And ranking at the last observed cycle mixes engines of very different ages. I would like an age-only baseline.

**R3:** I ran it. Ranking test engines by their observed number of cycles gives Spearman 0.601 against negative RUL. So the methods (0.85 to 0.86) are well above it and the result stands, but the baseline belongs in Table III, because a reader will ask. Per-engine references also interact with age: an engine observed for only 31 cycles has a window that overlaps its own reference period, so it looks healthy and is ranked low, which is usually correct because short-observed engines have high RUL. That is fine but should be stated.

**R3:** Statistics. The block bootstrap is the right tool. Two refinements: 300 replicates is low for two-sided p-values near 0.05 (the directional-map interval is [0.001, 0.065]); use at least 2,000. And with 10 comparisons against the proposed method, some correction (Holm) or at least a sentence on multiplicity is expected. The two conclusions that matter, better than type-1 TOPSIS and worse than mean |z|, will very likely survive both.

**R1:** One factual slip: the abstract and Section VI-G say the delay bound is "13 to 18 times" larger than the observed change. From `metrics_v2.json` the ratios are 18.1, 14.3 and 11.6 for tau = 1, 2, 5, so it should read "12 to 18 times". Also, the numerical check uses delta measured directly from the membership endpoints, which tests Proposition 3. Proposition 4's own constant, built from D^m and D^r, is never evaluated; it would be looser still, and should be reported.

**R2:** And the scope is still one simulated subset with one operating condition. The per-engine reference is the single biggest gain in the paper (0.686 to 0.846), and it is exactly the part that will be stressed by FD002 and FD004, where six operating conditions shift every sensor. That is the experiment I most want to see. It is a known, standard normalisation step (cluster by operating setting, normalise per cluster), so it is a revision-sized request, not a new paper.

**Chair:** Smaller items?

**R1:** Fig. 1 legend says "IT2 v1 (original)" while the caption says "earlier membership formula within the new pipeline"; make them match, because a reader will think the dotted line is the old pipeline. The primary map changed from what a reader might expect (the directional map is described as physically motivated but is worse); the paper should state that the absolute map was the pre-specified primary because it is the v1 map with the bug fixed.

**R3:** Table I says "not run" for the pooled-reference bootstrap; just run it. Uniform weights over 14 sensors still means the "multi-criteria" framing is thin; the limitation is stated, which is acceptable. ORCID is given only for A. Ali; the other authors should add theirs if they have them.

**Chair:** Decision.

**R1:** For an applied journal (IEEE Access, Applied Soft Computing, Engineering Applications of AI, Expert Systems with Applications): **Minor to major revision**; I say major because of the FOU analysis. For IEEE Transactions on Fuzzy Systems: still reject on novelty.

**R2:** **Major revision.** The FD002/FD004 run and a fair FOU comparison are needed before I can recommend acceptance. Both are doable.

**R3:** **Major revision.** The paper is now sound and reproducible. What remains is analysis, not repair.

**Chair:** Consensus: **Major Revision** at an applied venue. At IEEE TFS or similar theory-focused fuzzy journals: **Reject (insufficient methodological novelty)**, with a suggestion to submit to an applied venue.

---

## 3. Decision

| | R1 (Fuzzy/IT2) | R2 (PHM) | R3 (Streams/Eval) | **Panel** |
|---|---|---|---|---|
| Recommendation (applied venue) | Major revision | Major revision | Major revision | **Major Revision** |
| Recommendation (IEEE TFS) | Reject | Reject | Reject | **Reject (novelty)** |
| Novelty | 2/5 | 3/5 | 2/5 | **2.5/5** (was 2) |
| Technical soundness | 4/5 | 4/5 | 4/5 | **4/5** (was 2) |
| Experimental rigour | 3/5 | 3/5 | 4/5 | **3.5/5** (was 2) |
| Clarity | 4/5 | 4/5 | 4/5 | **4/5** (was 3) |
| Reproducibility | 5/5 | 5/5 | 5/5 | **5/5** (was 4) |

**Main strengths now:** correct model; honest positioning against interval TOPSIS; a held-out test set and strong baselines; proper uncertainty on comparisons; useful negative results (mean |z| wins, the per-engine reference matters most); exact incremental computation; full reproducibility in 35 s.

**Main remaining weaknesses:** limited fuzzy novelty; the FOU-flag advantage is not shown at a matched realised false-alarm rate or against a mean-plus-spread detector; a single operating condition.

---

## 4. Required changes for acceptance

### Major
**M1. FOU analysis at fair operating points.** Report ROC curves or AUC per fault type, and detection at equal realised false-alarm rate on engines 51 to 100. Add a mean-plus-spread baseline (upper membership alone, or `(|m - mu| + k r) / sigma`) and a zero-mean fault (variance increase over the whole window) so that the effect of the IT2 construction is separated from mean shift. Rewrite the abstract claim accordingly.

**M2. Multi-condition data.** Run FD002 and FD004 (and FD003 for the second fault mode) with operating-condition normalisation (cluster the three settings, normalise per cluster) and report Table I and Table III for them. N-CMAPSS is desirable but optional.

**M3. Statistical refinements.** At least 2,000 bootstrap replicates; Holm correction (or an explicit multiplicity statement) over the comparisons in Table I; run the bootstrap for the pooled-reference row.

### Minor
1. Fix "13 to 18 times" to "12 to 18 times" (abstract and Section VI-G); ratios are 18.1, 14.3, 11.6.
2. Also evaluate Proposition 4's own constant (from measured D^m, D^r), not only Proposition 3 with measured delta.
3. Add the age-only baseline to Table III (Spearman 0.601) and note the age/reference interaction for short test trajectories.
4. Drop Precision@10 from Table III or replace it with a less saturated measure (e.g. Precision@20 or recall of RUL <= 30 at k = number of positives).
5. Make Fig. 1 legend and caption consistent ("earlier membership formula").
6. State that the absolute map was the pre-specified primary (v1 map with the bug fixed) and the directional map an ablation.
7. Abstract: say that detrending helps ranking but hurts bias-step detection, rather than implying a flagging gain.
8. ORCID iDs for the other authors if available; a corresponding-author e-mail in the footnote.

---

## 5. Short letter to the editor

> The revision addresses every major point from round 1. The membership error is fixed and quantified, the model is honestly positioned as interval-valued TOPSIS, the proofs are corrected, and the evaluation now includes a held-out test set, strong statistical and supervised baselines, block-bootstrap comparisons, sensitivity, fault injection and an exact incremental implementation. The results are reported candidly: the method is competitive but not more accurate than the mean absolute z-score. Remaining work concerns the analysis of the uncertainty flags at matched false-alarm rates, multi-condition C-MAPSS subsets and minor statistical refinements. The panel recommends **Major Revision** for an applied computational-intelligence venue. For a theory-focused fuzzy-systems journal the methodological novelty remains insufficient.
