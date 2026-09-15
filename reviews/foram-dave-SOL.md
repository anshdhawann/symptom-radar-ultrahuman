# Adversarial review — FORAM / Dave working pack

## Recomputed results

I recomputed from `sections.json`, not from either script. To reproduce `_revision_stats.json`, ties within a section must contribute no direction and equal final scores must be ordered by species code.

| Item | Recomputed result | Assessment |
|---|---|---|
| Unique species | **41** (codes 1–42, code 24 absent) | Matches the note and `_revision_stats.json`. |
| Conflict pairs | **231** | Matches. This is 231/820 = **28.1707%** of all possible unordered pairs, but 231/787 = **29.3520%** of pairs that co-occur at least once; 33 pairs never co-occur and cannot be ordered. |
| Four-section taxa | **11**: 2, 4, 7, 8, 16, 17, 18, 21, 23, 25, 36 | Matches. Presence in all sections makes these ubiquitous taxa, not validated marker or “anchor” taxa. |
| Published normalized-LAD top 15 | **11, 1, 35, 28, 5, 20, 19, 40, 27, 32, 17, 34, 30, 37, 12** | Matches the note and stats file only under an unstated normalization described below. |
| Published “pairwise majority” top 15 | **21, 35, 11, 1, 8, 32, 2, 5, 17, 23, 30, 33, 40, 20, 18** | Matches the script’s pooled section-vote margin, not pairwise-majority wins minus losses. |
| Published within-5 result | **25/41 = 60.9756%** | Reproduced under the two actual calculations and species-code tie-breaking. It can be 24/41 under another valid ordering of tied scores. |
| Species 39 tops | Type **108**, Ogruknang **4**, Parsons **19**; absent from Kipnik | Matches. Under the calculation behind the stats, its section positions are 0.7578, 0, and 1, giving normalized-LAD rank 30 versus pooled-score rank 23. |

## Critical

### 1. The normalized-LAD result does not use the normalization stated in the working note

The published top 15 is reproduced by transforming each top as:

`(top - minimum species top in that section) / (maximum species top - minimum species top)`

That sets the lowest **LAD** in each section to zero. It does not set the lowest bed with data to zero. For example, the Type Section denominator is based on species tops 11–139 even though occurrence data extend to bed 1 and the file reports 141 beds.

Using beds 1 through `num_beds` as the available proxy for the method actually described gives:

**11, 1, 35, 5, 20, 28, 19, 40, 27, 32, 34, 17, 37, 12, 30**

The source workbook is needed to establish the real bed-coordinate minima and maxima, especially if numbering is non-contiguous. Do not fix this by merely changing prose to “min/max LAD normalization”: that scaling depends on which taxa happen to be observed and is unstable when taxa are added or removed. Define the section coordinate system, normalize against its bounds, and regenerate every result.

### 2. “Pairwise majority scoring” is not the calculation that produced the published order

`foram_analysis.py` builds majority edges, but the final score ignores those edges. It sums every section-level strict comparison:

`score(A) = sum over sections and co-occurring B of [A above B] - [B above A]`

That is a pooled pairwise vote margin. It is not “take the majority for each species pair, then rank by wins minus losses,” as the working note says. A literal pair-level majority/Copeland calculation gives this top 15:

**11, 1, 35, 21, 5, 33, 32, 40, 8, 30, 20, 2, 17, 28, 12**

With the currently published LAD order, literal majority scoring gives 28/41 = 68.3% within five ranks, not 25/41 = 61.0%. With the provisional 1-to-`num_beds` LAD normalization, it gives 26/41 = 63.4%. There is therefore no defensible single agreement percentage until both algorithms and tie rules are fixed.

The `DiGraph` does not derive the ranking, resolve cycles, or perform topological analysis. Calling this a graph method would mislead Dave. If the pooled score is retained, name it exactly and disclose that it weights repeated section-level comparisons and observation opportunity. Keep the spring-layout graph out of the collaborator-facing explanation unless its purely illustrative status is unmistakable.

### 3. Both executable paper paths still regenerate the disowned story

`foram_analysis.py` still:

- averages raw LAD bed numbers across the 141-bed Type Section and 19–30-bed wells (`lad_ranking`, lines 142–168);
- labels bed indices as `Depth (meters)` (line 383) and emits metre ranges in its paper;
- says the method applies topological analysis/sort although no topological sort is run (embedded paper, lines 710–713 and 742–748);
- claims `>70%` agreement (lines 679–685);
- presents Monte Carlo as a third probabilistic method and confidence source (lines 272–338 and 749–754), although it only adds arbitrary ±2 noise to the same raw LAD values.

`generate_paper_v2.py` still:

- describes raw cross-section LAD averaging (lines 313–318);
- calls bed indices metres throughout, including an arbitrary “±2 meters” uncertainty (lines 274–281, 326–329, and 518–519);
- claims three-method convergence is independent evidence of genuine signal and repeats 70%/over-70% claims (lines 419, 480–482, and 505–509);
- hard-codes stale rankings, scores, and confidence labels rather than loading the revised calculations (lines 430–448 and 585–602);
- assigns Dave sole authorship and corresponding-author responsibility for unvalidated AI-generated claims.

The Monte Carlo result is a noisy restatement of raw LAD averaging, on the same observations and incompatible scales. It is neither independent validation nor meaningful uncertainty quantification; ±2 local bed indices have no common physical meaning. Running or attaching either generator can recreate contradictory prose and figures. The old PDF, generated figures, and both scripts must not be included in Dave’s email pack in their present state.

## Warning

### Forced total order conceals missing information and coverage bias

Thirty-three species pairs never co-occur. Many others have tied section votes, and the pooled score has tied groups including 21/35, 2/5/17, and 30/33. Nevertheless, the output forces all 41 taxa into unique ranks using species-code order as an undocumented tie-break. The claimed within-5 count changes from 25 to 24 under permissible tied-score orderings.

The pooled score also gives taxa different numbers of voting opportunities depending on section presence and assemblage size. A four-section taxon generally contributes more comparisons than a one-section taxon; this is not the same as stronger chronological evidence. Species 28, for example, occurs only in the Type Section yet ranks fourth by the published normalized LAD. Report coverage and unresolved/tied groups, not only a total order.

### Species 39 needs a stronger caution

The raw tops are correctly reported, but the pattern is more diagnostic than the note says:

- Type Section: range 22–108;
- Ogruknang M-31: range 1–4;
- Parsons A-44: a single occurrence at bed 19, which is also the section maximum;
- absent from Kipnik O-20.

Its averaged normalized position hides extreme section disagreement and a top-boundary singleton that could reflect truncation, sampling, reworking, miscoding, facies, or a genuine range difference. Do not use Species 39 in a generalized zonation until Dave checks the occurrence sheets and taxonomy.

### “Youngest → oldest” depends on an unverified coordinate convention

The calculations use the maximum numeric value as the highest/youngest occurrence. That can be correct for bed numbers increasing upward, but it is the opposite of ordinary measured depth increasing downward. The scripts alternate between “bed,” “depth,” “higher,” and “shallower.” Confirm direction independently for all four sheets before presenting an age order.

### Ubiquity is not marker confidence

The set of 11 four-section taxa is correct, but occurrence in all four sections does not establish taxonomic reliability, synchroneity, facies independence, or a usable datum. “Ubiquitous taxa” or “candidate anchors for Dave to assess” is honest. The PDF’s “high confidence” and “reliable correlation marker” language is not.

### The current figures are not trustworthy

Beyond inheriting stale rankings and metre labels, the correlation-panel code scales all sections by a fixed 150 rather than by local section bounds and mixes species-rank and normalized-depth coordinates on one axis. Those figures should be regenerated only after the calculation is fixed and then checked visually.

## Nit

- `generate_paper_v2.py` says there are 208 total beds; the four reported counts sum to **218**.
- “28% of unordered pairs” is arithmetically correct against all 820 possible pairs, but the comparable-pair denominator (29.4%) should be shown to avoid treating 33 unobserved pairs as non-conflicts.
- `_revision_stats.json` reproduces the published numbers but records neither formulas nor tie handling, so it is not an audit trail.
- “Traditional LAD ranking” is too strong for an unweighted mean across section-local coordinates. Call it an exploratory normalized-LAD score.

## **Safe to email Dave?**

**No.**

Blockers:

1. Choose and document the actual section normalization using verified bed bounds and direction; regenerate the LAD order.
2. Choose either literal pairwise-majority scoring or the pooled section-vote score, name it accurately, document missing comparisons/ties, and regenerate the order and agreement statistic.
3. Check Species 39 against the source occurrence sheets and present it as unresolved.
4. Send no old PDF, generated figures, `foram_analysis.py`, or `generate_paper_v2.py` until their raw-scale, metre, topological-sort, >70%, hard-coded-result, and Monte Carlo claims are removed or the files are explicitly quarantined as obsolete.

A corrected note can still be framed as a tops-only exploratory ranking for Dave’s judgment. It cannot yet be framed as a validated zonation method or as a coherent reproducible pack.
