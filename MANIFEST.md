# Claim manifest

This file maps each result in the manuscript to the command that produces it and the file that holds it. The
manuscript is *MetaSplice: pathogenicity prediction and ACMG/AMP evidence for intronic variants outside the canonical
splice sites* (AJHG submission, version 7). Run the commands from the package folder after `. ./env.sh`, or run them all
with `./rerun.sh`.

**Needs** says what an analysis reads beyond this package:
- **—**: nothing.
- **cmp**: the precomputed SpliceAI and CADD scores, rebuilt with `scripts/comparators/lookup_comparators.py`.
- **raw**: also the raw SpliceAI scores, rebuilt with `scripts/comparators/run_spliceai_raw.py`.
- **svdb**: the SpliceVarDB classes, restored with `scripts/splicevardb/restore_splicevardb.py`.

Bootstrap CIs use fixed seeds, so reruns give the same values.

## Abstract and Results

| Claim | Command | Needs | Result file |
|---|---|---|---|
| Training set: 413,199 SNVs (41,032 P/LP) in 6,961 genes; P/LP share by region (Table 1) | `python scripts/build/V02_variant_sets.py` (build; needs the raw inputs) | — | `results/V02_variant_sets.txt`, `results/X2_distance_weighting_eval.txt` (by region) |
| Gene-grouped cross-validation: out-of-fold auPRC 0.994 overall, 0.878 non-canonical, 0.956 donor to 0.555 branch point (Table S4) | `python scripts/build/V06_train.py` (build) | — | `results/V06_train.txt` |
| Feature-group importance (Figure 3B) | `python scripts/analysis/P2_importance_within_gene.py --cohort pooled` | cmp | `results/P2_importance_within_gene_pooled.txt`, `.tsv` |
| Test set: 46,112 variants (6,056 P/LP); 40,619 non-canonical (602 P/LP); 40,296 scored by SpliceAI | `python scripts/build/V12_pooled_test_set.py` (build) | — | `results/V12_pooled_test_set.txt` |
| auPRC on all test variants (0.995, 0.976, 0.971) and non-canonical (0.916, 0.757, 0.364); paired differences with variant and gene CIs; by region and in the deep-intronic sensitivity set (Figure 2; Table S5) | `python scripts/analysis/Q1_pooled_comparison.py` | cmp | `results/Q1_pooled_comparison.txt`, `.tsv`, `Q1_pooled_pr_curves.tsv` |
| Single inputs (Pangolin 0.856, SpliceTransformer 0.717), baselines and the gain over Pangolin alone, overall and by region (Figure 3A; Table S6) | `python scripts/analysis/P1_inputs_baselines_ablations.py --cohort pooled` | cmp | `results/P1_inputs_baselines_ablations_pooled.txt`, `.tsv` |
| Gene-disjoint model: auPRC 0.911; full minus gene-disjoint 0.005; gene-disjoint minus SpliceAI 0.157 and minus CADD 0.548 | `python scripts/analysis/Q1_pooled_comparison.py` (scores from `data/A3b_gene_disjoint_scores.csv`, made by `A3b_gene_heldout_retrain.py`) | cmp | `results/Q1_pooled_comparison.txt`, `.tsv` |
| 40,239 of 40,619 non-canonical test variants (99.1%) in genes with training variants | the one-line count below | — | (computed; no result file) |
| Within-gene concordance in 353 genes (19,975 pairs): 0.990, 0.982, 0.878, and the differences (Figure 3C) | `python scripts/analysis/P2_importance_within_gene.py --cohort pooled` | cmp | `results/P2_importance_within_gene_pooled.txt`, `.tsv` |
| Ablations: gene-level, DANN and GPN-MSA, both; retrained with genes held out (0.903); by region (Figure 3A; Table S7) | `python scripts/analysis/P1_inputs_baselines_ablations.py --cohort pooled` | cmp | `results/P1_inputs_baselines_ablations_pooled.txt`, `.tsv` |
| Literature set: 640 of 697 (91.8%) at ≥ 0.1; 101 of 130 deep intronic (Figure 4C) | `python scripts/analysis/literature_recovery.py` | — | `results/literature_recovery.txt` |
| SpliceVarDB: 8,015 splice-altering and 4,716 normal; auPRC 0.993, 0.985, 0.981; detection and false-positive rates; within-region differences; broad endpoint (Figure 4A–B; Table S8) | `python scripts/analysis/A6_splicevardb.py` | cmp, svdb | `results/A6_splicevardb.txt` |
| Calibration thresholds in the training calibration cohort: PP3 0.052, 0.140, 0.544, 0.994; BP4 0.00386, 0.0017, 0.000275; SpliceAI 0.2, 0.3, 0.49 (Table S3) | `python scripts/analysis/S5_clingen_calibration.py`, then `python scripts/analysis/S6_review_status.py` | cmp | `results/S6_thresholds.tsv`, `results/S6_review_status.txt` (≥ 1 star, used); `results/S5_thresholds.tsv` (all statuses) |
| Prior 0.0441 | `python scripts/analysis/S11_prior.py` | cmp | `results/S11_prior.txt`, `S11_prior_frozen.json` |
| Likelihood ratios of the MetaSplice and SpliceAI intervals in the test set's calibration cohort, with exact CIs (Figure 5A; Table S9) | `python scripts/analysis/Q2_pooled_calibration_rule.py` | raw | `results/Q2_pooled_calibration_rule.txt`, `.tsv` |
| Candidate rule against the ClinGen SpliceAI rule: codes among P/LP and B/LB (505 of 578; 73 of 37,263; …), differences with CIs, rule LRs (446; 20.4) at three priors (Figure 5B; Table 2; Table S9) | `python scripts/analysis/Q2_pooled_calibration_rule.py` | raw | `results/Q2_pooled_calibration_rule.txt`, `.tsv` |
| Resolved VUS (859: 153 P/LP, 706 B/LB) and remaining VUS (60,776) under both rules (Table S10) | `python scripts/analysis/Q2_pooled_calibration_rule.py` | raw | `results/Q2_pooled_calibration_rule.txt`, `.tsv` |
| Raw SpliceAI scores (±4,999 nt) for the ClinGen rule | `python scripts/comparators/run_spliceai_raw.py …` (V11, V11b, V11c) | cmp | `results/V11_spliceai_raw.txt`, `V11b_…`, `V11c_…` |

Percentages and differences in the text that do not appear in a result file are arithmetic on the listed values.
Examples: 91.2% = 37,433 / 41,032 (canonical P/LP in training, Table 1); 90.1% = 5,454 / 6,056 (Table 1); 1.5% = 602
/ 40,619.

The 40,239 count is computed as follows:

    . ./env.sh && python -c "
    import pandas as pd
    t = pd.read_csv('data/v2_features_pooled_test.csv', low_memory=False)
    g = set(pd.read_csv('data/v2_variants_training.csv', usecols=['gene_id']).gene_id)
    nc = t[~t.region.str.startswith('Canonical')]
    print(int(nc.gene_id.isin(g).sum()), 'of', len(nc))"

## Methods values

| Value | Source |
|---|---|
| GENCODE v50 MANE Select and MANE Plus Clinical: 19,373 transcripts in 19,299 genes | `results/V01_annotation.txt` (`scripts/build/V01_annotation.py`) |
| Model settings (38 features, 300 trees, depth 6, learning rate 0.1, five gene-grouped folds) | `scripts/build/V06_train.py`, `results/V06_train.txt` |
| The production path (scorer from a VCF, with fresh SpliceTransformer and Pangolin runs) reproduces the stored features and scores on 1,000 training and 986 test variants (equivalence gate) | `results/V07_gate.txt` |
| SpliceAI raw configuration and installation check | `results/V11_spliceai_raw.txt` |

## Figures

| Figure | Command | Inputs |
|---|---|---|
| Figure 1 (study design) | `python scripts/figures/make_fig1_design_v3.py` | none (counts as in Table 1) |
| Figures 2–5 | `python scripts/figures/make_figs_v2.py` | `results/` files of Q1, P1 and P2 (pooled), A6, literature recovery, Q2 and S6 |

The figure scripts write to `figures/`. Fonts can differ between systems, so a regenerated figure may not be
byte-identical to the submitted one.

## Tables

| Table | Source |
|---|---|
| Table 1 (variant sets by region) | Training: `X2_distance_weighting_eval.txt`; test set: `V12_pooled_test_set.txt`; training calibration cohort: `S6_review_status.txt` |
| Table 2 (codes under each rule) | `Q2_pooled_calibration_rule.tsv` |
| Table S1 (other datasets) | `A6_splicevardb.txt`, `literature_recovery.txt`, `Q2_pooled_calibration_rule.txt`, `S11_reference_scores.txt` |
| Table S2 (features) | `scripts/score_v2.py` (feature definitions) |
| Table S3 (thresholds) | `S5_thresholds.tsv`, `S6_thresholds.tsv` |
| Table S4 (cross-validation) | `V06_train.txt` |
| Table S5 (discrimination) | `Q1_pooled_comparison.tsv` |
| Table S6 (inputs and baselines), Table S7 (ablations) | `P1_inputs_baselines_ablations_pooled.tsv` |
| Table S8 (SpliceVarDB) | `A6_splicevardb.txt` |
| Tables S9a–c (likelihood ratios and the rule), Table S10 (remaining VUS) | `Q2_pooled_calibration_rule.tsv`, `.txt` |

## Other analyses in this package

Other analyses were run during the study but are not reported in the manuscript. Their scripts and results are
included, and `rerun.sh` runs them:
- the April-only test set: A1, C0, S5–S10, S12;
- the September-only cohort: C1, F1;
- the separate-cohort versions of P1 and P2;
- P3;
- the distance analyses: X1, X2;
- the rule's variants: X3;
- the gnomAD reference: S11_reference_scores, S11_prior_diagnostic, S11_recalibrate.
