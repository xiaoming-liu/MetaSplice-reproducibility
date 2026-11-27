# Claim manifest

This file maps each result in the manuscript *MetaSplice: pathogenicity prediction and ACMG/AMP evidence for intronic
variants outside the canonical splice sites* to the command that produces it and the file that holds it. Run the
commands from the package folder after `. ./env.sh`, or run them all with `./rerun.sh`.

**Needs** says what a command reads beyond this package:
- **—**: nothing.
- **cmp**: the precomputed SpliceAI and CADD scores, rebuilt with `scripts/comparators/lookup_comparators.py`.
- **raw**: also the raw SpliceAI scores, rebuilt with `scripts/comparators/run_spliceai_raw.py`.
- **svdb**: the SpliceVarDB classes, restored with `scripts/splicevardb/restore_splicevardb.py`.
- **build**: the raw inputs of the build (README, "Building the data from raw inputs").

Bootstrap CIs use fixed seeds, so reruns give the same values.

## Results

| Claim | Command | Needs | Result file |
|---|---|---|---|
| Training set: 413,199 SNVs (41,032 P/LP); 37,433 P/LP (91.2%) at the canonical sites; P/LP share from 0.29% (deep intronic) to 7.0% (donor region) (Table S1a) | `python scripts/build/V02_variant_sets.py` | build | `results/V02_variant_sets.txt` |
| 6,961 training genes; gene-grouped cross-validation: out-of-fold auPRC 0.994 overall and 0.878 for the 375,497 non-canonical variants (3,599 P/LP) (Table S4) | `python scripts/build/V06_train.py` | build | `results/V06_train.txt` |
| Feature-group importance (Figure 2B) | `python scripts/analysis/P2_importance_within_gene.py` | cmp | `results/P2_importance_within_gene_pooled.txt`, `.tsv` |
| Test set: 46,112 variants (6,056 P/LP); 40,619 non-canonical (602 P/LP); 40,296 non-canonical scored by SpliceAI (600 P/LP) (Table S1a) | `python scripts/build/V12_pooled_test_set.py` | build | `results/V12_pooled_test_set.txt`; denominators checked in `results/Q1_pooled_comparison.txt` |
| auPRC on all test variants (0.995, 0.976, 0.971) and on non-canonical variants (0.916, 0.757, 0.364); 0.919 on the 40,296 scored by both, a paired gain of 0.162 over SpliceAI and 0.552 over CADD, with class-stratified and gene-resampled CIs; by region, including deep intronic (0.824 versus 0.551) (Figure 1; Table S5) | `python scripts/analysis/Q1_pooled_comparison.py` | cmp | `results/Q1_pooled_comparison.txt`, `.tsv`, `results/Q1_pooled_pr_curves.tsv` |
| Gene-disjoint model (7,194 genes in five folds): auPRC 0.911; full minus gene-disjoint 0.005; gene-disjoint minus SpliceAI 0.157 and minus CADD 0.548 | `python scripts/analysis/Q1_pooled_comparison.py` | cmp | `results/Q1_pooled_comparison.txt`, `.tsv` (section 2) |
| 40,239 of 40,619 non-canonical test variants (99.1%) in genes with training variants | the one-line count below | — | (computed; no result file) |
| Single inputs (Pangolin 0.856, SpliceTransformer 0.717, no other above 0.281), baselines, and MetaSplice's gains (0.060 over Pangolin, 0.037 and 0.061 over the baselines; 0.120 at deep intronic positions) (Figure 2A; Table S6) | `python scripts/analysis/P1_inputs_baselines_ablations.py` | cmp | `results/P1_inputs_baselines_ablations_pooled.txt`, `.tsv` |
| Ablations: gene-level (0.011), DANN and GPN-MSA (0.004), both (0.009); both, retrained gene-disjoint (0.903; 0.150 above SpliceAI) (Figure 2A; Table S7) | `python scripts/analysis/P1_inputs_baselines_ablations.py` | cmp | `results/P1_inputs_baselines_ablations_pooled.txt`, `.tsv` |
| Within-gene concordance in 353 genes (19,975 pairs): 0.990, 0.982, 0.878, and the differences (Figure 2C) | `python scripts/analysis/P2_importance_within_gene.py` | cmp | `results/P2_importance_within_gene_pooled.txt`, `.tsv` |
| Literature set: 640 of 697 (91.8%) at ≥ 0.1; 101 of 130 deep intronic (Figure 3C) | `python scripts/analysis/literature_recovery.py` | — | `results/literature_recovery.txt` |
| SpliceVarDB: 22,274 variants after removing 40 assay-discordant ones; 8,015 splice-altering and 4,716 normal; auPRC 0.993, 0.985, 0.981; 92.8% detected at 0.1 with a 1.2% false-positive rate; within-region differences; broad endpoint (Figure 3A–B; Table S8) | `python scripts/analysis/A6_splicevardb.py` | cmp, svdb | `results/A6_splicevardb.txt` |
| Training calibration cohort: 294,612 variants with ≥ 1 review star (3,018 P/LP) | `python scripts/analysis/S15_lower_bound_prior.py` | cmp, raw | `results/S15_lower_bound_prior.txt` (section 2) |
| Lower bound on the prior: 1,099 of 81,116,352 rare intronic gnomAD SNVs in the calibration's regions and genes (1.35 × 10⁻⁵); required likelihood ratios 9.34, 87.3, 7,613 and 5.8 × 10⁷ | `python scripts/analysis/S15a_prior_population.py`, then `python scripts/analysis/S15_lower_bound_prior.py` | build, cmp (S15a); cmp, raw (S15) | `results/S15a_prior_population.txt`, `results/S15_lower_bound_prior.txt` (section 1), `results/S15_pooled.tsv` |
| Derived thresholds: MetaSplice PP3 ≥ 0.182 (Supporting) and ≥ 0.794 (Moderate), BP4 ≤ 0.00071; SpliceAI 0.35 and 0.63; all review statuses (Table S3) | `python scripts/analysis/S15_lower_bound_prior.py` | cmp, raw | `results/S15_thresholds.tsv`, `results/S15_lower_bound_prior.txt` (section 2) |
| Interval likelihood ratios in the test set's calibration cohort: PP3 Moderate 628, PP3 Supporting 31.1, BP4 Supporting 0.0202, with exact CIs (Figure 4A; Table S9a) | `python scripts/analysis/S15_lower_bound_prior.py` | cmp, raw | `results/S15_lower_bound_prior.txt` (3a), `results/S15_pooled.tsv` (levels) |
| Candidate rule against the ClinGen SpliceAI rule in 37,841 variants (578 P/LP): PP3 92.2% and 0.44%, BP4 77.1% and 1.6%, against 90.7%, 4.44%, 88.5% and 6.7%; differences with CIs; rule likelihood ratios 208 and 0.0202; SpliceAI rule 20.4, BP4 upper limit 0.1074 against 0.1071 (Figure 4B; Table 1; Tables S9b–c) | `python scripts/analysis/S15_lower_bound_prior.py` | cmp, raw | `results/S15_lower_bound_prior.txt` (3b), `results/S15_pooled.tsv` (rule, rule difference) |
| Resolved VUS (859: 153 P/LP, 706 B/LB): PP3 for 15 (2.1%) against 220 (31.2%) later-B/LB, difference 29.0 points; BP4 for no later-P/LP against 8 (5.2%) (Table 1) | `python scripts/analysis/S15_lower_bound_prior.py` | cmp, raw | `results/S15_lower_bound_prior.txt` (3b), `results/S15_pooled.tsv` |
| Remaining VUS (60,776): PP3 30.4%, BP4 26.5%, neither 43.1%, against 38.7%, 51.3% and 10.0%; MetaSplice ≥ 0.1, 0.2 and 0.5: 33.1%, 30.0% and 25.1% (Table S10) | `python scripts/analysis/S15_lower_bound_prior.py` | cmp, raw | `results/S15_lower_bound_prior.txt` (3c), `results/S15_pooled.tsv` |

Percentages and differences in the text that do not appear in a result file are arithmetic on the listed values.
Examples, from the region counts of `V02_variant_sets.txt` and `V12_pooled_test_set.txt`: 37,433 = 20,443 + 16,990
(the two canonical sites); 91.2% = 37,433 / 41,032; 0.29% = 552 / 191,795 (deep intronic); 7.0% = 1,862 / 26,476
(donor region); 90.1% = 5,454 / 6,056; 1.5% = 602 / 40,619. Bounds are printed in scientific notation (1.3548e-05 is
the 1.35 × 10⁻⁵ of the text).

The 40,239 count is computed as follows:

    . ./env.sh && python -c "
    import pandas as pd
    t = pd.read_csv('data/v2_features_pooled_test.csv', low_memory=False)
    g = set(pd.read_csv('data/v2_variants_training.csv', usecols=['gene_id']).gene_id)
    nc = t[~t.region.str.startswith('Canonical')]
    print(int(nc.gene_id.isin(g).sum()), 'of', len(nc))"

## Supplemental Methods values

| Value | Source |
|---|---|
| GENCODE v50 MANE Select and MANE Plus Clinical: 19,373 transcripts, 19,299 genes, 204,067 exons; 17,072 gene entries narrowed to the MANE transcripts | `results/V01_annotation.txt` |
| 1,370,180 P/LP or B/LB SNVs in the training release, 413,199 eligible; 921,807, 22,569, 10,851 and 1,754 excluded by reason | `results/V02_variant_sets.txt` |
| Test set selection: 457,664 eligible SNVs, 411,552 removed, 46,112 test variants; SpliceAI scores 45,749 (6,015 P/LP) | `results/V12_pooled_test_set.txt`, `results/Q1_pooled_comparison.txt` |
| VUS cohort: 71,267 eligible (56,282 uncertain, 14,985 conflicting); 61,016 in the calibration cohort; 60,776 remaining | `results/V02_variant_sets.txt`, `results/V11b_spliceai_raw_vus.txt`, `results/S15_lower_bound_prior.txt` |
| Literature set: 746 variants lifted, 727 candidates, 30 in training, 697 included | `results/V02_variant_sets.txt` |
| SpliceVarDB set: 44,510 entries, 44,324 lifted, 24,372 eligible, 1,464 in training, 22,908 entries for 22,314 SNVs; 40 discordant; 22,274 (8,015, 9,543, 4,716) | `results/V02_variant_sets.txt`, `results/A6_splicevardb.txt` |
| SpliceVarDB per-study counts (13,925 SAVNet, 28,854 MFASS, 1,914 MiSplice; 1,256 MFASS splice-altering) | printed by `scripts/splicevardb/restore_splicevardb.py` (the reconstruction's classifier) when run on the source files |
| Gene-level values for 19,230 of the 19,299 MANE genes | `scripts/build/gene_level/README_gene_level.md` |
| Model settings (38 features, 300 trees, depth 6, learning rate 0.1, five gene-grouped folds) | `scripts/build/V06_train.py`, `results/V06_train.txt` |
| SpliceAI raw configuration and installation check | `results/V11_spliceai_raw.txt` |
| Mixture estimate of the prior: 0.0311 (95% CI 0.0286–0.0340), discrepancy 0.41, 25- and 100-bin estimates 0.084 and 0.0065; not accepted | `python scripts/analysis/S11_reference_scores.py`, then `python scripts/analysis/S11_prior.py` (cmp): `results/S11_prior.txt`, `.tsv` |
| The bound's population (81,116,352 = 80,831,262 not in ClinVar + 285,090 in ClinVar) and its checks | `results/S15a_prior_population.txt`, `results/V02b_gnomad_reference.txt` |
| Sensitivity of the bound (AF < 10⁻⁴: 1.29 × 10⁻⁵; all review statuses: 1.53 × 10⁻⁵; ≥ 2 stars: 7.6 × 10⁻⁶) and the thresholds there; Supporting requirement at other priors (2.08 at 0.1, 4.56 at 0.001) | `results/S15_lower_bound_prior.txt` (sections 1–2), `results/S15_pooled.tsv` |
| The bound and the rule by region (2.4 × 10⁻³ in the donor region to 2.0 × 10⁻⁶ deep intronic), and the gnomAD reference as the benign denominator (0.73% PP3, 39.9% BP4; likelihood ratios 126 and 0.039) | `results/S15_lower_bound_prior.txt` (section 4), `results/S15_pooled.tsv` |

## Figures

| Figure | Command | Inputs |
|---|---|---|
| Figures 1–3 | `python scripts/figures/make_figs_gm.py` | `results/` files of Q1, P1 and P2, A6 and literature recovery |
| Figure 4 | `python scripts/figures/make_fig4_gm.py` | `results/S15_pooled.tsv`, `results/S15_thresholds.tsv` |
| Figure S1 (study design) | `python scripts/figures/make_figS1_design.py` | none |

The figure scripts write to `figures/`. Fonts can differ between systems, so a regenerated figure may not be
byte-identical to the one in `figures/`.

## Tables

| Table | Source |
|---|---|
| Table 1 (codes under each rule) | `S15_pooled.tsv` (rule) |
| Table S1a (variant sets by region) | Training: `V02_variant_sets.txt`; deep-intronic sensitivity set: `V06_train.txt`; test set: `V12_pooled_test_set.txt`, `Q1_pooled_comparison.txt`; calibration cohorts: `S15_lower_bound_prior.txt` |
| Table S1b (other datasets) | `A6_splicevardb.txt`, `literature_recovery.txt`, `S15_lower_bound_prior.txt`, `V02b_gnomad_reference.txt` |
| Table S2 (features) | `scripts/score_v2.py` (feature definitions) |
| Table S3 (thresholds) | `S15_thresholds.tsv` |
| Table S4 (cross-validation) | `V06_train.txt` |
| Table S5 (discrimination) | `Q1_pooled_comparison.tsv` (section 1) |
| Table S6 (inputs and baselines), Table S7 (ablations) | `P1_inputs_baselines_ablations_pooled.tsv` |
| Table S8 (SpliceVarDB) | `A6_splicevardb.txt` |
| Tables S9a–c (likelihood ratios and the rule), Table S10 (remaining VUS) | `S15_pooled.tsv`, `S15_lower_bound_prior.txt` |
