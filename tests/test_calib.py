"""Unit tests of the v2.0 calibration module. Synthetic data only; no evaluation-set label is read."""
import os, sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis"))
import calib
import common


def test_clingen_lr_levels():
    c, need = calib.C_MAIN, calib.LR_MAIN
    assert c == 1124.0
    assert need["Supporting"] == pytest.approx(2.406, abs=1e-3)
    assert need["Moderate"] == pytest.approx(5.790, abs=1e-3)
    assert need["Strong"] == pytest.approx(33.53, abs=1e-2)
    assert need["VeryStrong"] == 1124.0


def test_zero_benign_interval_has_finite_lower_bound():
    lr, lo, hi = calib.lr_ci(116, 799, 0, 48155)
    assert np.isinf(lr) and np.isinf(hi)
    assert 1200 < lo < 1400            # 1,300 in v1's S5 run


def test_verdicts_distinguish_point_and_ci():
    point, ci, _ = calib.verdicts("BP4_Supporting", 0.384, 0.156, 0.797)
    assert (point, ci) == ("meets", "INCONCLUSIVE")
    point, ci, _ = calib.verdicts("PP3_Strong", 526, 353, 803)
    assert (point, ci) == ("meets", "clears")


def test_calibration_is_conservative_on_known_lr():
    rng = np.random.default_rng(1)
    s = np.r_[rng.normal(0, 1, 300000), rng.normal(2, 1, 3000)]
    y = np.r_[np.zeros(300000, int), np.ones(3000, int)]
    res = calib.calibrate(s, y, "synthetic", n_boot=300)
    _, need, pp3, _ = res["by_prior"][calib.PRIOR]
    true_strong = (np.log(need["Strong"]) + 2) / 2        # lr(s) = exp(2s - 2)
    assert pp3["Strong"] is not None and pp3["Strong"] >= true_strong


def test_assign_takes_the_highest_level_reached():
    s = np.array([np.nan, 0.001, 0.05, 0.5, 0.95])
    codes = calib.assign(s, {"Supporting": 0.4, "Moderate": 0.6, "Strong": 0.9, "VeryStrong": None},
                         {"Supporting": 0.1, "Moderate": 0.01, "Strong": None, "VeryStrong": None})
    assert list(codes) == ["NoScore", "BP4_Moderate", "BP4_Supporting", "PP3_Supporting", "PP3_Strong"]


def test_v2_has_one_model():
    assert calib.compared_models() == ["MetaSplice"]


def test_in_cohort_uses_region_af_and_gene_id():
    d = pd.DataFrame({"region": ["Deep Intronic", "Canonical Donor Site", "PPT Region", "Donor Region"],
                      "gnomad_af": [0.0, 0.0, 0.02, 0.009],
                      "gene_id": ["ENSG1.1", "ENSG1.1", "ENSG1.1", "ENSG2.1"]})
    assert list(calib.in_cohort(d, {"ENSG1.1"})) == [True, False, False, False]
    with pytest.raises(ValueError):
        calib.in_cohort(d.assign(gnomad_af=[np.nan, 0, 0, 0]), {"ENSG1.1"})


def test_one_star_excludes_zero_star_statuses():
    d = pd.DataFrame({"review_status": ["criteria_provided,_single_submitter", "no_assertion_criteria_provided"]})
    assert list(calib.one_star(d)) == [True, False]


def test_deep_intronic_sensitivity_and_flag_dtype():
    d = pd.DataFrame({"region": ["Deep Intronic"] * 3 + ["PPT Region"],
                      "canonical_any_gencode": [False, True, False, False],
                      "exonic_any_gencode": [False, False, True, False]})
    assert list(common.deep_intronic_sensitivity(d)) == [True, False, False, False]
    with pytest.raises(TypeError):        # a string column would make astype(bool) read "False" as True
        common.deep_intronic_sensitivity(d.astype({"exonic_any_gencode": str}))


def test_variant_key_adds_chr_prefix():
    d = pd.DataFrame({"chr": ["1", "chrX"], "pos": [10.0, 20], "ref": ["A", "C"], "alt": ["G", "T"]})
    assert list(common.variant_key(d)) == ["chr1:10:A:G", "chrX:20:C:T"]


def test_estimate_prior_recovers_known_mixture():
    rng = np.random.default_rng(2)
    p, b = rng.beta(5, 2, 2000), rng.beta(2, 5, 200000)
    ref = np.r_[rng.beta(5, 2, 3000), rng.beta(2, 5, 97000)]          # true prior 0.03
    assert abs(calib.estimate_prior(ref, p, b) - 0.03) < 0.005


def test_bootstrap_with_components_is_not_narrower():
    rng = np.random.default_rng(5)
    p, b = rng.beta(5, 2, 300), rng.beta(2, 5, 20000)                  # small P component
    ref = np.r_[rng.beta(5, 2, 3000), rng.beta(2, 5, 97000)]
    lo_r, hi_r = calib.prior_bootstrap(ref, p, b, n_boot=200, resample_components=False)
    lo_a, hi_a = calib.prior_bootstrap(ref, p, b, n_boot=200, resample_components=True)
    assert (hi_a - lo_a) > (hi_r - lo_r)


def test_mixture_fit_detects_misspecified_components():
    rng = np.random.default_rng(4)
    p, b = rng.beta(5, 2, 2000), rng.beta(2, 5, 200000)
    good = np.r_[rng.beta(5, 2, 3000), rng.beta(2, 5, 97000)]
    bad = np.r_[rng.beta(5, 2, 3000), rng.beta(2, 3, 97000)]           # benign part differs from ClinVar B
    assert calib.mixture_fit(good, p, b, calib.estimate_prior(good, p, b)) < 0.02
    assert calib.mixture_fit(bad, p, b, calib.estimate_prior(bad, p, b)) > 0.02


def test_reference_window_widens_windows():
    rng = np.random.default_rng(3)
    s = np.r_[rng.normal(0, 1, 50000), rng.normal(2, 1, 1000)]
    y = np.r_[np.zeros(50000, int), np.ones(1000, int)]
    grid = np.linspace(-2, 4, 50)
    base = calib.windows(grid, s, y, 100, 25, 0.0)
    wide = calib.windows(grid, s, y, 100, 25, 0.03, ref=rng.normal(0, 1, 100000))
    assert (wide >= base).all() and (wide > base).any()


def test_windows_unchanged_without_reference():
    rng = np.random.default_rng(6)
    s = np.r_[rng.normal(0, 1, 20000), rng.normal(2, 1, 500)]
    y = np.r_[np.zeros(20000, int), np.ones(500, int)]
    grid = np.linspace(-2, 4, 40)
    old = np.maximum(calib._kth_dist(np.sort(s), grid, max(100, int(np.ceil(0.03 * len(s))))),
                     np.maximum(calib._kth_dist(np.sort(s[y == 1]), grid, 25), calib._kth_dist(np.sort(s[y == 0]), grid, 25)))
    assert np.array_equal(calib.windows(grid, s, y, 100, 25, 0.03), old)
