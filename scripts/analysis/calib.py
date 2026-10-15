"""Shared ClinGen-style calibration code (Pejaver et al. 2022) for the v2.0 analyses. No side effects.

The framework (lr_levels through fmt_t) is v1's calib.py unchanged: prior 0.0441 with c = 1124, windows of
>= 100 labelled variants and >= 25 of each class, 10,000 bootstrap iterations, exact Clopper-Pearson
intervals. The data layer is v2.0's (docs/METASPLICE_V2_PLAN.md, Task 8):
  - one MetaSplice model, v2.0 (38 features); compared_models() is ["MetaSplice"], and the reduced-model
    machinery of v1 (G1 JSON, LOEUF recoding, refits) is gone: v2.0 encodes a missing LOEUF as missing;
  - pathogenic genes and the fold map are keyed by Ensembl gene ID;
  - in_cohort reads the scorer's gnomad_af, which is never missing (0.0 means looked up and not found);
  - SpliceAI and CADD come from data/v2_comparators.csv (V08).
Importing this module writes nothing.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import beta
from common import (ROOT, OUT, NONCANONICAL, load_train, load_test, add_comparators, score, training_release)

SEED = 42
N_BOOT = int(os.environ.get("S5_NBOOT", "10000"))
MIN_LABELLED = 100
MIN_CLASS = 25
MIN_FRACTION = 0.0
LEVELS = ["Supporting", "Moderate", "Strong", "VeryStrong"]
MODEL_DIR = f"{ROOT}/models"
# Review statuses with no ClinVar star (v1's S6)
ZERO_STAR = {"no_assertion_criteria_provided", "no_classification_provided",
             "no_classifications_from_unflagged_records", "no_interpretation_for_the_single_variant"}


def compared_models():
    """MetaSplice score columns reported side by side: v2.0 has one model."""
    return ["MetaSplice"]


# ----------------------------------------------------------------------------- framework

def lr_levels(prior, c=None):
    """LR needed for one line of evidence at each level (Tavtigian et al. 2018)."""
    odds = prior / (1 - prior)
    if c is None:
        c = (9.0 / odds) ** (4.0 / 3.0)   # 1 Strong + 1 Moderate -> posterior 0.90
    return c, {"Supporting": c ** (1 / 8), "Moderate": c ** (1 / 4),
               "Strong": c ** (1 / 2), "VeryStrong": c}


def post_from_lr(lr, prior):
    o = prior / (1 - prior)
    return lr * o / (1 + lr * o)


PRIOR = 0.0441
C_MAIN, LR_MAIN = lr_levels(PRIOR, c=1124.0)       # the published ClinGen constants


# ----------------------------------------------------------------------------- estimation


def score_grid(s):
    u = np.unique(s)
    if len(u) <= 2000:
        return u
    g = np.concatenate([np.quantile(s, np.linspace(0, 1, 1501)),
                        np.linspace(s.min(), s.max(), 501)])
    return np.unique(g)


def _kth_dist(S, grid, k):
    """Distance from each grid point to its k-th nearest value in sorted array S."""
    k = min(k, len(S))
    out = np.empty(len(grid))
    for i, t in enumerate(grid):
        p = np.searchsorted(S, t)
        lo, hi = max(0, p - k), min(len(S), p + k)
        out[i] = np.partition(np.abs(S[lo:hi] - t), k - 1)[k - 1]
    return out


def windows(grid, s, y, min_labelled, min_class, min_fraction, ref=None):
    """Half-width of the smallest symmetric window around each grid point that holds
    >= min_labelled variants, >= min_class of each class and >= min_fraction of all
    variants -- of the reference variants when `ref` is given (Pejaver's gnomAD rule), else of
    the labelled ones. Computed once on the full data and reused in every bootstrap replicate."""
    eps = _kth_dist(np.sort(s), grid, min_labelled)
    if min_fraction > 0:
        base = np.sort(ref) if ref is not None else np.sort(s)
        eps = np.maximum(eps, _kth_dist(base, grid, int(np.ceil(min_fraction * len(base)))))
    if min_class > 0:
        eps = np.maximum(eps, _kth_dist(np.sort(s[y == 1]), grid, min_class))
        eps = np.maximum(eps, _kth_dist(np.sort(s[y == 0]), grid, min_class))
    return eps


def local_lr(s, y, grid, eps, n_boot, seed):
    """Point estimate and bootstrap distribution of lr at each grid point."""
    sp, sb = np.sort(s[y == 1]), np.sort(s[y == 0])
    NP, NB = len(sp), len(sb)
    lo_p, hi_p = np.searchsorted(sp, grid - eps, "left"), np.searchsorted(sp, grid + eps, "right")
    lo_b, hi_b = np.searchsorted(sb, grid - eps, "left"), np.searchsorted(sb, grid + eps, "right")

    def lr_from(kp, kb):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (kp / NP) / (kb / NB)
        r = np.where(kb == 0, np.where(kp == 0, np.nan, np.inf), r)
        return r

    kp0, kb0 = (hi_p - lo_p).astype(float), (hi_b - lo_b).astype(float)
    est = lr_from(kp0, kb0)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        wp = np.bincount(rng.integers(0, NP, NP), minlength=NP)
        wb = np.bincount(rng.integers(0, NB, NB), minlength=NB)
        cp = np.concatenate([[0], np.cumsum(wp)])
        cb = np.concatenate([[0], np.cumsum(wb)])
        boots[b] = lr_from((cp[hi_p] - cp[lo_p]).astype(float), (cb[hi_b] - cb[lo_b]).astype(float))
    return est, kp0, kb0, boots


def thresholds(grid, lower, upper, lr_need):
    """Pejaver et al. rule applied on the grid.

    PP3 at a level: smallest grid score t such that the lower bound meets the level's LR at
    every grid score >= t. BP4 at a level: largest t such that the upper bound is at or
    below 1/LR at every grid score <= t. None where the level is not reached.
    """
    pp3, bp4 = {}, {}
    for lev in LEVELS:
        ok = np.nan_to_num(lower, nan=-np.inf) >= lr_need[lev]
        if ok[-1]:
            k = len(ok) - 1
            while k > 0 and ok[k - 1]:
                k -= 1
            pp3[lev] = float(grid[k])
        else:
            pp3[lev] = None
        okb = np.nan_to_num(upper, nan=np.inf) <= 1.0 / lr_need[lev]
        if okb[0]:
            k = 0
            while k < len(okb) - 1 and okb[k + 1]:
                k += 1
            bp4[lev] = float(grid[k])
        else:
            bp4[lev] = None
    return pp3, bp4


def calibrate(s, y, name, n_boot=N_BOOT, min_labelled=MIN_LABELLED, min_class=MIN_CLASS,
              min_fraction=MIN_FRACTION, priors=((PRIOR, 1124.0),), ref=None):
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    grid = score_grid(s)
    eps = windows(grid, s, y, min_labelled, min_class, min_fraction,
                  ref=None if ref is None else np.asarray(ref, float)[~np.isnan(np.asarray(ref, float))])
    est, kp, kb, boots = local_lr(s, y, grid, eps, n_boot, SEED)
    lower = np.nanpercentile(np.where(np.isinf(boots), 1e12, boots), 5, axis=0)
    upper = np.nanpercentile(np.where(np.isinf(boots), 1e12, boots), 95, axis=0)
    res = {"name": name, "grid": grid, "eps": eps, "est": est, "lower": lower, "upper": upper,
           "n": len(s), "P": int(y.sum()), "B": int((y == 0).sum()), "by_prior": {}}
    for prior, c in priors:
        cc, need = lr_levels(prior, c)
        res["by_prior"][prior] = (cc, need) + thresholds(grid, lower, upper, need)
    return res


# ----------------------------------------------------------------------------- application


def assign(s, pp3, bp4):
    """Evidence code for each score (NaN -> 'no score'; otherwise highest level reached)."""
    out = np.full(len(s), "Indeterminate", dtype=object)
    out[np.isnan(s)] = "NoScore"
    for lev in LEVELS:                       # ascending, so the highest level wins
        if pp3.get(lev) is not None:
            out[(~np.isnan(s)) & (s >= pp3[lev])] = "PP3_" + lev
        if bp4.get(lev) is not None:
            out[(~np.isnan(s)) & (s <= bp4[lev])] = "BP4_" + lev
    return out


CODES = ([f"PP3_{l}" for l in LEVELS[::-1]] + ["Indeterminate"]
         + [f"BP4_{l}" for l in LEVELS] + ["NoScore"])


def _cp(k, n, alpha):
    """Two-sided (1 - alpha) Clopper-Pearson interval for k successes in n."""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def lr_ci(kp, NP, kb, NB):
    """Interval LR with a conservative exact 95% CI (97.5% Clopper-Pearson per class)."""
    lr = (kp / NP) / (kb / NB) if kb else np.inf
    p_lo, p_hi = _cp(kp, NP, 0.025)
    b_lo, b_hi = _cp(kb, NB, 0.025)
    return lr, p_lo / b_hi, (p_hi / b_lo if b_lo > 0 else np.inf)


def interval_lr(codes, y):
    """Interval LR (TPR/FPR within each evidence interval) with a conservative exact 95% CI.

    Each class's rate gets a 97.5% Clopper-Pearson interval and the LR bounds are the
    ratio of the extreme rates, so the joint coverage is at least 95% (Bonferroni). This
    stays valid when a cell holds zero or a handful of variants, where resampling the
    observed variants cannot express the uncertainty (a zero-benign interval returns an
    infinite LR in every bootstrap replicate).
    """
    NP, NB = int((y == 1).sum()), int((y == 0).sum())
    rows = []
    for code in CODES:
        m = codes == code
        kp, kb = int(m[y == 1].sum()), int(m[y == 0].sum())
        if kp + kb:
            rows.append((code, kp, kb) + lr_ci(kp, NP, kb, NB))
    return rows


def verdicts(code, lr, lo, hi, lr_need=None):
    """(point-estimate verdict, CI verdict, requirement text) for one evidence interval."""
    lr_need = LR_MAIN if lr_need is None else lr_need
    if code.startswith("PP3"):
        need = lr_need[code[4:]]
        return ("meets" if lr >= need else "BELOW",
                "clears" if lo >= need else "INCONCLUSIVE", f">={need:.3g}")
    if code.startswith("BP4"):
        need = 1 / lr_need[code[4:]]
        return ("meets" if lr <= need else "ABOVE",
                "clears" if hi <= need else "INCONCLUSIVE", f"<={need:.3g}")
    return ("", "", "")


def fmt_t(v):
    return "not reached" if v is None else (f"{v:.3g}" if v >= 1e-3 else f"{v:.2e}")


# ----------------------------------------------------------------------------- data


def pathogenic_genes():
    """Ensembl gene IDs with >= 1 P/LP SNV of any region in the 2025-08-31 release, inside the gene's MANE
    transcripts (V02; v1's rule, keyed by gene ID)."""
    return set(pd.read_csv(f"{ROOT}/data/v2_pathogenic_genes.csv").gene_id)


def gene_fold_map():
    """Ensembl gene ID -> fold, as V06 assigned the gene-grouped folds."""
    return json.load(open(f"{MODEL_DIR}/metasplice_v2_folds.json"))


def in_cohort(df, pgenes):
    """Non-canonical region, gnomAD AF < 0.01, gene with >= 1 P/LP in the training release."""
    if df.gnomad_af.isna().any():
        raise ValueError(f"gnomad_af is missing for {int(df.gnomad_af.isna().sum())} rows; the scorer looks it up "
                         f"for every in-scope variant")
    return df.region.isin(NONCANONICAL) & (df.gnomad_af < 0.01) & df.gene_id.isin(pgenes)


def one_star(df):
    """True where the ClinVar record has >= 1 review star."""
    if df.review_status.isna().any():
        raise ValueError(f"review_status is missing for {int(df.review_status.isna().sum())} rows")
    return ~df.review_status.isin(ZERO_STAR)


def was_vus_2025(df):
    """True where the ClinVar ID was uncertain or conflicting in the 2025-08-31 release (S7's resolved VUS)."""
    clnsig = training_release()[2]
    return df.clinvar_id.map(clnsig).fillna("").str.contains("Uncertain|Conflicting")


def load_thresholds(tool, prior=PRIOR, path=None):
    """(pp3, bp4) dicts of score thresholds from S5_thresholds.tsv; unreached levels are None."""
    t = pd.read_csv(path or f"{OUT}/results/S5_thresholds.tsv", sep="\t")
    t = t[(t.tool == tool) & np.isclose(t.prior, prior)]
    pp3, bp4 = {}, {}
    for _, r in t.iterrows():
        v = None if pd.isna(r.score_threshold) else float(r.score_threshold)
        (pp3 if r.level.startswith("PP3") else bp4)[r.level[4:]] = v
    return pp3, bp4


def load_derivation_cohort(one_star_only=False):
    """Training variants in the calibration cohort, with MetaSplice = the out-of-fold score, SpliceAI and CADD."""
    tr = add_comparators(load_train())
    tr["MetaSplice"], tr["SpliceAI"], tr["CADD"] = tr.oof, tr.spliceai_max, tr.cadd_phred
    tr = tr[in_cohort(tr, pathogenic_genes())]
    return (tr[one_star(tr)] if one_star_only else tr).reset_index(drop=True)


def load_validation_cohort(subset="strict", one_star_only=False):
    """Test-set variants in the calibration cohort (subset as in common.load_test), with MetaSplice = the full
    model's score, SpliceAI and CADD."""
    t = add_comparators(load_test(subset))
    t = t[in_cohort(t, pathogenic_genes())]
    if one_star_only:
        t = t[one_star(t)]
    t = t.reset_index(drop=True)
    t["MetaSplice"], t["SpliceAI"], t["CADD"] = score(t), t.spliceai_max, t.cadd_phred
    return t


def one_star_cohorts(subset="strict"):
    """(derivation, validation) restricted to >= 1-star ClinVar records (v1's S6 cohorts)."""
    return load_derivation_cohort(one_star_only=True), load_validation_cohort(subset, one_star_only=True)


# ----------------------------------------------------------------------------- prior (Task 11; AJHG plan Task 3.3)

def _logit(s):
    s = np.clip(np.asarray(s, float), 1e-7, 1 - 1e-7)
    return np.log(s / (1 - s))


def _components(ref, s_p, s_b, bins):
    z_all = _logit(np.concatenate([s_p, s_b, ref]))
    edges = np.unique(np.quantile(z_all, np.linspace(0, 1, bins + 1)))
    f_p = np.histogram(_logit(s_p), edges)[0] + 0.5
    f_b = np.histogram(_logit(s_b), edges)[0] + 0.5
    n = np.histogram(_logit(ref), edges)[0]
    return f_p / f_p.sum(), f_b / f_b.sum(), n


def estimate_prior(ref, s_p, s_b, bins=50, tol=1e-10, max_iter=10000):
    """Mixing weight of the P/LP component in the reference score distribution (EM, fixed components)."""
    f_p, f_b, n = _components(ref, s_p, s_b, bins)
    pi = 0.05
    for _ in range(max_iter):
        w = pi * f_p / (pi * f_p + (1 - pi) * f_b)
        new = float((n * w).sum() / n.sum())
        if abs(new - pi) < tol:
            break
        pi = new
    return pi


def mixture_fit(ref, s_p, s_b, pi, bins=50):
    """Max |empirical CDF - fitted mixture CDF| of the reference over the bin edges."""
    f_p, f_b, n = _components(ref, s_p, s_b, bins)
    fitted = np.cumsum(pi * f_p + (1 - pi) * f_b)
    observed = np.cumsum(n) / n.sum()
    return float(np.max(np.abs(fitted - observed)))


def prior_bootstrap(ref, s_p, s_b, bins=50, n_boot=1000, seed=42, resample_components=True):
    """95% percentile CI for estimate_prior, resampling the reference and, by default, the labelled
    P/LP and B/LB components too (their uncertainty is part of the estimate)."""
    rng = np.random.default_rng(seed)
    ref, s_p, s_b = map(np.asarray, (ref, s_p, s_b))
    est = []
    for _ in range(n_boot):
        r = rng.choice(ref, len(ref))
        p = rng.choice(s_p, len(s_p)) if resample_components else s_p
        b = rng.choice(s_b, len(s_b)) if resample_components else s_b
        est.append(estimate_prior(r, p, b, bins))
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))
