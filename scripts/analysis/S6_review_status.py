"""S6 -- Calibration restricted to ClinVar records with >= 1 review star (Amendment 3).

S5's procedure (calib.calibrate; prior 0.0441; windows >= 100 labelled and >= 25 of each class; 10,000
bootstrap iterations) on the derivation and validation records whose review status has at least one
star. Thresholds are listed beside S5's. As in v1, S6's thresholds are the main candidate thresholds
used by S7 to S9. Denominators are those Amendment 3 records.

Writes results/S6_review_status.txt, S6_thresholds.tsv and S6_validation_intervals.tsv.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from common import OUT, Tee
from calib import (LEVELS, PRIOR, assign, calibrate, fmt_t, interval_lr, load_thresholds, one_star_cohorts,
                   verdicts)

TOOLS = ["MetaSplice", "SpliceAI"]
# Amendment 3: (n, P/LP) of each cohort for each tool's scored variants
EXPECTED = {"derivation": {"MetaSplice": (294612, 3018), "SpliceAI": (292638, 3004)},
            "validation": {"MetaSplice": (24049, 231), "SpliceAI": (23928, 230)}}


def main():
    log = Tee(f"{OUT}/results/S6_review_status.txt")
    log("=" * 88)
    log("S6 -- CALIBRATION ON CLINVAR RECORDS WITH >= 1 REVIEW STAR (Amendment 3)")
    log("=" * 88)
    der, val = one_star_cohorts()
    for name, d in [("derivation", der), ("validation", val)]:
        for t in TOOLS:
            got = (int(d[t].notna().sum()), int(d.label[d[t].notna()].sum()))
            if got != EXPECTED[name][t]:
                raise SystemExit(f"{name} / {t}: {got} differs from Amendment 3's {EXPECTED[name][t]}")
    log(f"derivation >= 1 star: n={len(der):,} P={int(der.label.sum()):,}; validation >= 1 star: n={len(val):,} "
        f"P={int(val.label.sum()):,} (denominators equal Amendment 3)")
    thr_rows, val_rows = [], []
    for t in TOOLS:
        res = calibrate(der[t].values.astype(float), der.label.values, t)
        c, need, pp3, bp4 = res["by_prior"][PRIOR]
        s5_pp3, s5_bp4 = load_thresholds(t)
        log(f"\n{t}: n={res['n']:,} (P={res['P']:,})")
        log(f"  {'level':<16}{'S6 (>=1 star)':>16}{'S5 (all)':>14}")
        for lev in LEVELS[::-1]:
            log(f"  PP3_{lev:<12}{fmt_t(pp3[lev]):>16}{fmt_t(s5_pp3.get(lev)):>14}")
        for lev in LEVELS:
            log(f"  BP4_{lev:<12}{fmt_t(bp4[lev]):>16}{fmt_t(s5_bp4.get(lev)):>14}")
        for lev in LEVELS:
            thr_rows += [(t, PRIOR, c, "PP3_" + lev, need[lev], pp3[lev]), (t, PRIOR, c, "BP4_" + lev, 1 / need[lev], bp4[lev])]
        codes = assign(val[t].values.astype(float), pp3, bp4)
        log(f"  validation (>=1 star):  {'interval':<16}{'P':>5}{'B':>7}{'LR':>11}   exact 95% CI         needed    point  CI")
        for code, kp, kb, lr, lo, hi in interval_lr(codes, val.label.values):
            point, ci, ns = verdicts(code, lr, lo, hi)
            his = "inf" if hi == float("inf") else f"{hi:.3g}"
            log(f"  {'':<24}{code:<16}{kp:>5}{kb:>7}{lr:>11.3g}   [{lo:.3g}, {his}]".ljust(80) + f"{ns:<10}{point:<7}{ci}")
            val_rows.append((t, code, kp, kb, lr, lo, hi, point, ci))
    pd.DataFrame(thr_rows, columns=["tool", "prior", "c", "level", "LR_needed", "score_threshold"]).to_csv(
        f"{OUT}/results/S6_thresholds.tsv", sep="\t", index=False)
    pd.DataFrame(val_rows, columns=["tool", "interval", "P", "B", "LR", "lo95_exact", "hi95_exact", "point_verdict",
                                    "ci_verdict"]).to_csv(f"{OUT}/results/S6_validation_intervals.tsv", sep="\t", index=False)
    log("\nwrote results/S6_thresholds.tsv and results/S6_validation_intervals.tsv")
    log.close()


if __name__ == "__main__":
    main()
