"""A1 -- Headline and comparators on the v2.0 strict test set (docs/AJHG_COMPARISON_PROTOCOL.md, Amendment 3).

v1's A1 on v2.0, with the methods Amendment 3 fixes:
  - all non-exonic, canonical and non-canonical variants: each tool's auPRC on the variants it scores,
    with a stratified bootstrap 95% CI (common.boot_ci, 1,000 iterations, seed 42), and its AUROC;
  - non-canonical variants: the paired differences of the primary comparison (common.paired_boot_diff);
  - each non-canonical region, paired against SpliceAI and against CADD, plus the deep-intronic
    sensitivity row (Deep Intronic, neither canonical nor exonic in any GENCODE v50 transcript).
Tools: MetaSplice v2.0 (common.score, the full model), SpliceAI v1.3 and CADD v1.7.

Writes results/A1_noncanonical.txt and .tsv, and figures/A1_pr_noncanonical.png.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score
from common import (CANONICAL, NONCANONICAL, OUT, Tee, add_comparators, auprc, boot_ci, deep_intronic_sensitivity,
                    load_test, paired_boot_diff, score)

TOOLS = [("MetaSplice v2.0", "MetaSplice"), ("SpliceAI", "spliceai_max"), ("CADD", "cadd_phred")]
NOT_INTERPRETED = {"Branch Point Region": "5 P/LP: reported, not interpreted"}


def main():
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    log = Tee(f"{OUT}/results/A1_noncanonical.txt")
    log("=" * 96)
    log("A1 -- METASPLICE v2.0 AND COMPARATORS ON THE STRICT TEST SET (Amendment 3)")
    log("=" * 96)
    t = add_comparators(load_test("strict"))
    t["MetaSplice"] = score(t)
    can, nc = t[t.region.isin(CANONICAL)], t[t.region.isin(NONCANONICAL)]
    log(f"strict test set: N={len(t):,}  P={int(t.label.sum()):,}  B={int((t.label == 0).sum()):,}")
    log(f"  canonical:     N={len(can):,}  P={int(can.label.sum()):,}")
    log(f"  non-canonical: N={len(nc):,}  P={int(nc.label.sum()):,}")
    rows = []

    log("\nEach tool on the variants it scores: auPRC [stratified bootstrap 95% CI], AUROC")
    for tag, sub in [("all non-exonic", t), ("canonical", can), ("non-canonical", nc)]:
        log(f"\n  {tag} (N={len(sub):,}, P={int(sub.label.sum()):,})")
        log(f"  {'tool':<16}{'n scored':>10}{'cov':>8}{'P':>7}{'auPRC':>9}{'95% CI':>20}{'AUROC':>9}")
        for name, col in TOOLS:
            s = sub[col].values.astype(float)
            ok = ~np.isnan(s)
            y = sub.label.values[ok]
            ap, lo, hi = boot_ci(sub.label.values, s)
            ar = roc_auc_score(y, s[ok]) if 0 < y.sum() < len(y) else np.nan
            rows.append(dict(analysis="marginal", subset=tag, tool=name, n=int(ok.sum()), P=int(y.sum()),
                             auprc=ap, lo=lo, hi=hi, auroc=ar))
            log(f"  {name:<16}{ok.sum():>10,}{ok.mean():>8.1%}{int(y.sum()):>7,}{ap:>9.4f}   [{lo:.4f}, {hi:.4f}]{ar:>9.4f}")

    log("\nPaired, MetaSplice v2.0 minus comparator, on the variants both score (1,000 iterations, seed 42)")
    subsets = [("all non-canonical", nc)] + [(r, nc[nc.region == r]) for r in NONCANONICAL] + \
              [("Deep Intronic, sensitivity set", nc[deep_intronic_sensitivity(nc)])]
    for tag, sub in subsets:
        for name, col in TOOLS[1:]:
            ok = sub[col].notna().values
            y = sub.label.values[ok]
            d, lo, hi, n = paired_boot_diff(sub.label.values, sub["MetaSplice"].values, sub[col].values)
            a_ms, a_ot = auprc(y, sub["MetaSplice"].values[ok]), auprc(y, sub[col].values[ok])
            note = NOT_INTERPRETED.get(tag, "")
            rows.append(dict(analysis="paired", subset=tag, tool=f"MetaSplice v2.0 vs {name}", n=n, P=int(y.sum()),
                             auprc=a_ms, auprc_comparator=a_ot, diff=d, lo=lo, hi=hi, note=note))
            log(f"  {tag:<32} vs {name:<9} n={n:>7,} P={int(y.sum()):>4}  {a_ms:.4f} vs {a_ot:.4f}  "
                f"diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + (f"  ({note})" if note else ""))
    pd.DataFrame(rows).to_csv(f"{OUT}/results/A1_noncanonical.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (sub, title) in zip(axes, [(t, f"All non-exonic (N={len(t):,})"), (nc, f"Non-canonical (N={len(nc):,})")]):
        for name, col in TOOLS:
            s = sub[col].values.astype(float)
            ok = ~np.isnan(s)
            y = sub.label.values[ok]
            prec, rec, _ = precision_recall_curve(y, s[ok])
            ax.plot(rec, prec, lw=1.8, label=f"{name} {auprc(y, s[ok]):.3f} ({ok.mean():.1%} scored)")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/A1_pr_noncanonical.png", dpi=200)
    log(f"\nwrote results/A1_noncanonical.tsv ({len(rows)} rows) and figures/A1_pr_noncanonical.png")
    log.close()


if __name__ == "__main__":
    main()
