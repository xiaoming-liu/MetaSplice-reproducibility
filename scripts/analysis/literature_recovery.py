"""Literature recovery (Amendment 3): the share of the 697 literature-confirmed splice-altering variants
(all P/LP, none in training) that MetaSplice v2.0 scores >= 0.1, >= 0.2 and >= 0.5, overall and by
region, with the deep-intronic sensitivity set (Deep Intronic 130; sensitivity set 114).

Writes results/literature_recovery.txt.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, REGIONS, Tee, deep_intronic_sensitivity, load_literature, score

THRESHOLDS = (0.1, 0.2, 0.5)
EXPECTED = {"all regions": 697, "Deep Intronic": 130, "Deep Intronic, sensitivity set": 114}    # Amendment 3


def main():
    log = Tee(f"{OUT}/results/literature_recovery.txt")
    log("=" * 88)
    log("LITERATURE RECOVERY -- MetaSplice v2.0 on literature-confirmed splice-altering variants (Amendment 3)")
    log("=" * 88)
    lit = load_literature()
    lit["MetaSplice"] = score(lit)
    groups = [("all regions", lit.index == lit.index)]
    for reg in REGIONS:
        groups.append((reg, (lit.region == reg).values))
        if reg == "Deep Intronic":
            groups.append(("Deep Intronic, sensitivity set", deep_intronic_sensitivity(lit).values))
    for name, n in EXPECTED.items():
        got = int(dict(groups)[name].sum())
        if got != n:
            raise SystemExit(f"{name}: {got} variants; Amendment 3 recorded {n}")
    log(f"  {'subset':<32}{'n':>6}" + "".join(f"{'>= ' + str(t):>16}" for t in THRESHOLDS))
    for name, m in groups:
        s = lit.MetaSplice.values[m]
        log(f"  {name:<32}{len(s):>6}" + "".join(f"{f'{int((s >= t).sum())} ({(s >= t).mean():.1%})':>16}" for t in THRESHOLDS))
    log.close()


if __name__ == "__main__":
    main()
