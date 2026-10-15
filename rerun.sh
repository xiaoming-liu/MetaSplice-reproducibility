#!/usr/bin/env bash
# Rerun the analyses of this package from its data, in dependency order, writing results/ and figures/.
#
#   ./rerun.sh                 run everything that the data present allow
#   PYTHON=python3.12 ./rerun.sh
#
# Most analyses compare MetaSplice with SpliceAI and CADD, whose scores are not redistributed here. They run only after
# scripts/comparators/lookup_comparators.py (and, for the analyses that use raw SpliceAI scores, run_spliceai_raw.py) has
# rebuilt those scores and they match originals.md5. Likewise, the SpliceVarDB analyses need the classifications restored
# by scripts/splicevardb/restore_splicevardb.py. Analyses whose inputs are absent are listed as skipped. Exits non-zero
# if any analysis fails.
set -uo pipefail
PY="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=env.sh
. "$ROOT/env.sh"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
if [[ ! -f data/v2_features_training.csv ]]; then
  echo "data/ is missing: run 'python get_data.py' first (it downloads and checks the data from the release)" >&2
  exit 1
fi
mkdir -p results/logs figures

# Comparator scores present and equal to the originals? (precomputed SpliceAI and CADD; raw SpliceAI)
check() { grep -E "$1" scripts/comparators/originals.md5 | md5sum -c --quiet --status 2>/dev/null; }
HAVE_CMP=0; check 'v2_comparators|v2_gnomad_spliceai|v2_features_pooled|v2_features_fresh|S12_mechanism' && HAVE_CMP=1
HAVE_RAW=0; check 'v2_spliceai_raw' && HAVE_RAW=1
HAVE_SVDB=0; check 'splicevardb' && HAVE_SVDB=1
state() { [[ $1 == 1 ]] && echo present || echo absent; }
echo "SpliceAI and CADD scores $(state $HAVE_CMP); raw SpliceAI $(state $HAVE_RAW); SpliceVarDB classes $(state $HAVE_SVDB)"

# name | script and arguments | needs: none, or a comma-separated list of cmp (precomputed SpliceAI and CADD),
# raw (raw SpliceAI) and svdb (SpliceVarDB classes)
STEPS=(
  "literature_recovery|scripts/analysis/literature_recovery.py|none"
  "S11_reference_scores|scripts/analysis/S11_reference_scores.py|none"
  "X2_distance_weighting_eval|scripts/analysis/X2_distance_weighting_eval.py|none"
  "C0_cohorts|scripts/analysis/C0_cohorts.py|cmp,svdb"
  "A1_noncanonical|scripts/analysis/A1_noncanonical.py|cmp"
  "A3b_gene_heldout_retrain|scripts/analysis/A3b_gene_heldout_retrain.py|cmp"
  "A6_splicevardb|scripts/analysis/A6_splicevardb.py|cmp,svdb"
  "S5_clingen_calibration|scripts/analysis/S5_clingen_calibration.py|cmp"
  "S5_calibration_figure|scripts/analysis/S5_calibration_figure.py|none"
  "S6_review_status|scripts/analysis/S6_review_status.py|cmp"
  "S7_resolved_vus|scripts/analysis/S7_resolved_vus.py|cmp"
  "S8_splicevardb_endpoint|scripts/analysis/S8_splicevardb_endpoint.py|cmp,svdb"
  "S9_per_gene|scripts/analysis/S9_per_gene.py|cmp"
  "S10_head_to_head|scripts/analysis/S10_head_to_head.py|cmp"
  "S11_prior|scripts/analysis/S11_prior.py|cmp"
  "S11_prior_diagnostic|scripts/analysis/S11_prior_diagnostic.py|cmp"
  "S11_recalibrate|scripts/analysis/S11_recalibrate.py|cmp"
  "S12_mechanism|scripts/analysis/S12_mechanism.py|cmp"
  "X1_distance_control|scripts/analysis/X1_distance_control.py|cmp"
  "C1_fresh_cohorts|scripts/analysis/C1_fresh_cohorts.py|cmp"
  "F1_fresh_validation|scripts/analysis/F1_fresh_validation.py|cmp"
  "X3_supporting_rule|scripts/analysis/X3_supporting_rule.py|cmp"
  "X3_supporting_rule_raw|scripts/analysis/X3_supporting_rule.py --spliceai raw|cmp,raw"
  "X3_supporting_rule_raw_allgenes|scripts/analysis/X3_supporting_rule.py --spliceai raw_allgenes|cmp,raw"
  "P1_inputs_baselines_ablations|scripts/analysis/P1_inputs_baselines_ablations.py|cmp"
  "P1_inputs_baselines_ablations_pooled|scripts/analysis/P1_inputs_baselines_ablations.py --cohort pooled|cmp"
  "P2_importance_within_gene|scripts/analysis/P2_importance_within_gene.py|cmp"
  "P2_importance_within_gene_pooled|scripts/analysis/P2_importance_within_gene.py --cohort pooled|cmp"
  "P3_vus_rule|scripts/analysis/P3_vus_rule.py|cmp,raw"
  "Q1_pooled_comparison|scripts/analysis/Q1_pooled_comparison.py|cmp"
  "Q2_pooled_calibration_rule|scripts/analysis/Q2_pooled_calibration_rule.py|cmp,raw"
  "figures_ajhg|scripts/figures/make_figs_v2.py|none"
  "figure1_design|scripts/figures/make_fig1_design_v3.py|none"
)

failed=(); skipped=()
for s in "${STEPS[@]}"; do
  IFS='|' read -r name cmd needs <<< "$s"
  missing=""
  for n in ${needs//,/ }; do
    [[ $n == cmp && $HAVE_CMP == 0 ]] || [[ $n == raw && $HAVE_RAW == 0 ]] || [[ $n == svdb && $HAVE_SVDB == 0 ]] \
      && missing+="$n "
  done
  if [[ -n $missing ]]; then skipped+=("$name (needs ${missing% })"); continue; fi
  echo "=== $(date +%H:%M) $name"
  # shellcheck disable=SC2086
  if ! "$PY" -u $cmd > "results/logs/$name.log" 2>&1; then
    echo "  FAILED: $name (results/logs/$name.log)"; failed+=("$name")
  fi
done
echo "=== $(date +%H:%M) unit tests"
"$PY" -m pytest -q -p no:cacheprovider tests > results/logs/tests.log 2>&1 || { echo "  FAILED: tests (results/logs/tests.log)"; failed+=(tests); }

if ((${#skipped[@]})); then
  echo; echo "Skipped ${#skipped[@]} analyses whose inputs are not present (README.md says how to rebuild them):"
  printf '  %s\n' "${skipped[@]}"
  [[ $HAVE_CMP == 0 ]] && echo "  cmp:  $PY scripts/comparators/lookup_comparators.py --spliceai DIR --cadd FILE --bundle DIR"
  [[ $HAVE_RAW == 0 ]] && echo "  raw:  $PY scripts/comparators/run_spliceai_raw.py --spliceai-env ENV --bundle DIR"
  [[ $HAVE_SVDB == 0 ]] && echo "  svdb: $PY scripts/splicevardb/restore_splicevardb.py --savnet XLSX --misplice XLSX --mfass TXT"
fi
if ((${#failed[@]})); then echo; echo "FAILED: ${failed[*]}"; exit 1; fi
echo; echo "all analyses that could run completed"
