# Source this file (". ./env.sh") before running a script of this package by hand; rerun.sh does it itself.
# The analyses read the scorer and the training release's ClinVar records from these paths.
_ms_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export METASPLICE_V2_ROOT="$_ms_root"
export METASPLICE_V2_SCORER="$_ms_root/scripts/score_v2.py"
export METASPLICE_V2_CLINVAR_2025_GRCH38="$_ms_root/data/clinvar/clinvar_20250831.GRCh38.summary.gz"
export METASPLICE_V2_CLINVAR_2025_GRCH37="$_ms_root/data/clinvar/clinvar_20250831.GRCh37.summary.gz"
unset _ms_root
