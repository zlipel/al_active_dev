#!/usr/bin/env bash
# Archive legacy MCSC (MODEL_COMPARISON_STELLAR_CURR / MODEL_COMPARISON)
# and older-pipeline artefacts out of runs/<MODEL>/MODELS/ into
# archive/runs_mcsc_YYYYMMDD/, preserving directory structure.
#
# KEEP (untouched):
#   runs/<MODEL>/MODELS/MOE_{PS,NONPS,RF}_iter10_epsilon_kriging_believer_yeoj_upper.{pt,pkl}
#   runs/<MODEL>/MODELS/GPR_iter{N}_epsilon_kriging_believer_yeoj.pt               (AL global GPRs, N=0..10)
#   runs/<MODEL>/MODELS/GPR_iter{N}_epsilon_kriging_believer_yeoj_upper.pt         (multitask "upper" variants)
#   runs/<MODEL>/MODELS/GPR_multitask_iter10_..._FIT_*.png                          (final iter fit plots)
#   runs/<MODEL>/MODELS/GPR_iter{N}_epsilon_kriging_believer_yeoj_FIT_*.png         (per-iter fit plots — reproducible, small)
#   runs/<MODEL>/DIAGNOSTIC/                                                        (al_active_dev diagnostics)
#   runs/<MODEL>/logs/                                                              (submit logs)
#   runs/<MODEL>/features_init.csv, labels_init.csv, seq_init.txt                   (seed data)
#
# ARCHIVE:
#   MODELS/REGIME_DIAGNOSTICS/                          (MCSC OOF + legacy bundles)
#   MODELS/MOE/                                         (MCSC MoE bundle — HPS_URRY only)
#   MODELS/GPR_{diff,exp_density}_iter*.pt              (legacy per-property GPRs)
#   MODELS/GPR_{diff,exp_density}_iter*FIT.png          (legacy per-property fit plots)
#   MODELS/GPR_iter*_epsilon_{constant_liar,front_augmentation,similarity_penalty,standard}*
#   MODELS/GPR_iter*_standard_*                         (pre-epsilon acquisition variants)
#   MODELS/GPR_iter*_epsilon_kriging_believer.pt        (no-yeoj legacy)
#   MODELS/GPR_iter*_epsilon_kriging_believer_FIT_*.png (no-yeoj legacy fit plots)
#   MODELS/GPR_iter*_epsilon_kriging_believer_TEMP.pt   (no-yeoj legacy temp)
#   MODELS/GPR_iter*_epsilon_kriging_believer_log*      (log-transform legacy)
#   MODELS/GPR_iter*_epsilon_kriging_believer_yeoj_MC.pt (Monte Carlo diagnostic variants)
#   MODELS/GPR_iter*_epsilon_kriging_believer_yeoj_TEMP.pt (transient AL-round temp checkpoints)
#   MODELS/GPR_iter*_epsilon_kriging_believer_yeoj_upper_TEMP.pt (same, upper variant)
#
# Scratch (features_gen*, labels_gen*, sequences per iteration) is NEVER touched
# by this script — it operates only on the HOME-side runs/ tree.
#
# Default: dry-run. Pass --execute to actually move.
#
# Usage:
#   ./archive_mcsc.sh                              # dry-run against ./runs
#   ./archive_mcsc.sh --execute                    # actually move
#   ./archive_mcsc.sh --runs-root /path/to/runs    # override runs/ location
#   ./archive_mcsc.sh --archive-root /path/dest    # override archive/ location

set -eo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
RUNS_ROOT="$SCRIPT_DIR/runs"
ARCHIVE_ROOT=""
EXECUTE=false

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --execute)       EXECUTE=true ;;
        --runs-root)     RUNS_ROOT="$2"; shift ;;
        --archive-root)  ARCHIVE_ROOT="$2"; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# //; s/^#//'
            exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ -z "$ARCHIVE_ROOT" ]]; then
    ARCHIVE_ROOT="$(dirname "$RUNS_ROOT")/archive/runs_mcsc_$(date +%Y%m%d)"
fi

if [[ ! -d "$RUNS_ROOT" ]]; then
    echo "Error: runs root does not exist: $RUNS_ROOT" >&2
    exit 1
fi

echo "runs root:    $RUNS_ROOT"
echo "archive root: $ARCHIVE_ROOT"
echo "mode:         $([[ "$EXECUTE" == true ]] && echo EXECUTE || echo 'DRY-RUN (pass --execute to move files)')"
echo

MODELS=(CALVADOS HPS_URRY MPIPI HPS_KR)
n_files=0
n_dirs=0

move_path() {
    local src="$1"
    [[ -e "$src" ]] || return 0
    local rel="${src#"$RUNS_ROOT/"}"
    local dst="$ARCHIVE_ROOT/$rel"
    local dst_dir
    dst_dir="$(dirname "$dst")"

    if [[ -d "$src" && ! -L "$src" ]]; then
        printf '  DIR   %s\n' "$rel"
        n_dirs=$((n_dirs + 1))
    else
        printf '  FILE  %s\n' "$rel"
        n_files=$((n_files + 1))
    fi

    if [[ "$EXECUTE" == true ]]; then
        mkdir -p "$dst_dir"
        mv "$src" "$dst"
    fi
}

for m in "${MODELS[@]}"; do
    d="$RUNS_ROOT/$m"
    [[ -d "$d" ]] || continue
    echo "== $m =="

    # MCSC directories
    move_path "$d/MODELS/REGIME_DIAGNOSTICS"
    move_path "$d/MODELS/MOE"

    # Legacy per-property GPRs (pre-multitask pipeline)
    for f in "$d"/MODELS/GPR_diff_iter*.pt \
             "$d"/MODELS/GPR_exp_density_iter*.pt \
             "$d"/MODELS/GPR_diff_iter*FIT.png \
             "$d"/MODELS/GPR_exp_density_iter*FIT.png; do
        move_path "$f"
    done

    # Legacy acquisition-strategy variants
    for f in "$d"/MODELS/GPR_iter*_epsilon_constant_liar* \
             "$d"/MODELS/GPR_iter*_epsilon_front_augmentation* \
             "$d"/MODELS/GPR_iter*_epsilon_similarity_penalty* \
             "$d"/MODELS/GPR_iter*_epsilon_standard* \
             "$d"/MODELS/GPR_iter*_standard_*; do
        move_path "$f"
    done

    # No-yeoj legacy (early pipeline with different label transform)
    for f in "$d"/MODELS/GPR_iter*_epsilon_kriging_believer.pt \
             "$d"/MODELS/GPR_iter*_epsilon_kriging_believer_FIT_*.png \
             "$d"/MODELS/GPR_iter*_epsilon_kriging_believer_TEMP.pt \
             "$d"/MODELS/GPR_iter*_epsilon_kriging_believer_log*; do
        move_path "$f"
    done

    # Monte Carlo diagnostic variants (from an earlier study)
    for f in "$d"/MODELS/GPR_iter*_epsilon_kriging_believer_yeoj_MC.pt \
             "$d"/MODELS/GPR_iter*_epsilon_kriging_believer_yeoj_MC_FIT_*.png \
             "$d"/MODELS/GPR_iter*_epsilon_kriging_believer_yeoj_MC_TEMP.pt; do
        move_path "$f"
    done

    # Transient AL-round temp checkpoints (iter 10 is done — no resume state needed)
    for f in "$d"/MODELS/GPR_iter*_epsilon_kriging_believer_yeoj_TEMP.pt \
             "$d"/MODELS/GPR_iter*_epsilon_kriging_believer_yeoj_upper_TEMP.pt; do
        move_path "$f"
    done
done

echo
echo "Summary: $n_files files, $n_dirs directories"
if [[ "$EXECUTE" != true ]]; then
    echo "This was a DRY RUN. Rerun with --execute to actually move."
else
    echo "Files moved to: $ARCHIVE_ROOT"
fi
