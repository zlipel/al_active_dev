#!/bin/bash
# diff_calc.sh — submit a diffusivity analysis job.
#
# Usage:
#   diff_calc.sh MODEL ITER [RUN_TAG] [INNER OMP NSEQ]               # init (ITER=0) or AL iteration
#   diff_calc.sh MODEL --validation SCOPE [INNER OMP NSEQ]           # post-hoc validation scope
#
# RUN_TAG (iteration only): parallel-policy run tag matching the suffix the AL
# loop appended to the candidate/seq filenames (e.g. soft|hard|global). Pass the
# same tag used for make_diff.sh --run_tag so this reads seq_gen{ITER}_{tag}.txt.
# Default: none (production naming). Not applicable to validation scopes.
# NOTE: RUN_TAG sits between ITER and the optional INNER/OMP/NSEQ tuning knobs;
# if you pass those positionally, prepend the run tag (or "" for none) first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=$1

if [[ "${2:-}" == "--validation" ]]; then
    SCOPE="$3"
    SCOPE_SPEC="validation:$SCOPE"     # process_diff_sims.sh decodes this token
    TAG="val_${SCOPE,,}"
    RUN_TAG=""
    INNER_JOBS=${4:-4}
    OMP_THREADS=${5:-4}
    NSEQ_JOBS=${6:-6}
else
    SCOPE_SPEC="$2"                    # integer iteration (0 = init)
    TAG="$2"
    RUN_TAG="${3:-}"                   # optional parallel-policy run tag
    INNER_JOBS=${4:-4}
    OMP_THREADS=${5:-4}
    NSEQ_JOBS=${6:-6}
fi
[[ -n "$RUN_TAG" ]] && TAG="${TAG}_${RUN_TAG}"

mkdir -p simlogs/DIFF
sbatch \
  --job-name="diff_${MODEL}_${TAG}" \
  --output="simlogs/DIFF/process_diff_${MODEL}_${TAG}.out" \
  --error="simlogs/DIFF/process_diff_${MODEL}_${TAG}.err" \
  "${SCRIPT_DIR}/process_diff_sims.sh" "$MODEL" "$SCOPE_SPEC" "$INNER_JOBS" "$OMP_THREADS" "$NSEQ_JOBS" "$RUN_TAG"
