#!/bin/bash
# eos_calc.sh — submit an EOS analysis job.
#
# Usage:
#   eos_calc.sh MODEL NBOOT ITER [RUN_TAG]       # init (ITER=0) or AL iteration
#   eos_calc.sh MODEL NBOOT --validation SCOPE   # post-hoc validation scope
#
# RUN_TAG (iteration only): parallel-policy run tag matching the suffix the AL
# loop appended to the candidate/seq filenames (e.g. soft|hard|global). Pass the
# same tag used for make_eos.sh --run_tag so this reads seq_gen{ITER}_{tag}.txt.
# Default: none (production naming). Not applicable to validation scopes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=$1
NBOOT=$2

if [[ "${3:-}" == "--validation" ]]; then
    SCOPE="$4"
    SCOPE_SPEC="validation:$SCOPE"     # process_eos_sims.sh decodes this token
    TAG="val_${SCOPE,,}"
    RUN_TAG=""
else
    SCOPE_SPEC="$3"                    # integer iteration (0 = init)
    TAG="$3"
    RUN_TAG="${4:-}"                   # optional parallel-policy run tag
fi
[[ -n "$RUN_TAG" ]] && TAG="${TAG}_${RUN_TAG}"

mkdir -p simlogs/EOS
sbatch \
  --job-name="eos_${MODEL}_${TAG}" \
  --output="simlogs/EOS/process_eos_${MODEL}_${TAG}.out" \
  --error="simlogs/EOS/process_eos_${MODEL}_${TAG}.err" \
  "${SCRIPT_DIR}/process_eos_sims.sh" "$MODEL" "$NBOOT" "$SCOPE_SPEC" "$RUN_TAG"
