#!/bin/bash
# Stage EoS thermo.avg files for the report's flagged sequences, mirroring
# runs/<model>/anomalies/poly<globalrow>/.  Run ON THE CLUSTER, then rsync
# $STAGE/ down into the repo's runs/ directory.
set -eo pipefail
REPO_ROOT="${AL_ACTIVE_DEV:-$HOME/PROJECTS/al_active_dev}"
source "$REPO_ROOT/config/cluster.env"
WEBB="${WEBB:-/projects/WEBB/from_zach/MODEL_COMPARISON}"
STAGE="${STAGE:-$HOME/loop_eos_pull}"
echo "Staging into $STAGE ; rsync only thermo.avg"
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly37 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/SIMULATIONS/EOS/poly37/ "$STAGE"/HPS_URRY/anomalies/poly37/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly63 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/SIMULATIONS/EOS/poly63/ "$STAGE"/HPS_URRY/anomalies/poly63/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly85 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/SIMULATIONS/EOS/poly85/ "$STAGE"/HPS_URRY/anomalies/poly85/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly86 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/SIMULATIONS/EOS/poly86/ "$STAGE"/HPS_URRY/anomalies/poly86/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly98 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/SIMULATIONS/EOS/poly98/ "$STAGE"/HPS_URRY/anomalies/poly98/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly158 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$WEBB"/HPS_URRY/GENERATIONS/iteration_1/SIMULATIONS/EOS/poly38/ "$STAGE"/HPS_URRY/anomalies/poly158/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly228 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$WEBB"/HPS_URRY/GENERATIONS/iteration_3/SIMULATIONS/EOS/poly12/ "$STAGE"/HPS_URRY/anomalies/poly228/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly250 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$WEBB"/HPS_URRY/GENERATIONS/iteration_3/SIMULATIONS/EOS/poly34/ "$STAGE"/HPS_URRY/anomalies/poly250/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly300 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$WEBB"/HPS_URRY/GENERATIONS/iteration_4/SIMULATIONS/EOS/poly36/ "$STAGE"/HPS_URRY/anomalies/poly300/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly308 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$WEBB"/HPS_URRY/GENERATIONS/iteration_4/SIMULATIONS/EOS/poly44/ "$STAGE"/HPS_URRY/anomalies/poly308/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly344 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/GENERATIONS/iteration_5/SIMULATIONS/EOS/poly32/ "$STAGE"/HPS_URRY/anomalies/poly344/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly367 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/GENERATIONS/iteration_6/SIMULATIONS/EOS/poly7/ "$STAGE"/HPS_URRY/anomalies/poly367/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly389 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/GENERATIONS/iteration_6/SIMULATIONS/EOS/poly29/ "$STAGE"/HPS_URRY/anomalies/poly389/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly402 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/GENERATIONS/iteration_6/SIMULATIONS/EOS/poly42/ "$STAGE"/HPS_URRY/anomalies/poly402/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly417 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/GENERATIONS/iteration_7/SIMULATIONS/EOS/poly9/ "$STAGE"/HPS_URRY/anomalies/poly417/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly424 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/GENERATIONS/iteration_7/SIMULATIONS/EOS/poly16/ "$STAGE"/HPS_URRY/anomalies/poly424/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly461 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/GENERATIONS/iteration_8/SIMULATIONS/EOS/poly5/ "$STAGE"/HPS_URRY/anomalies/poly461/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly469 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/GENERATIONS/iteration_8/SIMULATIONS/EOS/poly13/ "$STAGE"/HPS_URRY/anomalies/poly469/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly477 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/GENERATIONS/iteration_8/SIMULATIONS/EOS/poly21/ "$STAGE"/HPS_URRY/anomalies/poly477/
mkdir -p "$STAGE"/HPS_URRY/anomalies/poly509 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/HPS_URRY/GENERATIONS/iteration_9/SIMULATIONS/EOS/poly5/ "$STAGE"/HPS_URRY/anomalies/poly509/
mkdir -p "$STAGE"/MPIPI/anomalies/poly139 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_1/SIMULATIONS/EOS/poly19/ "$STAGE"/MPIPI/anomalies/poly139/
mkdir -p "$STAGE"/MPIPI/anomalies/poly142 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_1/SIMULATIONS/EOS/poly22/ "$STAGE"/MPIPI/anomalies/poly142/
mkdir -p "$STAGE"/MPIPI/anomalies/poly151 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_1/SIMULATIONS/EOS/poly31/ "$STAGE"/MPIPI/anomalies/poly151/
mkdir -p "$STAGE"/MPIPI/anomalies/poly158 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_1/SIMULATIONS/EOS/poly38/ "$STAGE"/MPIPI/anomalies/poly158/
mkdir -p "$STAGE"/MPIPI/anomalies/poly203 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_2/SIMULATIONS/EOS/poly35/ "$STAGE"/MPIPI/anomalies/poly203/
mkdir -p "$STAGE"/MPIPI/anomalies/poly222 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_3/SIMULATIONS/EOS/poly6/ "$STAGE"/MPIPI/anomalies/poly222/
mkdir -p "$STAGE"/MPIPI/anomalies/poly310 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_4/SIMULATIONS/EOS/poly46/ "$STAGE"/MPIPI/anomalies/poly310/
mkdir -p "$STAGE"/MPIPI/anomalies/poly317 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_5/SIMULATIONS/EOS/poly5/ "$STAGE"/MPIPI/anomalies/poly317/
mkdir -p "$STAGE"/MPIPI/anomalies/poly318 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_5/SIMULATIONS/EOS/poly6/ "$STAGE"/MPIPI/anomalies/poly318/
mkdir -p "$STAGE"/MPIPI/anomalies/poly321 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_5/SIMULATIONS/EOS/poly9/ "$STAGE"/MPIPI/anomalies/poly321/
mkdir -p "$STAGE"/MPIPI/anomalies/poly324 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_5/SIMULATIONS/EOS/poly12/ "$STAGE"/MPIPI/anomalies/poly324/
mkdir -p "$STAGE"/MPIPI/anomalies/poly408 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_7/SIMULATIONS/EOS/poly0/ "$STAGE"/MPIPI/anomalies/poly408/
mkdir -p "$STAGE"/MPIPI/anomalies/poly409 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_7/SIMULATIONS/EOS/poly1/ "$STAGE"/MPIPI/anomalies/poly409/
mkdir -p "$STAGE"/MPIPI/anomalies/poly425 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_7/SIMULATIONS/EOS/poly17/ "$STAGE"/MPIPI/anomalies/poly425/
mkdir -p "$STAGE"/MPIPI/anomalies/poly537 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_9/SIMULATIONS/EOS/poly33/ "$STAGE"/MPIPI/anomalies/poly537/
mkdir -p "$STAGE"/MPIPI/anomalies/poly582 && rsync -a --prune-empty-dirs --include='*/' --include='thermo.avg' --exclude='*' "$SCRATCH_AL"/MPIPI/GENERATIONS/iteration_10/SIMULATIONS/EOS/poly30/ "$STAGE"/MPIPI/anomalies/poly582/
echo "done. Now: rsync -av $STAGE/ <LOCAL_REPO>/runs/"
