cd ~/PROJECTS/al_active_dev
mkdir -p runs
for model in MPIPI CALVADOS HPS_URRY; do
  [[ -d ~/PROJECTS/MODEL_COMPARISON/$model ]] && \
    mv ~/PROJECTS/MODEL_COMPARISON/$model runs/$model
done