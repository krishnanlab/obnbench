#!/bin/bash --login
#
# Example:
# $ sh GCN BioGRID
# $ sh LogReg+Adj BioGRID

# GLOBAL VAR ####
[[ -z $SEED ]] && SEED=0
[[ -z $NUM_RUNS ]] && NUM_RUNS=5
[[ -z $DRY_RUN ]] && DRY_RUN=0
[[ -z $RUN_MODE ]] && RUN_MODE=production
[[ -z $USE_WANDB ]] && USE_WANDB=True
#################

model=$1
network=$2

basescript="python main.py run_mode=${RUN_MODE} wandb.use=${USE_WANDB} num_runs=${NUM_RUNS} seed=${SEED} "
basescript+="model=${model} model.name=${model}+CS model.post_cands.enable=True dataset.network=${network}"

script="${basescript} dataset.label=GOBP & "
script+="${basescript} dataset.label=DisGeNET & "
script+="${basescript} dataset.label=DISEASES &"

echo $script
[[ $DRY_RUN == 0 ]] && eval $script

wait
