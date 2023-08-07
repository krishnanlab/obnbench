#!/bin/bash --login
#
# Example (run under the project root directory):
# $ sh run/run_abl_gnn_features.sh GCN BioGRID Node2vec

# GLOBAL VAR ####
[[ -z $SEED ]] && SEED=0
[[ -z $NUM_RUNS ]] && NUM_RUNS=5
[[ -z $DRY_RUN ]] && DRY_RUN=0
[[ -z $RUN_MODE ]] && RUN_MODE=production
[[ -z $USE_WANDB ]] && USE_WANDB=True
[[ -z $PARALLEL ]] && PARALLEL=1
#################

model=$1
network=$2
feature=$3

basescript="python main.py run_mode=${RUN_MODE} wandb.use=${USE_WANDB} num_runs=${NUM_RUNS} seed=${SEED} "
basescript+="model=${model} dataset.network=${network} dataset.node_encoders=${feature} model.name=${model}+${feature}"

if [[ $PARALLEL == 1 ]]; then
    script="${basescript} dataset.label=GOBP & "
    script+="${basescript} dataset.label=DisGeNET & "
    script+="${basescript} dataset.label=DISEASES &"
    echo $script && [[ $DRY_RUN == 0 ]] && eval $script
else
    for label in GOBP DisGeNET DISEASES; do
        script="${basescript} dataset.label=${label}"
        echo $script && [[ $DRY_RUN == 0 ]] && eval $script
    done
fi

wait
