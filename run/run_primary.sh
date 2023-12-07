#!/bin/bash --login

# GLOBAL VAR ####
[[ -z $SEED ]] && SEED=0
[[ -z $NUM_RUNS ]] && NUM_RUNS=5
[[ -z $DRY_RUN ]] && DRY_RUN=0
[[ -z $RUN_MODE ]] && RUN_MODE=production
[[ -z $USE_WANDB ]] && USE_WANDB=True

NETWORKS=(
    BioGRID
    HumanNet
)
LABELS=(
    DisGeNET
    GOBP
)
MODELS=(
    LabelProp
    # Logistic Regression
    LogReg+Adj
    LogReg+Node2vec
    LogReg+LapEigMap
    # GNN
    GAT
    GCN
    # GNN with C&S
    GAT+CS
    GCN+CS
    # GNN with bag of tricks (Node2vec features, label reuse, correct and smooth)
    GAT+BoT
    GCN+BoT
)

CUR_FILE_DIR=$(dirname $(realpath $0))
HOME_DIR=$(dirname $CUR_FILE_DIR)

echo SEED=$SEED
echo NUM_RUNS=$NUM_RUNS
echo DRY_RUN=$DRY_RUN
echo RUN_MODE=$RUN_MODE
echo USE_WANDB=$USE_WANDB
echo CUR_FILE_DIR=$CUR_FILE_DIR
echo HOME_DIR=$HOME_DIR
echo
#################

base_script="python main.py run_mode=${RUN_MODE} wandb.use=${USE_WANDB} num_runs=1"

launch() {
    local network=$1 label=$2 model=$3 script run_seed

    [[ $model == *"+tuned" ]] && model_key="+model_tuned" || model_key="model"
    script="${base_script} dataset.network=${network} dataset.label=${label} ${model_key}=${model}"

    for run_seed in $(seq $SEED $(( $NUM_RUNS + $SEED - 1 ))); do
        seeded_script="${script} seed=${run_seed}"
        echo $seeded_script && [[ $DRY_RUN == 0 ]] && eval $seeded_script
    done
}

for network in ${NETWORKS[@]}; do
    for label in ${LABELS[@]}; do
        for model in ${MODELS[@]}; do
            launch $network $label $model
        done

        # Dataset-specific-tuned GNNs
        for gnn in GCN GAT; do
            launch $network $label ${network}-${label}-${gnn}+tuned
        done
    done
done

wait
