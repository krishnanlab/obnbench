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
    LogReg+SVD
    LogReg+LINE1
    LogReg+LINE2
    LogReg+Node2vec
    LogReg+Walklets
    LogReg+LapEigMap
    # GNN
    GAT
    GCN
    GIN
    GatedGCN
    SAGE
    # GNN with bag of tricks (Node2vec features, label reuse, correct and smooth)
    GAT+BoT
    GCN+BoT
    GIN+BoT
    GatedGCN+BoT
    SAGE+BoT
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

parse_arg() {
    local name=$1
    local arg=$2

    if [[ $name == network ]]; then
        position=first
        valid_args=${NETWORKS[@]}
    elif [[ $name == label ]]; then
        position=second
        valid_args=${LABELS[@]}
    elif [[ $name == model ]]; then
        position=third
        valid_args=${MODELS[@]}
    else
        >&2 echo Unknwon arg type ${name}. Please fix!
        exit 1
    fi

    # Check if argument parsed
    [[ -z $arg ]]  && >&2 echo Please specify ${name} as the ${position} argument && exit 1

    # Check if argument is valid
    [[ $arg == all ]] && echo ${valid_args[@]} && return
    for valid_arg in ${valid_args[@]}; do
        [[ $arg == $valid_arg ]] && echo $arg && return
    done
    >&2 echo Invalid ${name} specification: ${arg} && exit 1
}

launch() {
    local network=$1 label=$2 model=$3 script run_seed

    script="${base_script} dataset.network=${network} dataset.label=${label} model=${model}"

    for run_seed in $(seq $SEED $(( $NUM_RUNS + $SEED - 1 ))); do
        seeded_script="${script} seed=${run_seed}"
        echo $seeded_script && [[ $DRY_RUN == 0 ]] && eval $seeded_script
    done
}

networks=$(parse_arg network $1)
labels=$(parse_arg label $2)
models=$(parse_arg model $3)

for network in ${networks[@]}; do
    for label in ${labels[@]}; do
        for model in ${models[@]}; do
            launch $network $label $model
        done
    done
done

wait
