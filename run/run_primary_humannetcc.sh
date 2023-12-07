#!/bin/bash --login

# GLOBAL VAR ####
[[ -z $SEED ]] && SEED=0
[[ -z $NUM_RUNS ]] && NUM_RUNS=5
[[ -z $DRY_RUN ]] && DRY_RUN=0
[[ -z $RUN_MODE ]] && RUN_MODE=production
[[ -z $USE_WANDB ]] && USE_WANDB=True

NETWORK=HumanNet_CC  # NOTE: might be deprecated beyond obnbdata-0.1.0
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

base_script="python main.py run_mode=${RUN_MODE} wandb.use=${USE_WANDB} num_runs=1 group=humannetcc"

launch() {
    local label=$1 model=$2 randsplit_flag=$3 script run_seed

    [[ $model == *"+tuned" ]] && model_key="+model_tuned" || model_key="model"
    script="${base_script} dataset.network=${NETWORK} dataset.label=${label} ${model_key}=${model}"
    # [[ $randsplit_flag == 1 ]] && script+=" dataset.random_split=True group=humannetcc_randsplit" || script+=" group=humannetcc"
    [[ $randsplit_flag == 1 ]] && script+=" dataset.random_split=True name_tag=randsplit"

    for run_seed in $(seq $SEED $(( $NUM_RUNS + $SEED - 1 ))); do
        seeded_script="${script} seed=${run_seed}"
        echo $seeded_script && [[ $DRY_RUN == 0 ]] && eval $seeded_script
    done
}

# for randsplit_flag in 0 1; do
#     for label in ${LABELS[@]}; do
#         for model in ${MODELS[@]}; do
#             launch $label $model $randsplit_flag
#         done
# 
#         # Dataset-specific-tuned GNNs
#         for gnn in GCN GAT; do
#             launch $label ${NETWORK}-${label}-${gnn}+tuned $randsplit_flag
#         done
#     done
# done

# METHOD=LabelProp

# launch DisGeNET ${METHOD} 0 && launch GOBP ${METHOD} 0 && launch DisGeNET ${METHOD} 1 && launch GOBP ${METHOD} 1

# launch DisGeNET ${METHOD} 0 &
# launch GOBP ${METHOD} 0 &
# launch DisGeNET ${METHOD} 1 &
# launch GOBP ${METHOD} 1 &

# launch DisGeNET HumanNet-DisGeNET-${METHOD}+tuned 0 &
# launch GOBP HumanNet-GOBP-${METHOD}+tuned 0 &
# launch DisGeNET HumanNet-DisGeNET-${METHOD}+tuned 1 &
# launch GOBP HumanNet-GOBP-${METHOD}+tuned 1 &

wait
