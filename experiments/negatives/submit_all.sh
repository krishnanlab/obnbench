#!/bin/bash --login

# -------------------------
TEMPLATE=job_template.sb
PROJECT_SLURM_DIR=../../run
EXP_DIR=experiments/negatives
NUM_REPS=5

NETWORKS=(
    BioGRID
    HumanNet
)
LABELS=(
    DisGeNET
    DISEASES
    GOBP
)
MODELS=(
    GCN
    GIN
    GAT
    GraphSAGE
)

BASE_SCRIPT="/bin/time -v python main.py homedir=${EXP_DIR}"
# -------------------------

homedir=$(dirname $(realpath $0))
slurm_hist_dir=${homedir}/slurm_history
slurm_out_path=${slurm_hist_dir}/slurm-%x-%A_%a.out

echo homedir=$homedir
echo slurm_hist_dir=$slurm_hist_dir
mkdir -p $slurm_hist_dir

cd $homedir
cd $PROJECT_SLURM_DIR
echo run directory: $(pwd)

function submit_job {
    network=$1
    label=$2
    model=$3
    neg=$4

    [[ $neg == True ]] && neg_name=noneg || neg_name=useneg

    job_name="${network,,}-${label,,}-${model,,}-${neg_name}"
    echo $job_name

    script="${BASE_SCRIPT} network=${network} label=${label} model=${model}"
    script+=" gnn_params.use_negative=${neg} name_tag=${neg_name}"
    sbatch_extra_settings="--array=1-${NUM_REPS} -J ${job_name} -o ${slurm_out_path} -C NOAUTO:amd20"
    full_script="sbatch --gres=gpu:v100s:1 ${sbatch_extra_settings} ${TEMPLATE} ${script}"

    # echo $script && echo $full_script
    eval $full_script
}

for network in ${NETWORKS[@]}; do
    for label in ${LABELS[@]}; do
        for model in ${MODELS[@]}; do
            submit_job $network $label $model True
            submit_job $network $label $model False
        done
    done
done
