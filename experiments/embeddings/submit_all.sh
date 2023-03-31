#!/bin/bash --login

# -------------------------
TEMPLATE=job_template.sb
PROJECT_SLURM_DIR=../../run
EXP_DIR=experiments/embeddings
NUM_REPS=5

NETWORKS=(
    BioGRID
    ConsensusPathDB
    HumanNet
    STRING
)
LABELS=(
    DisGeNET
    DISEASES
    GOBP
)
MODELS=(
    N2V-LogReg
    LINE1-LogReg
    LINE2-LogReg
    HOPE-LogReg
    LapEig-LogReg
    Walklets-LogReg
    SVD-LogReg
    RandNE-LogReg
    LouvainNE-LogReg
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
    gpu=$4

    job_name="${network,,}-${label,,}-${model,,}"
    echo $job_name

    script="${BASE_SCRIPT} network=${network} label=${label} model=${model}"
    sbatch_extra_settings="--array=1-${NUM_REPS} -J ${job_name} -o ${slurm_out_path} -C NOAUTO:amd20"

    if [[ $model == Walklets-LogReg ]] || [[ $model == HOPE-LogReg ]]; then
        full_script="sbatch -c 8 --mem=64GB ${sbatch_extra_settings} ${TEMPLATE} ${script}"
    else
        full_script="sbatch ${sbatch_extra_settings} ${TEMPLATE} ${script}"
    fi

    # echo $script && echo $full_script
    eval $full_script
}

for network in ${NETWORKS[@]}; do
    for label in ${LABELS[@]}; do
        for model in ${MODELS[@]}; do
            submit_job $network $label $model 0
        done
    done
done
