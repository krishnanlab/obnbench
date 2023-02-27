#!/bin/bash --login

# -------------------------
TEMPLATE=job_template.sb
PROJECT_SLURM_DIR=../../run

NETWORKS=(HumanNet ConsensusPathDB)
LABELS=(
    DisGeNET
    DisGeNET_Curated
    DisGeNET_BEFREE
    DisGeNET_Animal
    DisGeNET_GWAS
    DISEASES
    DISEASES_ExperimentsFiltered
    DISEASES_KnowledgeFiltered
    DISEASES_TextminingFiltered
    HPO
)
GML_MODELS=(
    ADJ-LogReg
    # ADJ-SVM
    N2V-LogReg
    # N2V-SVM
    LabelProp
)
GNN_MODELS=(
    GCN
    # GIN
    # GAT
    GraphSAGE
)

BASE_SCRIPT="/bin/time -v python main.py +experiment=disease_gsc_channels"
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

    if (( gpu == 1 )); then
        full_script="sbatch -J ${job_name} -o ${slurm_out_path} --gres=gpu:v100:1 ${TEMPLATE} ${script}"
    else
        # full_script="sbatch -J ${job_name} -o ${slurm_out_path} -C NOAUTO:amd20 -t 24:00:00 ${TEMPLATE} ${script}"
        full_script="sbatch -J ${job_name} -o ${slurm_out_path} -C NOAUTO:amd20 ${TEMPLATE} ${script}"
    fi

    # echo $script && echo $full_script
    eval $full_script
}

for network in ${NETWORKS[@]}; do
    for label in ${LABELS[@]}; do
        for model in ${GNN_MODELS[@]}; do
            submit_job $network $label $model 1
        done

        for model in ${GML_MODELS[@]}; do
            submit_job $network $label $model 0
        done
    done
done
