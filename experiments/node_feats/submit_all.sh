#!/bin/bash --login

# -------------------------
TEMPLATE=job_template.sb
PROJECT_SLURM_DIR=../../run
EXP_DIR=experiments/node_feats
NUM_REPS=1

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
NODEFEATS=(
    const
    logdeg
    onehotlogdeg
    random
    node2vec
    lappe
    rwse
    randprojg
    randprojs
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
    type=$4
    dim=$5

    job_name="${network,,}-${label,,}-${model,,}-${type,,}${dim}"
    echo $job_name

    script="${BASE_SCRIPT} network=${network} label=${label} model=${model}"
    script+=" gnn_params.node_feat_type=${type} gnn_params.node_feat_dim=${dim} name_tag=${type,,}${dim}"
    sbatch_extra_settings="--array=1-${NUM_REPS} -J ${job_name} -o ${slurm_out_path} -C NOAUTO:amd20"
    [[ $type == lappe ]] && sbatch_extra_settings+=" -c 4 -t 24:00:00"  # need more CPUs & time for computing LapPEs
    full_script="sbatch --gres=gpu:v100s:1 ${sbatch_extra_settings} ${TEMPLATE} ${script}"

    # echo $script && echo $full_script
    eval $full_script
}

for network in ${NETWORKS[@]}; do
    for label in ${LABELS[@]}; do
        for model in ${MODELS[@]}; do
            for nodefeat in ${NODEFEATS[@]}; do
                if [[ $nodefeat == const ]] || [[ $nodefeat == logdeg ]]; then
                    submit_job $network $label $model $nodefeat 1
                else
                    for i in 32 64 128; do
                        submit_job $network $label $model $nodefeat $i
                    done
                fi
            done
        done
    done
done
