#!/bin/bash --login

mkdir -p ../slurm_history

networks=(BioGRID HumanNet STRING)
labels=(DisGeNet GOBP)
gml_models=(ADJ-LogReg ADJ-SVM N2V-LogReg N2V-SVM LabelProp)
gnn_models=(GCN GIN GAT GraphSAGE)

function submit_job {
    network=$1
    label=$2
    model=$3
    gpu=$4

    job_name="${network,,}-${label,,}-${model,,}"
    echo $job_name

    script="/bin/time -v python main.py network=${network} label=${label} model=${model}"

    if (( gpu == 1 )); then
        sbatch -J $job_name --gres=gpu:v100:1 job_template.sb $script
    elif (( gpu == 2 )); then
        sbatch -J $job_name --gres=gpu:v100:1 -t 12:00:00 job_template.sb $script
    else
        sbatch -J $job_name -C NOAUTO:amd20 job_template.sb $script
    fi
}

for network in ${networks[@]}; do
    for label in ${labels[@]}; do
        for model in ${gnn_models[@]}; do
            if [[ $network == STRING ]] && [[ $model == GAT ]]; then
                continue  # OOM on V100
            elif [[ $network == STRING ]] && [[ $model == GCN ]]; then
                submit_job $network $label $model 1  # 2
            else
                submit_job $network $label $model 1
            fi
        done

        for model in ${gml_models[@]}; do
            submit_job $network $label $model 0
        done
    done
done
