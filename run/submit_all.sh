#!/bin/bash --login

mkdir -p ../slurm_history

networks=(BioGRID HumanNet STRING)
labels=(DisGeNet GOBP)
gml_models=(ADJ-LogReg ADJ-SVM N2V-LogReg N2V-SVM, LabelProp)
gnn_models=(GCN GIN GAT GraphSAGE)

function submit_job {
    network=$1
    label=$2
    model=$3
    gpu=$4

    job_name="${network,,}-${label,,}-${model,,}"
    echo $job_name

    script="/bin/time -v python main.py --network_name ${network} --label_name ${label} --model_name ${model}"

    if (( gpu == 1 ))
    then
        sbatch -J $job_name --gres=gpu:v100:1 job_template.sb $script
    else
        sbatch -J $job_name -C amd20 job_template.sb $script
    fi
}

for network in ${networks[@]}; do
    for label in ${labels[@]}; do
        for model in ${gnn_models[@]}; do
            if [[ $network == STRING ]] && [[ $model == GAT ]]; then
                continue  # OOM
            fi

            submit_job $network $label $model 1
        done

        for model in ${gml_models[@]}; do
            submit_job $network $label $model 0
        done
    done
done
