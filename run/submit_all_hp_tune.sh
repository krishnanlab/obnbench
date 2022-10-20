#!/bin/bash --login

mkdir -p ../slurm_history

networks=(BioGRID HumanNet STRING)
labels=(DisGeNet GOBP)

n2v_models=(N2V-LogReg N2V-SVM)
lp_models=(LabelProp)
gnn_models=(GCN GIN GAT GraphSAGE)

gnn_lr_opts=(0.001 0.005 0.01 0.05 0.1)
gnn_hid_dim_opts=(16 32 64 128)
gnn_num_layers_opts=(3 4 5)

n2v_hid_dim_opts=(64 128 256 512)
n2v_walk_length_opts=(20 40 80 120 160)
n2v_window_size_opts=(2 4 8 12 16)

lp_beta_opts=(0.1 0.3 0.5 0.7 0.8 0.85 0.9 0.95)

function submit_job {
    network=$1
    label=$2
    model=$3

    name="${network,,}-${label,,}-${model,,}"

    script="/bin/time -v python main.py network=${network} label=${label} model=${model}"
    script+=" out_dir=hp_tune_results hp_tune=true gnn_params.epochs=5000 gnn_params.eval_steps=100"

    if [[ ${n2v_models[*]} =~ $model ]]; then
        hp_tune_n2v $script
    elif [[ ${lp_models[*]} =~ $model ]]; then
        hp_tune_lp $script
    elif [[ ${gnn_models[*]} =~ $model ]]; then
        hp_tune_gnn $script
    else
        echo ERROR: unknown model ${model}
        exit 1
    fi
}

function hp_tune_gnn {
    for lr in ${gnn_lr_opts[@]}; do
        for hid_dim in ${gnn_hid_dim_opts[@]}; do
            for num_layers in ${gnn_num_layers_opts[@]}; do
                jobscript="sbatch -J hp_tune-${name} --gres=gpu:v100:1 job_template_single.sb "
                jobscript+="${script} gnn_params.num_layers=${num_layers} gnn_params.lr=${lr} "
                jobscript+="gnn_params.hid_dim=${hid_dim}"
                echo ${jobscript}
                eval ${jobscript}
            done
        done
    done
}

function hp_tune_n2v {
    for hid_dim in ${n2v_hid_dim_opts[@]}; do
        for walk_length in ${n2v_walk_length_opts[@]}; do
            for window_size in ${n2v_walk_length_opts[@]}; do
                jobscript="sbatch -J hp_tune-${name} -C amd20 job_template_single.sb "
                jobscript+="${script} n2v_params.hid_dim=${hid_dim} "
                jobscript+="n2v_params.walk_length=${walk_length} n2v_params.window_size=${window_size}"
                echo ${jobscript}
                eval ${jobscript}
            done
        done
    done
}

function hp_tune_lp {
    for beta in ${lp_beta_opts[@]}; do
        jobscript="sbatch -J hp_tune-${name} -C amd20 job_template_single.sb ${script} lp_params.beta=${beta}"
        echo ${jobscript}
        eval ${jobscript}
    done
}

for network in ${networks[@]}; do
    for label in ${labels[@]}; do
        for model in ${gnn_models[@]}; do
            if [[ $network == STRING ]] && [[ $model == GAT ]]; then
                continue  # OOM
            fi

            submit_job $network $label $model
        done

        for model in ${n2v_models[@]}; do
            submit_job $network $label $model
        done

        for model in ${lp_models[@]}; do
            submit_job $network $label $model
        done
    done
done