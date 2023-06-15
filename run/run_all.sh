NETWORKS=(
    # Medium
    BioGRID
    HumanNet
    ComPPIHumanInt
    # Small
    BioPlex
    FunCoup
    HuRI
    OmniPath
    PCNet
    ProteomeHD
    SIGNOR
)
FEATURES=(
    Constant
    RandomNormal
    OneHotLogDeg
    RandomWalkDiag
    Adj
    SVD
    LapEigMap
    RandProjGaussian
    RandProjSparse
    LINE1
    LINE2
    Node2vec
    Walklets
    Embedding
    AdjEmbBag
)
GNNS=(
    GCN
    GAT
    GIN
    SAGE
    GatedGCN
)
LOGREGS=(
    LogReg+Adj
    LogReg+SVD
    LogReg+LINE1
    LogReg+LINE2
    LogReg+Node2vec
    LogReg+Walklets
    LOgReg+LapEigMap
)

# Main experiments
sh run_main.sh

# Tuned GNN experiments
for network in BioGRID HumanNet; do
    for label in DisGeNET GOBP; do
        python main.py dataset.network=${network} dataset.label=${label} +model_tuned=${network}-${label}-GCN+tuned
        python main.py dataset.network=${network} dataset.label=${label} +model_tuned=${network}-${label}-GAT+tuned
    done
done

# Ablation studies
for network in ${NETWORKS[@]}; do
    for gnn in ${GNNS[@]}; do
        PARALLEL=0 sh run_abl_cs.sh ${gnn} ${network}
        PARALLEL=0 sh run_abl_gnn_label.sh ${gnn} ${network}
        PARALLEL=0 sh run_abl_gnn_cs_label.sh ${gnn} ${network}
        PARALLEL=0 sh run_abl_gnn_bot.sh ${gnn} ${network}

        for feature in ${FEATURES[@]}; do
            PARALLEL=0 sh run_abl_gnn_feature.sh ${gnn} ${network} ${feature}
        done
    done

    for logreg in ${LOGREGS[@]}; do
        PARALLEL=0 sh run_abl_cs.sh ${logreg} ${network} ${feature}
    done
done
