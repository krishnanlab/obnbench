from pathlib import Path

import numba
import torch
from nleval.metric import auroc, log2_auprc_prior

################
# Basic settings
################
HOME_DIR = Path(__file__).resolve().parent
DATA_DIR = HOME_DIR / "datasets"
GENE_LIST_PATH = HOME_DIR / "genes.txt"
DATA_VERSION = "nledata-v0.1.0-dev2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = numba.config.NUMBA_DEFAULT_NUM_THREADS

####################
# Experiment options
####################
NETWORKS = ["STRING", "BioGRID", "HumanNet"]
LABELS = ["DisGeNet", "GOBP"]
GNN_METHODS = ["GCN", "GAT", "GIN", "GraphSAGE"]
GML_METHODS = ["ADJ-LogReg", "ADJ-SVM", "N2V-LogReg", "N2V-SVM"]
ALL_METHODS = GNN_METHODS + GML_METHODS + ["LabelProp"]

####################
# Evaluation metrics
####################
METRICS = {"log2pr": log2_auprc_prior, "auroc": auroc}
METRIC_BEST = "log2pr"
