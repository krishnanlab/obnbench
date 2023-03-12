import json
import os

import hydra
import nleval
import numpy as np
import torch.nn as nn
from nleval import Dataset
from nleval.ext.pecanpy import pecanpy_embed
from nleval.ext.grape import grape_embed
from nleval.ext.sknetwork import sknetwork_embed
from nleval.feature import FeatureVec
from nleval.metric import auroc, log2_auprc_prior
from nleval.model.label_propagation import RandomWalkRestart
from nleval.model_trainer.gnn import SimpleGNNTrainer
from nleval.model_trainer import SupervisedLearningTrainer, LabelPropagationTrainer
from nleval.util.logger import display_pbar
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import LinearSVC
from torch_geometric import nn as pygnn
from typing import Dict, List, Optional, Tuple, Union

from get_data import load_data
from utils import get_device, get_num_workers, normalize_path

GNN_METHODS = ["GCN", "GAT", "GIN", "GraphSAGE"]
GML_METHODS = [
    "ADJ-LogReg",
    "ADJ-SVM",
    "N2V-LogReg",
    "N2V-SVM",
    "LINE-LogReg",
    "HOPE-LogReg",
    "LapEig-LogReg",
    "Walklets-LogReg",
    "SVD-LogReg",
    "RandNE-LogReg",
    "LouvainNE-LogReg",
]
ALL_METHODS = GNN_METHODS + GML_METHODS + ["LabelProp"]
METRICS = {"log2pr": log2_auprc_prior, "auroc": auroc}


def parse_params(cfg: DictConfig) -> Tuple[Dict[str, int], Dict[str, int], str, str]:
    """Parse model specific parameters."""
    mdl_name = cfg.model
    optim_params = cfg.optim_params.get(mdl_name)

    # Parse model and train parameters (hp params is for setting up  paths)
    if mdl_name in GNN_METHODS:
        params = cfg.gnn_params
        parser = _parse_gnn_params
    elif mdl_name in GML_METHODS:
        params = cfg.n2v_params
        parser = _parse_n2v_params
    elif mdl_name == "LabelProp":
        params = cfg.lp_params
        parser = _parse_lp_params
    else:
        raise ValueError(f"Unrecognized model option {mdl_name!r}")

    # Overwrite parameters using optimally tuned params for the model
    if not cfg.hp_tune and not mdl_name.startswith("ADJ"):
        nleval.logger.info(f"{cfg.hp_tune=}, overwriting model parameters using optimal params")
        nleval.logger.info(f"  Before: {params=}")
        params.update(optim_params or {})
        nleval.logger.info(f"  After: {params=}")

    hp_opts, mdl_opts, trainer_opts = parser(params)
    result_path, log_path = _get_paths(cfg, hp_opts)

    return mdl_opts, trainer_opts, result_path, log_path


def _parse_gnn_params(gnn_params: DictConfig):
    hp_opts = {
        "hidden_channels": gnn_params.hid_dim,
        "num_layers": gnn_params.num_layers,
        "lr": gnn_params.lr,
    }  # hyper-parameters to be tuned
    gnn_opts = {
        "hidden_channels": gnn_params.hid_dim,
        "num_layers": gnn_params.num_layers,
    }
    trainer_opts = {
        "lr": gnn_params.lr,
        "epochs": gnn_params.epochs,
        "eval_steps": gnn_params.eval_steps,
    }
    return hp_opts, gnn_opts, trainer_opts


def _parse_n2v_params(n2v_params: DictConfig):
    hp_opts = {
        "dim": n2v_params.hid_dim,
        "window_size": n2v_params.window_size,
        "walk_length": n2v_params.walk_length,
    }
    n2v_opts = {
        "dim": n2v_params.hid_dim,
        "window_size": n2v_params.window_size,
        "walk_length": n2v_params.walk_length,
        "num_walks": n2v_params.num_walks,
    }
    return hp_opts, n2v_opts, None


def _parse_lp_params(lp_params: DictConfig):
    hp_opts = lp_opts = {"beta": lp_params.beta}
    return hp_opts, lp_opts, None


def _get_paths(cfg: DictConfig, opt: Optional[Dict[str, float]] = None) -> Tuple[str, str]:
    # Get output file name and path
    out_dir = os.path.join(cfg.homedir, cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Only results are saved directly under the out_dir as json, no logs
    # <out_dir>/{network}_{label}_{model}_{runid}.json
    if not cfg.hp_tune:
        log_path = None
        exp_name = "_".join([i.lower() for i in (cfg.network, cfg.label, cfg.model)])
        result_path = os.path.join(cfg.homedir, out_dir, f"{exp_name}_{cfg.runid}.json")

    # Nested dir struct for organizing different hyperparameter tuning exps
    # <out_dir>/{method}/{settings}/{dataset}/{runid}
    else:
        dataset = "_".join(i.lower() for i in (cfg.network, cfg.label))
        settings = "_".join(f"{i.replace('_', '-')}={j}" for i, j in opt.items()) if opt else "none"
        out_path = os.path.join(cfg.homedir, out_dir, cfg.model, settings, dataset, str(cfg.runid))
        os.makedirs(out_path, exist_ok=True)

        result_path = os.path.join(out_path, "score.json")
        log_path = os.path.join(out_path, "run.log")

    nleval.logger.info(f"Results will be saved to {result_path}")

    return result_path, log_path


class GNN(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        conv_name: str,
        conv_kwargs,
        *,
        num_pre_mp_layers: int = 1,
        num_post_mp_layers: int = 1,
    ):
        super().__init__()
        dim_hid = conv_kwargs["hidden_channels"]

        self.pre_mp = pygnn.MLP(in_channels=dim_in, hidden_channels=dim_hid, out_channels=dim_hid,
                                num_layers=num_pre_mp_layers)
        self.conv = getattr(pygnn, conv_name)(in_channels=dim_hid, out_channels=dim_hid, jk="last", dropout=0.5,
                                              norm=pygnn.DiffGroupNorm(dim_hid, 5), **conv_kwargs)
        self.post_mp = pygnn.MLP(in_channels=dim_hid, hidden_channels=dim_hid, out_channels=dim_out,
                                 num_layers=num_post_mp_layers)
        nleval.logger.info(f"Model constructed:\n{self}")

    def forward(self, x, adj):
        for m in self.children():
            x = m(x, adj) if isinstance(m, pygnn.models.basic_gnn.BasicGNN) else m(x)
        return x


def set_up_mdl(cfg: DictConfig, g, lsc, log_level="INFO"):
    """Set up model, trainer, graph, and features."""
    mdl_name = cfg.model
    feat_dim = cfg.gnn_params.node_feat_dim

    feat = None
    dense_g = g.to_dense_graph()
    mdl_opts, trainer_opts, result_path, log_path = parse_params(cfg)

    if mdl_name in GNN_METHODS:
        num_tasks = lsc.size
        if mdl_name == "GraphSAGE":
            mdl_opts.update({"aggr": "add", "normalize": True})

        one_hot_disc_deg = KBinsDiscretizer(
            n_bins=feat_dim,
            encode="onehot-dense",
            strategy="uniform",
        ).fit_transform(np.log(dense_g.mat.sum(axis=1, keepdims=True)))
        nleval.logger.info(f"Bins stats: {one_hot_disc_deg.sum(0)}")
        feat = nleval.feature.FeatureVec.from_mat(one_hot_disc_deg, g.idmap)

        mdl = GNN(dim_in=feat_dim, dim_out=num_tasks, conv_name=mdl_name, conv_kwargs=mdl_opts)

        mdl_trainer = SimpleGNNTrainer(METRICS, metric_best=cfg.metric_best,
                                       device=cfg.device, log_level=log_level,
                                       log_path=log_path, **trainer_opts)

    elif mdl_name in GML_METHODS:
        # Get network features
        if mdl_name.startswith("ADJ"):
            feat = FeatureVec.from_mat(dense_g.mat, g.idmap)
        elif mdl_name.startswith("N2V"):
            feat = pecanpy_embed(g, mode="PreCompFirstOrder", workers=cfg.num_workers,
                                 verbose=display_pbar(log_level), **mdl_opts)
        elif mdl_name.startswith("LINE"):
            feat = grape_embed(g, "FirstOrderLINEEnsmallen", dim=128)
        elif mdl_name.startswith("HOPE"):
            feat = grape_embed(g, "HOPEEnsmallen", dim=128)
        elif mdl_name.startswith("LapEig"):
            feat = grape_embed(g, "LaplacianEigenmapsEnsmallen", dim=128)
        elif mdl_name.startswith("Walklets"):
            feat = grape_embed(g, "WalkletsSkipGramEnsmallen", dim=128)
        elif mdl_name.startswith("SVD"):
            feat = sknetwork_embed(g, "SVD", dim=128)
        elif mdl_name.startswith("RandNE"):
            feat = sknetwork_embed(g, "RandomProjection", dim=128)
        elif mdl_name.startswith("LouvainNE"):
            feat = sknetwork_embed(g, "LouvainNE", dim=128)
        else:
            raise ValueError(f"Unrecognized model {mdl_name!r}")

        # Initialize model
        if mdl_name.endswith("LogReg"):
            mdl = LogisticRegression(penalty="l2", solver="lbfgs", n_jobs=1, max_iter=2000)
        elif mdl_name.endswith("SVM"):
            mdl = LinearSVC(penalty="l2", max_iter=2000)
        else:
            raise ValueError(f"Unrecognized model option {mdl_name!r}")
        mdl_trainer = SupervisedLearningTrainer(METRICS, log_level=log_level)

    elif mdl_name == "LabelProp":
        g = dense_g
        mdl = RandomWalkRestart(max_iter=20, warn=False, **mdl_opts)
        mdl_trainer = LabelPropagationTrainer(METRICS, log_level=log_level)

    else:
        raise ValueError(f"Unrecognized model option {mdl_name!r}")

    return mdl, mdl_trainer, g, feat, result_path


def results_to_json(
    label_ids: List[str],
    results: Dict[str, List[float]],
) -> List[Dict[str, Union[float, str]]]:
    """Convert results into JSON format."""
    results_json = []
    for i, label_id in enumerate(label_ids):
        new_item = {"task_name": label_id}
        for name in results:
            new_item[name] = results[name][i]
        results_json.append(new_item)
    return results_json


def get_gnn_results(mdl, dataset, device) -> Dict[str, List[float]]:
    """Generate final results for the GNN model."""
    results = {}
    data = dataset.to_pyg_data(device=device)
    for metric_name, metric_func in METRICS.items():
        mdl.eval()
        y_pred = mdl(data.x, data.edge_index).detach().cpu().numpy()
        y_true = dataset.y
        for mask_name in dataset.masks:
            mask = dataset.masks[mask_name][:, 0]
            scores = metric_func(y_true[mask], y_pred[mask], reduce="none")
            results[f"{mask_name}_{metric_name}"] = scores
    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    nleval.logger.info(f"Running with settings:\n{OmegaConf.to_yaml(cfg)}")

    cfg.homedir = normalize_path(cfg.homedir)
    cfg.device = get_device(cfg.device)
    cfg.num_workers = get_num_workers(cfg.num_workers)

    # Load data
    g, lsc, splitter = load_data(cfg.homedir, cfg.network, cfg.label, cfg.log_level)
    mdl, mdl_trainer, g, feat, result_path = set_up_mdl(cfg, g, lsc, cfg.log_level)
    dataset = Dataset(graph=g, feature=feat, label=lsc, splitter=splitter)
    nleval.logger.info(lsc.stats())

    # Train and evaluat model
    if cfg.model in GNN_METHODS:
        mdl_trainer.train(mdl, dataset)
        results = get_gnn_results(mdl, dataset, cfg.device)
    else:
        results = mdl_trainer.eval_multi_ovr(mdl, dataset)

    # Save results as JSON
    results_json = results_to_json(lsc.label_ids, results)
    with open(result_path, "w") as f:
        json.dump(results_json, f, indent=4)


if __name__ == "__main__":
    main()
