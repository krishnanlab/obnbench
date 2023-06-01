import json
import os
import warnings
from math import ceil

import hydra
import lightning.pytorch as pl
import nleval
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import wandb
from lightning.pytorch.loggers import WandbLogger
from nleval import Dataset
from nleval.dataset_pyg import OpenBiomedNetBench
from nleval.ext.grape import grape_embed
from nleval.ext.pecanpy import pecanpy_embed
from nleval.ext.sknetwork import sknetwork_embed
from nleval.feature import FeatureVec
from nleval.graph import DenseGraph
from nleval.metric import auroc, log2_auprc_prior
from nleval.model.label_propagation import RandomWalkRestart
from nleval.model_trainer import SupervisedLearningTrainer, LabelPropagationTrainer
from nleval.model_trainer.gnn import SimpleGNNTrainer
from nleval.util.logger import display_pbar
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.svm import LinearSVC
from torch_geometric import nn as pygnn
from typing import Dict, List, Optional, Tuple, Union

from get_data import load_data
from obnbench.data_module import DataModule
from obnbench.model import ModelModule
from obnbench.preprocess import precompute_features, infer_dimensions
from obnbench.utils import (
    get_device,
    get_num_workers,
    normalize_path,
    get_data_dir,
    get_gene_list_path,
)

GNN_METHODS = ["GCN", "GAT", "GIN", "GraphSAGE"]
GML_METHODS = [
    "ADJ-LogReg",
    "ADJ-SVM",
    "N2V-LogReg",
    "N2V-SVM",
    "LINE1-LogReg",
    "LINE2-LogReg",
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
        with open_dict(params):
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
    if "addopts" in gnn_params:
        gnn_opts.update(gnn_params.addopts)
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
    # If name_tag is specified, then
    # <out_dir>/{network}_{label}_{model}_{name_tag}_{runid}.json
    if not cfg.hp_tune:
        log_path = None
        exp_name = "_".join([i.lower() for i in (cfg.network, cfg.label, cfg.model)])
        if cfg.name_tag is not None:
            exp_name = f"{exp_name}_{cfg.name_tag}"
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


# class GNN(nn.Module):
# 
#     def __init__(
#         self,
#         dim_in: int,
#         dim_out: int,
#         conv_name: str,
#         conv_kwargs,
#         *,
#         num_pre_mp_layers: int = 1,
#         num_post_mp_layers: int = 1,
#     ):
#         super().__init__()
#         dim_hid = conv_kwargs["hidden_channels"]
# 
#         self.pre_mp = pygnn.MLP(
#             in_channels=dim_in,
#             hidden_channels=dim_hid,
#             out_channels=dim_hid,
#             num_layers=num_pre_mp_layers,
#         )
#         self.conv = getattr(pygnn, conv_name)(
#             in_channels=dim_hid,
#             out_channels=dim_hid,
#             jk="last",
#             dropout=0.5,
#             norm=pygnn.DiffGroupNorm(dim_hid, 5),
#             **conv_kwargs,
#         )
#         self.post_mp = pygnn.MLP(
#             in_channels=dim_hid,
#             hidden_channels=dim_hid,
#             out_channels=dim_out,
#             num_layers=num_post_mp_layers,
#         )
#         nleval.logger.info(f"Model constructed:\n{self}")
# 
#     def forward(self, x, adj):
#         for m in self.children():
#             x = m(x, adj) if isinstance(m, pygnn.models.basic_gnn.BasicGNN) else m(x)
#         return x


def construct_features(cfg: DictConfig, g: DenseGraph) -> nleval.feature.FeatureVec:
    feat_type = cfg.gnn_params.node_feat_type
    feat_dim = cfg.gnn_params.node_feat_dim

    if feat_type == "onehotlogdeg":
        log_deg = np.log(g.mat.sum(axis=1, keepdims=True))
        feat_ary = KBinsDiscretizer(
            n_bins=feat_dim,
            encode="onehot-dense",
            strategy="uniform",
        ).fit_transform(log_deg)
        nleval.logger.info(f"Bins stats: {feat_ary.sum(0)}")
    elif feat_type in "const":
        if feat_dim != 1:
            raise ValueError(f"Constant feature only allows dimension of 1, got {feat_dim!r}")
        feat_ary = np.ones((g.size, 1))
    elif feat_type in "logdeg":
        if feat_dim != 1:
            raise ValueError(f"Degree feature only allows dimension of 1, got {feat_dim!r}")
        feat_ary = np.log(g.mat.sum(axis=1, keepdims=True))
    elif feat_type == "random":
        feat_ary = np.random.default_rng(0).random((g.size, feat_dim))
    elif feat_type == "node2vec":
        feat_ary = pecanpy_embed(g, mode="DenseOTF", workers=cfg.num_workers,
                                 verbose=display_pbar(cfg.log_level), dim=feat_dim, as_array=True)
    elif feat_type == "lappe":
        L = sp.csr_matrix(np.diag(g.mat.sum(1)) - g.mat)
        assert (L != L.T).size == 0, "The input network must be undirected."
        eigvals, eigvecs = sp.linalg.eigsh(L, which="SM", k=feat_dim + 1)
        assert (eigvals[1:] > 1e-8).all(), f"Network appears to be disconnected.\n{eigvals=}"
        feat_ary = eigvecs[:, 1:] / np.sqrt((eigvecs[:, 1:] ** 2).sum(0))
    elif feat_type == "rwse":
        P = g.mat / g.mat.sum(0)
        feat_ary = np.zeros((g.size, feat_dim))
        vec = np.ones(g.size)
        for i in range(feat_dim):
            vec = P @ vec
            feat_ary[:, i] = vec
    elif feat_type == "randprojg":
        feat_ary = GaussianRandomProjection(n_components=feat_dim).fit_transform(g.mat)
    elif feat_type == "randprojs":
        feat_ary = SparseRandomProjection(n_components=feat_dim, dense_output=True).fit_transform(g.mat)
        nleval.logger.info(f"Zero rate: {(feat_ary == 0).sum() / feat_ary.size:.2%}")
        nleval.logger.info(f"All zero rate: {(feat_ary == 0).all(1).sum() / feat_ary.shape[0]:.2%}")
    else:
        raise ValueError(f"Unknown feature type {feat_type!r}")

    return nleval.feature.FeatureVec.from_mat(feat_ary, g.idmap)


def set_up_mdl(cfg: DictConfig, g, lsc, log_level="INFO"):
    """Set up model, trainer, graph, and features."""
    mdl_name = cfg.model
    num_tasks = lsc.size

    feat = None
    dense_g = g.to_dense_graph()
    mdl_opts, trainer_opts, result_path, log_path = parse_params(cfg)

    if mdl_name in GNN_METHODS:
        feat = construct_features(cfg, dense_g)
        mdl = GNN(dim_in=feat.dim, dim_out=num_tasks, conv_name=mdl_name, conv_kwargs=mdl_opts)
        mdl_trainer = SimpleGNNTrainer(METRICS, metric_best=cfg.metric_best,
                                       use_negative=cfg.gnn_params.use_negative,
                                       device=cfg.device, log_level=log_level,
                                       log_path=log_path, **trainer_opts)

    elif mdl_name in GML_METHODS:
        # Get network features
        if mdl_name.startswith("ADJ"):
            feat = FeatureVec.from_mat(dense_g.mat, g.idmap)
        elif mdl_name.startswith("N2V"):
            feat = pecanpy_embed(g, mode="PreCompFirstOrder", workers=cfg.num_workers,
                                 verbose=display_pbar(log_level), **mdl_opts)
        elif mdl_name.startswith("LINE1"):
            feat = grape_embed(g, "FirstOrderLINEEnsmallen", dim=128)
        elif mdl_name.startswith("LINE2"):
            feat = grape_embed(g, "SecondOrderLINEEnsmallen", dim=128)
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

    mdl.eval()
    y_pred = mdl(data.x, data.edge_index).detach().cpu().numpy()
    y_true = dataset.y
    y_mask = dataset.y_mask

    for metric_name, metric_func in METRICS.items():
        for mask_name in dataset.masks:
            mask = dataset.masks[mask_name][:, 0]
            scores = metric_func(y_true[mask], y_pred[mask], reduce="none",
                                 y_mask=y_mask[mask])
            results[f"{mask_name}_{metric_name}"] = scores

    return results


def setup_callbacks(cfg: DictConfig):
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval="epoch",
        log_momentum=True,
    )
    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=None,  # use default set by Trainer's default_root_dir
        monitor=f"val/{cfg.metric.best}",
        verbose=True,
        save_last=True,
        save_top_k=5,
        mode=cfg.metric.obj,
        every_n_epochs=cfg.trainer.eval_interval,
    )
    early_stopping = pl.callbacks.EarlyStopping(
        monitor=f"val/{cfg.metric.best}",
        verbose=False,
        patience=ceil(cfg.trainer.early_stopping_patience / cfg.trainer.eval_interval),
        mode=cfg.metric.obj,
        check_finite=True,
    )
    return [lr_monitor, ckpt, early_stopping]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    nleval.logger.info(f"Running with settings:\n{OmegaConf.to_yaml(cfg)}")
    cfg.homedir = normalize_path(cfg.homedir)
    cfg.device = get_device(cfg.device)
    cfg.num_workers = get_num_workers(cfg.num_workers)
    pl.seed_everything(cfg.seed)

    # Load data
    gene_list_path = get_gene_list_path(cfg.homedir)
    data_dir = get_data_dir(cfg.homedir)
    gene_list = np.loadtxt(gene_list_path, dtype=str).tolist()
    dataset = OpenBiomedNetBench(data_dir, cfg.network, cfg.label, selected_genes=gene_list)

    # Preprocessing
    g = getattr(nleval.data, cfg.network)(data_dir, log_level="WARNING")
    precompute_features(cfg, dataset, g)
    infer_dimensions(cfg, dataset)

    # g, lsc, splitter = load_data(cfg.homedir, cfg.network, cfg.label, cfg.log_level)
    # mdl, mdl_trainer, g, feat, result_path = set_up_mdl(cfg, g, lsc, cfg.log_level)
    # dataset = Dataset(graph=g, feature=feat, label=lsc, splitter=splitter)
    # nleval.logger.info(lsc.stats())

    # model = GNN(cfg)
    model = ModelModule(cfg)
    nleval.logger.info(f"Model constructed:\n{model}")

    data = DataModule(dataset, num_workers=cfg.num_workers, pin_memory=True)
    wandb_logger = WandbLogger(project=cfg.wandb.project, entity=cfg.wandb.entity)
    callbacks = setup_callbacks(cfg)
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=cfg.trainer.eval_interval,
        fast_dev_run=cfg.trainer.fast_dev_run,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=1,  # full-batch training
    )

    with warnings.catch_warnings():
        warnings.simplefilter("once")

        if not cfg.trainer.inference_only:
            trainer.fit(model, datamodule=data)

        trainer.validate(model, datamodule=data, verbose=True)

    # # Train and evaluat model
    # if cfg.model in GNN_METHODS:
    #     mdl_trainer.train(mdl, dataset)
    #     results = get_gnn_results(mdl, dataset, cfg.device)
    # else:
    #     results = mdl_trainer.eval_multi_ovr(mdl, dataset, consider_negative=True)

    # # Save results as JSON
    # results_json = results_to_json(lsc.label_ids, results)
    # with open(result_path, "w") as f:
    #     json.dump(results_json, f, indent=4)


if __name__ == "__main__":
    main()
