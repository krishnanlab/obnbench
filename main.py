import json
import os

import hydra
import nleval
from nleval import Dataset
from nleval.feature import FeatureVec
from nleval.model.label_propagation import RandomWalkRestart
from nleval.model_trainer.gnn import SimpleGNNTrainer
from nleval.model_trainer import SupervisedLearningTrainer, LabelPropagationTrainer
from nleval.util.logger import display_pbar
from omegaconf import DictConfig, OmegaConf
from pecanpy.pecanpy import PreCompFirstOrder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torch_geometric import nn as pygnn
from typing import Dict, List, Tuple, Union

import config
from get_data import load_data


def _parse_gnn_params(gnn_params: DictConfig) -> Tuple[Dict[str, int], Dict[str, int]]:
    gnn_opts = {
        "hidden_channels": gnn_params.hid_dim,
        "num_layers": gnn_params.num_layers,
    }
    trainer_opts = {
        "epochs": gnn_params.epochs,
        "eval_steps": gnn_params.eval_steps,
    }
    return gnn_opts, trainer_opts


def _parse_n2v_params(n2v_params: DictConfig) -> Dict[str, int]:
    n2v_opts = {
        "dim": n2v_params.hid_dim,
        "window_size": n2v_params.window_size,
        "walk_length": n2v_params.walk_length,
        "num_walks": n2v_params.num_walks,
    }
    return n2v_opts


def set_up_mdl(cfg: DictConfig, g, lsc, log_level="INFO", log_path=None):
    """Set up model, trainer, graph, and features."""
    mdl_name = cfg.model

    feat = None
    dense_g = g.to_dense_graph()

    if mdl_name in config.GNN_METHODS:
        num_tasks = lsc.size
        gnn_opts, trainer_opts = _parse_gnn_params(cfg.gnn_params)
        if mdl_name == "GraphSAGE":
            gnn_opts.update({"aggr": "add", "normalize": True})

        mdl = getattr(pygnn, mdl_name)(in_channels=1, out_channels=num_tasks, **gnn_opts)
        mdl_trainer = SimpleGNNTrainer(config.METRICS, metric_best=config.METRIC_BEST,
                                       device=config.DEVICE, log_level=log_level,
                                       log_path=log_path, **trainer_opts)

    elif mdl_name in config.GML_METHODS:
        feat = dense_g.mat
        n2v_opts = _parse_n2v_params(cfg.n2v_params)

        # Node2vec embedding
        if mdl_name.startswith("N2V"):
            pecanpy_verbose = display_pbar(log_level)
            pecanpy_g = PreCompFirstOrder.from_mat(dense_g.mat, g.node_ids,
                                                   workers=config.NUM_WORKERS,
                                                   verbose=pecanpy_verbose)
            feat = pecanpy_g.embed(verbose=pecanpy_verbose, **n2v_opts)
        feat = FeatureVec.from_mat(feat, g.idmap)

        # Initialize model
        if mdl_name.endswith("LogReg"):
            mdl = LogisticRegression(penalty="l2", solver="lbfgs", n_jobs=1, max_iter=2000)
        elif mdl_name.endswith("SVM"):
            mdl = LinearSVC(penalty="l2", max_iter=2000)
        else:
            raise ValueError(f"Unrecognized model option {mdl_name!r}")
        mdl_trainer = SupervisedLearningTrainer(config.METRICS, log_level=log_level,
                                                log_path=log_path)

    elif mdl_name == "LabelProp":
        g = dense_g
        mdl = RandomWalkRestart(max_iter=20, warn=False)
        mdl_trainer = LabelPropagationTrainer(config.METRICS, log_level=log_level,
                                              log_path=log_path)

    else:
        raise ValueError(f"Unrecognized model option {mdl_name!r}")

    return mdl, mdl_trainer, g, feat


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


def get_gnn_results(mdl, dataset) -> Dict[str, List[float]]:
    """Generate final results for the GNN model."""
    results = {}
    data = dataset.to_pyg_data(device=config.DEVICE)
    for metric_name, metric_func in config.METRICS.items():
        y_pred = mdl(data.x, data.edge_index).detach().cpu().numpy()
        y_true = dataset.y
        for mask_name in dataset.masks:
            mask = dataset.masks[mask_name][:, 0]
            scores = metric_func(y_true[mask], y_pred[mask], reduce="none")
            results[f"{mask_name}_{metric_name}"] = scores
    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    nleval.logger.info(f"Runing with settings:\n{OmegaConf.to_yaml(cfg)}")

    # Get output file name and path
    os.makedirs(cfg.out_dir, exist_ok=True)
    exp_name = "_".join([i.lower() for i in (cfg.network, cfg.label, cfg.model)])
    out_path = os.path.join(cfg.out_dir, f"{exp_name}_{cfg.runid}.json")
    nleval.logger.info(f"Results will be saved to {out_path}")

    # Get log file path for performing hyperparameter tuning
    log_path = None

    # Load data
    g, lsc, splitter = load_data(cfg.network, cfg.label)
    mdl, mdl_trainer, g, feat = set_up_mdl(cfg, g, lsc, log_path=log_path)
    dataset = Dataset(graph=g, feature=feat, label=lsc, splitter=splitter)
    nleval.logger.info(lsc.stats())

    # Train and evaluat model
    if cfg.model in config.GNN_METHODS:
        mdl_trainer.train(mdl, dataset)
        results = get_gnn_results(mdl, dataset)
    else:
        results = mdl_trainer.eval_multi_ovr(mdl, dataset)

    # Save results as JSON
    results_json = results_to_json(lsc.label_ids, results)
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=4)


if __name__ == "__main__":
    main()
