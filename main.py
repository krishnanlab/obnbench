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
from typing import Dict, List, Literal, Optional, Tuple, Union

import config
from get_data import load_data


def parse_params(
    name: Literal["gnn", "n2v", "lp"],
    /,
    *,
    params: DictConfig,
) -> Tuple[Dict[str, int], Dict[str, int], Optional[Dict[str, int]]]:
    """Parse model specific hyper parameters.

    Args:
        name: Name of the model class (gnn, n2v, lp).
        params: Parameters specific to the model class to be used.

    Returns:
        Hyper parameters dictionary used for constructing result path; model
            specific parameters; trainer specific parameters (optional).

    """
    if name == "gnn":
        parser = _parse_gnn_params
    elif name == "n2v":
        parser = _parse_n2v_params
    elif name == "lp":
        parser = _parse_lp_params
    else:
        raise ValueError(f"Unknown model class {name!r}")

    return parser(params)


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
    """<out_dir>/{method}/{settings}/{dataset}/{runid}"""
    # Get output file name and path
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Only results are saved directly under the out_dir as json
    if not cfg.hp_tune:
        log_path = None
        exp_name = "_".join([i.lower() for i in (cfg.network, cfg.label, cfg.model)])
        result_path = os.path.join(cfg.out_dir, f"{exp_name}_{cfg.runid}.json")

    # Nested dir struct for organizing different hyperparameter tuning exps
    else:
        dataset = "_".join(i.lower() for i in (cfg.network, cfg.label))
        settings = "_".join(f"{i.replace('_', '-')}={j}" for i, j in opt.items()) if opt else "none"
        out_path = os.path.join(cfg.out_dir, cfg.model, settings, dataset, str(cfg.runid))
        os.makedirs(out_path, exist_ok=True)

        result_path = os.path.join(out_path, "score.json")
        log_path = os.path.join(out_path, "run.log")

    nleval.logger.info(f"Results will be saved to {result_path}")

    return result_path, log_path


def set_up_mdl(cfg: DictConfig, g, lsc, log_level="INFO"):
    """Set up model, trainer, graph, and features."""
    mdl_name = cfg.model

    feat = None
    dense_g = g.to_dense_graph()

    if mdl_name in config.GNN_METHODS:
        num_tasks = lsc.size
        hp_opts, gnn_opts, trainer_opts = parse_params("gnn", cfg.gnn_params)
        result_path, log_path = _get_paths(cfg, hp_opts)
        if mdl_name == "GraphSAGE":
            gnn_opts.update({"aggr": "add", "normalize": True})

        mdl = getattr(pygnn, mdl_name)(in_channels=1, out_channels=num_tasks, **gnn_opts)
        mdl_trainer = SimpleGNNTrainer(config.METRICS, metric_best=config.METRIC_BEST,
                                       device=config.DEVICE, log_level=log_level,
                                       log_path=log_path, **trainer_opts)

    elif mdl_name in config.GML_METHODS:
        feat = dense_g.mat
        hp_opts, n2v_opts, _ = parse_params("n2v", cfg.n2v_params)
        # Will return n2v hp specific paths only when cfg.hp_tune=True
        result_path = _get_paths(cfg, hp_opts)[0]

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
        mdl_trainer = SupervisedLearningTrainer(config.METRICS, log_level=log_level)

    elif mdl_name == "LabelProp":
        hp_opts, lp_opts, _ = parse_params("lp", cfg.lp_params)
        result_path = _get_paths(cfg, hp_opts)[0]
        g = dense_g
        mdl = RandomWalkRestart(max_iter=20, warn=False, **lp_opts)
        mdl_trainer = LabelPropagationTrainer(config.METRICS, log_level=log_level)

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

    # Load data
    g, lsc, splitter = load_data(cfg.network, cfg.label, cfg.log_level)
    mdl, mdl_trainer, g, feat, result_path = set_up_mdl(cfg, g, lsc, cfg.log_level)
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
    with open(result_path, "w") as f:
        json.dump(results_json, f, indent=4)


if __name__ == "__main__":
    main()
