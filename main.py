import argparse
import json
import os

import nleval
from nleval import Dataset
from nleval.feature import FeatureVec
from nleval.model.label_propagation import RandomWalkRestart
from nleval.model_trainer.gnn import SimpleGNNTrainer
from nleval.model_trainer import SupervisedLearningTrainer, LabelPropagationTrainer
from nleval.util.logger import display_pbar
from pecanpy.pecanpy import PreCompFirstOrder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torch_geometric import nn as pygnn
from typing import Dict, List, Union

import config
from get_data import load_data


def set_up_mdl(mdl_name: str, g, lsc, log_level="INFO"):
    """Set up model, trainer, graph, and features."""
    feat = None
    dense_g = g.to_dense_graph()

    if mdl_name in config.GNN_METHODS:
        num_tasks = lsc.size
        opts = {"aggr": "add", "normalize": True} if mdl_name == "GraphSAGE" else {}
        mdl = getattr(pygnn, mdl_name)(in_channels=1, hidden_channels=config.HID_DIM,
                                       num_layers=config.NUM_LAYERS,
                                       out_channels=num_tasks, **opts)
        mdl_trainer = SimpleGNNTrainer(config.METRICS, metric_best=config.METRIC_BEST,
                                       device=config.DEVICE, epochs=config.EPOCHS,
                                       eval_steps=config.EVAL_STEPS, log_level=log_level)

    elif mdl_name in config.GML_METHODS:
        feat = dense_g.mat

        # Node2vec embedding
        if mdl_name.startswith("N2V"):
            pecanpy_verbose = display_pbar(log_level)
            pecanpy_g = PreCompFirstOrder.from_mat(dense_g.mat, g.node_ids,
                                                   workers=config.NUM_WORKERS,
                                                   verbose=pecanpy_verbose)
            feat = pecanpy_g.embed(dim=config.N2V_DIM, num_walks=config.N2V_NUM_WALKS,
                                   walk_length=config.N2V_WALK_LENGTH,
                                   window_size=config.N2V_WINDOW_SIZE,
                                   verbose=pecanpy_verbose)
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
        g = dense_g
        mdl = RandomWalkRestart(max_iter=20, warn=False)
        mdl_trainer = LabelPropagationTrainer(config.METRICS, log_level=log_level)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_name", required=True, choices=config.NETWORKS)
    parser.add_argument("--label_name", required=True, choices=config.LABELS)
    parser.add_argument("--model_name", required=True, choices=config.ALL_METHODS)
    parser.add_argument("--rep", type=int, default=0, help="Repetition number.")
    parser.add_argument("--out_dir", default="results", help="Output directory.")

    args = parser.parse_args()
    nleval.logger.info(args)

    return args


def main():
    # Get optsion
    args = parse_args()
    network_name = args.network_name
    label_name = args.label_name
    model_name = args.model_name
    rep = args.rep
    out_dir = args.out_dir

    # Get output file name and path
    os.makedirs(out_dir, exist_ok=True)
    exp_name = "_".join([i.lower() for i in (network_name, label_name, model_name)])
    out_path = os.path.join(out_dir, f"{exp_name}_{rep}.json")
    nleval.logger.info(f"Results will be saved to {out_path}")

    # Load data
    g, lsc, splitter = load_data(network_name, label_name)
    mdl, mdl_trainer, g, feat = set_up_mdl(model_name, g, lsc)
    dataset = Dataset(graph=g, feature=feat, label=lsc, splitter=splitter)
    nleval.logger.info(lsc.stats())

    # Train and evaluat model
    if model_name in config.GNN_METHODS:
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
