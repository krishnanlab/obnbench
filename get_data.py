from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import nleval
import pandas as pd
from nleval import data, label
from nleval.util.converter import GenePropertyConverter
from omegaconf import DictConfig, OmegaConf

from utils import get_data_dir, get_gene_list_path, normalize_path


def get_network_construct(network_name):
    # Try to extract channel name {network}-{channel}
    if len(terms := network_name.split("-")) == 1:
        name = network_name
        kwargs = {}
    else:
        name, channel = terms
        kwargs = {"channel": channel}

    gcls = getattr(data, name)

    return gcls, kwargs


def load_data(
    homedir: Path,
    network_name: str,
    label_name: str,
    log_level: str = "INFO",
):
    datadir = get_data_dir(homedir)

    gcls, kwargs = get_network_construct(network_name)
    g = gcls(datadir, **kwargs)

    splitter, filter_ = get_splitter_filter(homedir)
    lsc = getattr(data, label_name)(datadir, transform=filter_)

    return g, lsc, splitter


def print_label_stats(lsc, splitter, common_genes):
    y, masks = lsc.split(target_ids=tuple(common_genes), splitter=splitter)
    num_nodes, num_classes = y.shape

    effective_class_dict: Dict[Tuple[int, ...], int] = {}
    y_effective: List[int] = [0] * y.shape[0]
    current = 0
    for i, j in enumerate(y.astype(int)):
        if j.sum() > 0:
            if (z := tuple(j.tolist())) not in effective_class_dict:
                effective_class_dict[z] = (current := current + 1)

            y_effective[i] = effective_class_dict[z]

    df = pd.DataFrame(y_effective)
    num_effective_classes = df[0].unique().size - 1
    nleval.logger.info(f"Total number of classes: {num_classes}")
    nleval.logger.info(f"Total number of effective classes: {num_effective_classes:,}")

    stats_lst = []
    mask_names = ["train", "val", "test"]
    for mask_name in mask_names:
        mask = masks[mask_name][:, 0]
        num_pos_per_class = y[mask].sum(axis=0)
        num_pos_per_eff_class = (df.iloc[np.where(mask)[0], 0]
                                 .value_counts().values.tolist())
        num_pos_per_eff_class = (num_pos_per_eff_class
                                 + [0] * (num_effective_classes
                                          - len(num_pos_per_eff_class)))  # pad with zeros
        stats_lst.append(
            (
                (y[mask].sum(axis=1) > 0).sum() / num_nodes,  # label rate
                num_pos_per_class.mean(),  # avg number of examples
                num_pos_per_class.std(),
                np.mean(num_pos_per_eff_class),  # effective avg number of examples
                np.std(num_pos_per_eff_class),
            )
        )
    stats_df = pd.DataFrame(stats_lst).rename(
        columns={
            0: "Label rate",
            1: "Number of examples per class (avg)",
            2: "Number of examples per class (std)",
            3: "Effective number of examples per class (avg)",
            4: "Effective number of examples per class (std)",
        },
        index={i: j for i, j in enumerate(mask_names)},
    )
    nleval.logger.info(f"\n{stats_df.to_markdown(index=False)}")


def get_splitter_filter(homedir: Path, log_level: str = "INFO"):
    datadir = get_data_dir(homedir)
    gene_list_path = get_gene_list_path(homedir)

    pubmedcnt_converter = GenePropertyConverter(datadir, name="PubMedCount")
    splitter = label.split.RatioPartition(
        *(0.6, 0.2, 0.2),
        ascending=False,
        property_converter=pubmedcnt_converter,
    )

    common_genes = np.loadtxt(gene_list_path, dtype=str).tolist()
    filter_ = label.filters.Compose(
        label.filters.EntityExistenceFilter(common_genes, log_level=log_level),
        label.filters.LabelsetRangeFilterSize(min_val=50, log_level=log_level),
        label.filters.LabelsetRangeFilterSplit(min_val=10, splitter=splitter),
        log_level=log_level,
    )

    return splitter, filter_


@hydra.main(version_base=None, config_path="conf", config_name="data_config")
def main(cfg: DictConfig):
    nleval.logger.info(f"Running with settings:\n{OmegaConf.to_yaml(cfg)}")

    cfg.homedir = normalize_path(cfg.homedir)
    datadir = get_data_dir(cfg.homedir)
    gene_list_path = get_gene_list_path(cfg.homedir)

    common_genes = None
    for network_name in cfg.networks:
        gcls, kwargs = get_network_construct(network_name)
        g = gcls(datadir, version=cfg.data_version, **kwargs)
        print(
            f"{network_name:<15}# nodes = {g.num_nodes:,}, # edges = {g.num_edges:,}, "
            f"edge density = {g.num_edges / g.num_nodes / (g.num_nodes - 1):.4f}",
        )

        if common_genes is None:
            common_genes = set(g.node_ids)
        else:
            common_genes = common_genes.intersection(set(g.node_ids))

    nleval.logger.info(f"Exporting {len(common_genes):,} common genes {gene_list_path}")
    with open(gene_list_path, "w") as f:
        for i in sorted(common_genes):
            f.write(f"{i}\n")

    splitter, filter_ = get_splitter_filter(cfg.homedir)
    for label_name in cfg.labels:
        lsc = getattr(data, label_name)(datadir, transform=filter_,
                                        version=cfg.data_version)

        nleval.logger.info(f"Start obtaining stats for {label_name}")
        print_label_stats(lsc, splitter, common_genes)


if __name__ == "__main__":
    main()
