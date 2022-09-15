from typing import Dict, List, Tuple

import numpy as np
import nleval
import pandas as pd
from nleval import data, label
from nleval.util.converter import GenePropertyConverter

import config


def load_data(network_name: str, label_name: str, log_level: str = "INFO"):
    g = getattr(data, network_name)(config.DATA_DIR, version=config.DATA_VERSION)

    splitter, filter_ = get_splitter_filter()
    lsc = getattr(data, label_name)(config.DATA_DIR, version=config.DATA_VERSION,
                                    transform=filter_)

    return g, lsc, splitter


def setup_data():
    common_genes = None
    for network_name in config.NETWORKS:
        g = getattr(data, network_name)(config.DATA_DIR, version=config.DATA_VERSION)
        print(
            f"{network_name:<15}# nodes = {g.num_nodes:,}, # edges = {g.num_edges:,}, "
            f"edge density = {g.num_edges / g.num_nodes / (g.num_nodes - 1):.4f}",
        )

        if common_genes is None:
            common_genes = set(g.node_ids)
        else:
            common_genes = common_genes.intersection(set(g.node_ids))

    nleval.logger.info(f"Exporting {len(common_genes):,} common genes to "
                       f"{config.GENE_LIST_PATH}")
    with open(config.GENE_LIST_PATH, "w") as f:
        for i in sorted(common_genes):
            f.write(f"{i}\n")

    splitter, filter_ = get_splitter_filter()
    for label_name in config.LABELS:
        lsc = getattr(data, label_name)(config.DATA_DIR, transform=filter_,
                                        version=config.DATA_VERSION)

        nleval.logger.info(f"Start obtaining stats for {label_name}")
        print_label_stats(lsc, splitter, common_genes)


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
    nleval.logger.info(f"\n{stats_df}")


def get_splitter_filter(log_level: str = "INFO"):
    pubmedcnt_converter = GenePropertyConverter(config.DATA_DIR, name="PubMedCount")
    splitter = label.split.RatioPartition(
        *(0.6, 0.2, 0.2),
        ascending=False,
        property_converter=pubmedcnt_converter,
    )

    common_genes = np.loadtxt(config.GENE_LIST_PATH, dtype=str).tolist()
    filter_ = label.filters.Compose(
        label.filters.EntityExistenceFilter(common_genes, log_level=log_level),
        label.filters.LabelsetRangeFilterSize(min_val=50, log_level=log_level),
        label.filters.LabelsetRangeFilterSplit(min_val=10, splitter=splitter),
        log_level=log_level,
    )

    return splitter, filter_


if __name__ == "__main__":
    setup_data()
