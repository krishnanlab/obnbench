import numpy as np
import nleval
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

        if common_genes is None:
            common_genes = set(g.node_ids)
        else:
            common_genes = common_genes.intersection(set(g.node_ids))

    nleval.logger.info(f"Exporting {len(common_genes):,} common genes to "
                       f"{config.GENE_LIST_PATH}")
    with open(config.GENE_LIST_PATH, "w") as f:
        for i in sorted(common_genes):
            f.write(f"{i}\n")

    _, filter_ = get_splitter_filter()
    for label_name in config.LABELS:
        getattr(data, label_name)(config.DATA_DIR, version=config.DATA_VERSION,
                                  transform=filter_)


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
