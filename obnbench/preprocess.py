import time
from functools import wraps
from pprint import pformat

import nleval
import numpy as np
import scipy.sparse as sp
import torch
from nleval.ext.grape import grape_embed
from nleval.ext.orbital_features import orbital_feat_extract
from nleval.ext.pecanpy import pecanpy_embed
from nleval.graph import SparseGraph
from nleval.util.logger import display_pbar
from omegaconf import DictConfig, open_dict
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from torch_geometric.data import Dataset

from obnbench.model_layers import feature_encoders

precomp_func_register = {}


class PreCompFeatureWrapper:

    def __init__(self, name: str):
        self.name = name
        self.feat_name = f"rawfeat_{name}"
        self.fe_name = f"{name}FeatureEncoder"
        assert hasattr(feature_encoders, self.fe_name)

    def __call__(self, func):

        @wraps(func)
        def wrapped_func(dataset: Dataset, *args, **kwargs) -> Dataset:
            nleval.logger.info(f"Precomputing raw features for {self.fe_name}")
            feat = func(*args, dataset=dataset, **kwargs)
            if not isinstance(feat, torch.Tensor):
                feat = torch.from_numpy(feat.astype(np.float32))

            # Handle dataset attr
            dataset._data_list = None
            dataset._data[self.feat_name] = feat
            if dataset.slices is not None:
                dataset.slices[self.feat_name] = torch.LongTensor([0, feat.shape[1]])

            return dataset

        precomp_func_register[self.name] = wrapped_func

        return wrapped_func


@PreCompFeatureWrapper("OneHotLogDeg")
def get_onehot_logdeg(feat_dim: int, adj: np.ndarray, **kwargs) -> np.ndarray:
    log_deg = np.log(adj.sum(axis=1, keepdims=True))
    feat = KBinsDiscretizer(
        n_bins=feat_dim,
        encode="onehot-dense",
        strategy="uniform",
    ).fit_transform(log_deg)
    nleval.logger.info(f"Bins stats:\n{feat.sum(0)}")
    return feat


@PreCompFeatureWrapper("Constant")
def get_const(feat_dim: int, adj: np.ndarray, **kwargs) -> np.ndarray:
    if feat_dim != 1:
        raise ValueError(
            "Constant feature only allows dimension of 1, "
            f"got {feat_dim!r}",
        )
    feat = np.ones((adj.shape[0], 1))
    return feat


@PreCompFeatureWrapper("RandomNormal")
def get_random_normal(feat_dim: int, adj: np.ndarray, **kwargs) -> np.ndarray:
    feat = np.random.default_rng(0).random((adj.shape[0], feat_dim))
    return feat


@PreCompFeatureWrapper("Orbital")
def get_orbital_counts(
    g: SparseGraph,
    n_jobs: int = 1,
    log_level: str = "INFO",
    **kwargs,
) -> np.ndarray:
    feat = orbital_feat_extract(
        g,
        graphlet_size=4,
        n_jobs=n_jobs,
        as_array=True,
        verbose=display_pbar(log_level),
    )
    return feat


@PreCompFeatureWrapper("SVD")
def get_svd_emb(feat_dim: int, adj: np.ndarray, **kwargs) -> np.ndarray:
    A = sp.csr_matrix(adj)
    feat, _, _ = sp.linalg.svds(A, k=feat_dim, which="LM")
    return feat


@PreCompFeatureWrapper("LapEigMap")
def get_lap_eig_map(feat_dim: int, adj: np.ndarray, **kwargs) -> np.ndarray:
    L = sp.csr_matrix(np.diag(adj.sum(1)) - adj)
    assert (L != L.T).size == 0, "The input network must be undirected."
    eigvals, eigvecs = sp.linalg.eigsh(L, which="SM", k=feat_dim + 1)
    assert (eigvals[1:] > 1e-8).all(), f"Network appears to be disconnected.\n{eigvals=}"
    feat = eigvecs[:, 1:] / np.sqrt((eigvecs[:, 1:] ** 2).sum(0))
    return feat


@PreCompFeatureWrapper("RandomWalkDiag")
def get_rand_walk_diag(feat_dim: int, adj: np.ndarray, **kwargs) -> np.ndarray:
    P = adj / adj.sum(0)
    feat = np.zeros((adj.shape[0], feat_dim))
    vec = np.ones(adj.shape[0])
    for i in range(feat_dim):
        vec = P @ vec
        feat[:, i] = vec
    return feat


@PreCompFeatureWrapper("RandProjGaussian")
def get_rand_proj_gaussian(feat_dim: int, adj: np.ndarray, **kwargs) -> np.ndarray:
    grp = GaussianRandomProjection(n_components=feat_dim)
    feat = grp.fit_transform(adj)
    return feat


@PreCompFeatureWrapper("RandProjSparse")
def get_rand_proj_sparse(feat_dim: int, adj: np.ndarray, **kwargs) -> np.ndarray:
    srp = SparseRandomProjection(n_components=feat_dim, dense_output=True)
    feat = srp.fit_transform(adj)
    return feat


@PreCompFeatureWrapper("EmbLINE1")
def get_line1_emb(feat_dim: int, g: SparseGraph, **kwargs) -> np.ndarray:
    feat = grape_embed(g, "FirstOrderLINEEnsmallen", dim=feat_dim)
    return feat


@PreCompFeatureWrapper("EmbLINE2")
def get_line2_emb(feat_dim: int, g: SparseGraph, **kwargs) -> np.ndarray:
    feat = grape_embed(g, "SecondOrderLINEEnsmallen", dim=feat_dim)
    return feat


@PreCompFeatureWrapper("EmbNode2vec")
def get_n2v_emb(feat_dim: int, g: SparseGraph, n_jobs=1, **kwargs) -> np.ndarray:
    feat = pecanpy_embed(
        g,
        mode="PreCompFirstOrder",
        workers=n_jobs,
        verbose=True,
        dim=feat_dim,
        as_array=True,
    )
    return feat


@PreCompFeatureWrapper("EmbWalklets")
def get_walklets_emb(feat_dim: int, g: SparseGraph, **kwargs) -> np.ndarray:
    feat = grape_embed(
        g,
        "WalkletsSkipGramEnsmallen",
        dim=feat_dim,
        as_array=True,
        grape_enable=True,
    )
    return feat


@PreCompFeatureWrapper("LabelReuse")
def get_label_resuse(dataset: Dataset, **kwargs) -> torch.Tensor:
    feat = torch.zeros_like(dataset.data.y, dtype=torch.float)
    train_mask = dataset.data.train_mask[:, 0]
    feat[train_mask] = dataset.data.y[train_mask]
    feat /= feat.sum(0)  # normalize
    return feat


def precompute_features(cfg: DictConfig, dataset: Dataset, g: SparseGraph):
    # Catch invalid node encoders before executing
    invalid_fe = []
    for feat_name in (node_encoders := cfg.node_encoders.split("+")):
        if feat_name not in precomp_func_register:
            invalid_fe.append(feat_name)
    if invalid_fe:
        raise ValueError(f"Invalid node encoders {invalid_fe} in {cfg.node_encoders}")

    # Prepare shared data arguments
    data_dict = {"dataset": dataset, "g": g, "adj": g.to_dense_graph().mat}

    tic = time.perf_counter()
    nleval.logger.info("Start pre-computing features")
    for feat_name in node_encoders:
        fe_cfg = cfg.node_encoder_params.get(feat_name)
        feat_dim = fe_cfg.get("raw_dim", None)
        precomp_func_register[feat_name](
            feat_dim=feat_dim,
            log_level=cfg.log_level,
            **data_dict,
        )
    elapsed = time.perf_counter() - tic
    nleval.logger.info(f"Precomputation done! Took {elapsed:.2f} seconds.")


def infer_dimensions(cfg: DictConfig, dataset: Dataset):
    # NOTE: this function is called after precompute_features, so we don't need
    # to check the validity of the node_encoders setting again.
    node_encoders = cfg.node_encoders.split("+")

    # Infer number of tasks
    dim_out = dataset.data.y.shape[1]

    # Infer feature encoder dimensions
    fe_raw_dims, fe_processed_dims = [], []
    for feat_name in node_encoders:
        fe_cfg = cfg.node_encoder_params.get(feat_name)
        raw_dim = dataset._data[f"rawfeat_{feat_name}"].shape[1]
        encoded_dim = raw_dim if fe_cfg.layers == 0 else fe_cfg.enc_dim

        fe_raw_dims.append(raw_dim)
        fe_processed_dims.append(encoded_dim)

    # Infer composed feature dimension and message passing input dimension
    if len(node_encoders) > 1:  # single feature encoder
        composed_fe_dim_in = None
        mp_dim_in = fe_processed_dims[0]
    else:  # composed feature encoder
        composed_fe_dim_in = sum(fe_processed_dims)
        fe_cfg = cfg.node_encoder_params.Composed
        mp_dim_in = composed_fe_dim_in if fe_cfg.layers == 0 else fe_cfg.enc_dim

    # Infer prediction head input dimension
    pred_head_dim_in = (
        mp_dim_in if cfg.model.mp_layers == 0
        else cfg.model.hid_dim
    )

    inferred_dims_dict = {
        "fe_raw_dims": fe_raw_dims,
        "fe_processed_dims": fe_processed_dims,
        "composed_fe_dim_in": composed_fe_dim_in,
        "mp_dim_in": mp_dim_in,
        "pred_head_dim_in": pred_head_dim_in,
        "dim_out": dim_out,
    }
    nleval.logger.info(f"Node encoders: {node_encoders}")
    nleval.logger.info(f"Inferred module dimensions:\n{pformat(inferred_dims_dict)}")

    with open_dict(cfg):
        cfg._shared = inferred_dims_dict
