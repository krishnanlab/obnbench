import time
from typing import Any, Dict, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import AdamW

import metrics
from model_layers import feature_encoders, mp_layers


class ModelModule(pl.LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.feature_encoder = build_feature_encoder(cfg)
        self.mp_layers = build_mp_module(cfg)
        self.pred_head = build_pred_head(cfg)

        # FIX: post prop and corr

        self.post_propagation = None
        self.post_correction = None

        self.setup_metrics(cfg)

        # self.reset_parameters()

        self.save_hyperparameters(cfg)

    def setup_metrics(self, cfg):
        self.metrics = nn.ModuleDict()
        metric_kwargs = {
            "task": "multilabel",
            "num_labels": cfg._shared.dim_out,
            "validate_args": False,
            "average": "macro",
        }
        for split in ["train", "val", "test"]:
            for metric_name in cfg.metrics:
                metric_cls = getattr(metrics, metric_name)
                self.metrics[f"{split}/{metric_name}"] = metric_cls(**metric_kwargs)

    def forward(self, batch):
        batch = self.feature_encoder(batch)
        batch = self.mp_layers(batch)
        pred, true = self.pred_head(batch)
        pred = self._post_processing(pred, batch)  # FIX: move to end of steps
        return pred, true

    def _shared_step(self, batch, split):
        # XXX: only allow full batch training now for several reasons (1) post
        # propagation and correction needs access to the full graph, (2) metrics
        # should be computed on the full graph.
        # To resolve (1), need to extract post processing (accordingly the
        # metrics computations also) to a shared private end_of_epoch func.
        # This will also automatically resolve (2).
        tic = time.perf_counter()
        batch.split = split

        # NOTE: split masking is done inside the prediction head
        pred, true = self(batch)

        logger_opts = {
            "on_step": False,
            "on_epoch": True,
            "logger": True,
            "batch_size": pred.shape[0],
        }

        # Compute classification loss for training
        if split == "train":
            loss = F.binary_cross_entropy(pred, true)
            self.log(f"{split}/loss", loss.item(), **logger_opts)
        else:
            loss = None

        # Compute and log metrics at eval epoch
        self._maybe_log_metrics(pred, true, split, logger_opts)

        self.log(f"{split}/time_epoch", time.perf_counter() - tic, **logger_opts)

        return loss

    # TODO: reset_parameters

    @torch.no_grad()
    def _maybe_log_metrics(self, pred, true, split, logger_opts):
        if (self.current_epoch + 1) % self.hparams.eval_interval != 0:
            return

        for metric_name, metric_obj in self.metrics.items():
            if metric_name.startswith(f"{split}/"):
                metric_obj(pred, true)
                self.log(metric_name, metric_obj, **logger_opts)

    def _post_processing(self, pred, batch):
        if self.post_propagation is not None:
            pred = self.post_propagation(pred, batch.edge_index)

        pred = F.sigmoid(pred)

        if self.post_correction is not None and batch.split != "train":
            # pred = self.post_correction.correct(pred, batch.y, batch.trian_mask, DAD)
            # pred = self.post_correction.smooth(pred, batch.y, batch.trian_mask, DA)
            pred = self.post_correction(
                pred,
                batch.y,
                batch.train_mask,
                batch.edge_index,
                batch.edge_weight,
            )

        return pred

    def training_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="train")

    def validation_step(self, batch, *args, **kwargs):
        self._shared_step(batch, split="val")
        # HACK: Enable early testing that was deliberaly disabled by Lightning
        # https://github.com/Lightning-AI/lightning/issues/5245
        self._shared_step(batch, split="test")

    def test_step(self, batch, *args, **kwargs):
        self._shared_step(batch, split="test")

    def configure_optimizers(self):
        # FIX: parse from config
        optimizer = AdamW(
            self.parameters(),
            lr=0.001,
            weight_decay=1e-6,
        )
        return optimizer
        # scheduler = ???
        # return [optimizer], [scheduler]


class MPModule(nn.Module):

    _residual_opts = ["none", "skipsum", "catlast", "catall", "catcompact"]

    def __init__(
        self,
        mp_cls: nn.Module,
        dim: int,
        num_layers: int,
        residual_type: str = "none",
        mp_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        # TODO: residual norm?

        # Set residual_type last to make sure we have registered all params
        # Setting residual_type will automatically set up the forward function
        # and the dimensions for the hidden layers.
        self.residual_type = residual_type

        # Set up the message passing layers using the dimensions prepared
        self.layers = nn.ModuleList()
        mp_kwargs = mp_kwargs or {}
        for dim_in, dim_out in zip(self._layer_dims[:-1], self._layer_dims[1:]):
            self.layers.append(mp_cls(dim_in, dim_out, **mp_kwargs))

    def extra_repr(self) -> str:
        extra_param_dict = {
            "residual_type": self.residual_type,
            # "residual_norm": self.residual_norm,
        }
        return "\n".join(f"{i}: {j}" for i, j in extra_param_dict.items())

    @property
    def residual_type(self) -> str:
        return self._residual_type

    @residual_type.setter
    def residual_type(self, val: str):
        if val == "none":
            self._forward = self._stack_forward
            self._layer_dims = [self.dim] * (self.num_layers + 1)
        elif val == "skipsum":
            self._forward = self._skipsum_forward
            self._layer_dims = [self.dim] * (self.num_layers + 1)
        elif val == "catlast":
            self._forward = self._catlast_forward
            self._layer_dims = (
                [self.dim] * (self.num_layers - 1)
                + [self.dim * self.num_layers, self.dim]
            )
        elif val == "catall":
            self._forward = self._catall_forward
            self._layer_dims = (
                [self.dim * (i + 1) for i in range(self.num_layers)]
                + [self.dim]
            )
        elif val == "catcompact":
            self._forward = self._catcompact_forward
            self._layer_dims = (
                [self.dim] + [2 * self.dim] * (self.num_layers - 1) + [self.dim]
            )
        else:
            raise ValueError(
                f"Unknown residual type {val!r}, available options are:\n"
                f"    {self._residual_opts}",
            )
        self._residual_type = val

    def _stack_forward(self, batch):
        for layer in self.layers:
            batch = layer(batch)
        return batch

    def _skipsum_forward(self, batch):
        for layer in self.layers:
            x_prev = batch.x
            batch = layer(batch)
            batch.x += x_prev
        return batch

    def _catlast_forward(self, batch):
        xs = []
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                batch.x = torch.cat([batch.x] + xs, dim=1)
            batch = layer(batch)
            if i < self.layers - 1:
                xs.append(batch.x)
        return batch

    def _catall_forward(self, batch):
        for i, layer in enumerate(self.layers):
            x_prev = batch.x
            batch = layer(batch)
            if i < self.layers - 1:
                batch.x = torch.cat([batch.x, x_prev], dim=1)
        return batch

    def _catcompact_forward(self, batch):
        for i, layer in enumerate(self.layers):
            x_prev = batch.x
            batch = layer(batch)
            if i < self.layers - 1:
                batch.x = torch.cat([batch.x, x_prev], dim=1)
        return batch

    def forward(self, batch):
        return self._forward(batch)


class PredictionHeadModule(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_layers: int = 1,
        dim_inner: int = 128,
    ):
        super().__init__()
        self.layers = pygnn.MLP(
            in_channels=dim_in,
            out_channels=dim_out,
            hidden_channels=dim_inner,
            num_layers=num_layers,
            act="relu",
            norm="batch_norm",
            bias=True,
            plain_last=True,
        )

    def _apply_mask(self, batch, pred, true) -> Tuple[Tensor, Tensor]:
        if batch.split is not None:
            mask = batch[f"{batch.split}_mask"].squeeze(-1)
            pred, true = pred[mask], true[mask]
        return pred, true

    def forward(self, batch):
        pred = self.layers(batch.x)
        return self._apply_mask(batch, pred, batch.y)


def build_feature_encoder(cfg: DictConfig):
    feat_names = cfg.node_encoders.split("+")

    fe_list = []
    for feat_name in feat_names:
        fe_cfg = cfg.node_encoder_params.get(feat_name)
        fe_cls = getattr(feature_encoders, f"{feat_name}FeatureEncoder")
        fe = fe_cls(
            dim_feat=cfg._shared.fe_raw_dims[0],
            dim_encoder=cfg.model_params.hid_dim,
            layers=fe_cfg.layers,
            raw_dropout=fe_cfg.raw_dropout,
            raw_bn=fe_cfg.raw_bn,
        )
        fe_list.append(fe)

    if len(fe_list) == 1:
        return fe_list[0]
    else:
        fe_cfg = cfg.node_encoder_params.Compoased
        return feature_encoders.CompoasedFeatureEncoder(
            dim_feat=cfg._shared.composed_fe_dim_in,
            dim_encoder=cfg.model_params.hid_dim,
            layers=fe_cfg.layers,
            raw_dropout=fe_cfg.raw_dropout,
            raw_bn=fe_cfg.raw_bn,
            fe_list=fe_list,
        )


def build_mp_module(cfg: DictConfig):
    mp_cls = getattr(mp_layers, cfg.model_params.mp_type)
    mp_kwargs = cfg.model_params.get("mp_kwargs", None)
    return MPModule(
        mp_cls,
        dim=cfg.model_params.hid_dim,
        num_layers=cfg.model_params.mp_layers,
        residual_type=cfg.model_params.mp_residual_type,
        mp_kwargs=mp_kwargs,
    )


def build_pred_head(cfg: DictConfig):
    return PredictionHeadModule(
        dim_in=cfg._shared.pred_head_dim_in,
        dim_out=cfg._shared.dim_out,
        num_layers=cfg.model_params.pred_head_layers,
        dim_inner=cfg.model_params.hid_dim,
    )
