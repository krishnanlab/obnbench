import time
from collections import OrderedDict
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

act_register = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "elu": nn.ELU,
}

norm_register = {
    "BatchNorm": pygnn.BatchNorm,
    "LayerNorm": pygnn.LayerNorm,
    "PairNorm": pygnn.PairNorm,
    "DiffGroupNorm": pygnn.DiffGroupNorm,
    "none": nn.Identity,
}


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
            self._maybe_log_grad_norm(logger_opts)
        else:
            loss = None

        # Compute and log metrics at eval epoch
        self._maybe_log_metrics(pred, true, split, logger_opts)

        self.log(f"{split}/time_epoch", time.perf_counter() - tic, **logger_opts)

        return loss

    # TODO: reset_parameters

    def _maybe_log_grad_norm(self, logger_opts):
        if not self.hparams.watch_grad_norm:
            return

        grad_norms = [
            p.grad.detach().norm(2)
            for p in self.parameters() if p.grad is not None
        ]
        grad_norm = torch.stack(grad_norms).norm(2).item() if grad_norms else 0
        self.log("train/grad_norm", grad_norm, **logger_opts)

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
        # Note that the early access to testing performance is **not** used for
        # model selection and hyperparameter tuning by any means. Instead, it is
        # only used to see if the trend of the testing and validation curves
        # differ significantly, which indicates there is some problem witht the
        # data split.
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

    _residual_opts = [
        "none",
        "skipsum",
        "skipsumbnorm",
        "skipsumlnorm",
        "catlast",
        "catall",
        "catcompact",
    ]

    def __init__(
        self,
        mp_cls: nn.Module,
        dim: int,
        num_layers: int,
        dropout: float = 0.0,
        norm_type: str = "BatchNorm",
        act: str = "relu",
        act_first: bool = False,
        residual_type: str = "none",
        mp_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm_type = norm_type
        self.act = act
        self.act_first = act_first

        self.norm_type = norm_type
        self.norm_kwargs = norm_kwargs or {}
        if norm_type == "LayerNorm":
            self.norm_kwargs.setdefault("mode", "node")
        if norm_type != "PairNorm":
            # Need to pass feature dimension except for PairNorm
            self.norm_kwargs["in_channels"] = None

        # Set residual_type last to make sure we have registered all params
        # Setting residual_type will automatically set up the forward function
        # and the dimensions for the hidden layers.
        self.residual_type = residual_type

        # Set up the message passing layers using the dimensions prepared
        self.layers = nn.ModuleList()
        self.res_norms = nn.ModuleList()
        mp_kwargs = mp_kwargs or {}
        for dim_in in self._layer_dims:
            self._build_layer(mp_cls, dim_in, dim, mp_kwargs=mp_kwargs)

    def _build_layer(
        self,
        mp_cls: nn.Module,
        dim_in: int,
        dim_out: int,
        mp_kwargs: Dict[str, Any],
    ) -> nn.Module:
        conv_layer = mp_cls(dim_in, dim_out, **mp_kwargs)
        activation = act_register.get(self.act)()
        dropout = nn.Dropout(self.dropout)

        # Check if 'in_channels' is set to determine whether we need to pass
        # the feature dimension to initialize the normalization layer
        if "in_channels" in self.norm_kwargs:
            self.norm_kwargs["in_channels"] = dim_out
        norm_layer = norm_register.get(self.norm_type)(**self.norm_kwargs)

        # Graph convolution layer
        new_layer = nn.ModuleDict()
        new_layer["conv"] = conv_layer

        # Post convolution layers
        post_conv = []
        if self.act_first:
            post_conv.extend([("act", activation), "norm", norm_layer])
        else:
            post_conv.extend([("norm", norm_layer), ("act", activation)])
        post_conv.append(("dropout", dropout))
        new_layer["post_conv"] = nn.Sequential(OrderedDict(post_conv))
        self.layers.append(new_layer)

        # Residual normalizations
        if self.residual_type == "skipsumbnorm":
            self.res_norms.append(norm_register["BatchNorm"](dim_out))
        elif self.residual_type == "skipsumlnorm":
            self.res_norms.append(norm_register["LayerNorm"](dim_out, mode="node"))

    def extra_repr(self) -> str:
        return f"residual_type: {self.residual_type}"

    @property
    def residual_type(self) -> str:
        return self._residual_type

    @residual_type.setter
    def residual_type(self, val: str):
        if val == "none":
            self._forward = self._stack_forward
            self._layer_dims = [self.dim] * self.num_layers
        elif val in ["skipsum", "skipsumbnorm", "skipsumlnorm"]:
            self._forward = self._skipsum_forward
            self._layer_dims = [self.dim] * self.num_layers
        elif val == "catlast":
            self._forward = self._catlast_forward
            self._layer_dims = (
                [self.dim] * (self.num_layers - 1)
                + [self.dim * self.num_layers]
            )
        elif val == "catall":
            self._forward = self._catall_forward
            self._layer_dims = [self.dim * (i + 1) for i in range(self.num_layers)]
        else:
            raise ValueError(
                f"Unknown residual type {val!r}, available options are:\n"
                f"    {self._residual_opts}",
            )
        self._residual_type = val

    @staticmethod
    def _layer_forward(layer, batch):
        batch = layer["conv"](batch)
        batch.x = layer["post_conv"](batch.x)
        return batch

    def _stack_forward(self, batch):
        for layer in self.layers:
            batch = self._layer_forward(layer, batch)
        return batch

    def _skipsum_forward(self, batch):
        for i, layer in enumerate(self.layers):
            x_prev = batch.x
            batch = self._layer_forward(layer, batch)
            if self.res_norms:
                batch.x = self.res_norms[i](batch.x)
            batch.x = batch.x + x_prev
        return batch

    def _catlast_forward(self, batch):
        xs = []
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                batch.x = torch.cat([batch.x] + xs, dim=1)
            batch = self._layer_forward(layer, batch)
            if i < self.num_layers - 1:
                xs.append(batch.x)
        return batch

    def _catall_forward(self, batch):
        for i, layer in enumerate(self.layers):
            x_prev = batch.x
            batch = self._layer_forward(layer, batch)
            if i < self.num_layers - 1:
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
    return MPModule(
        mp_cls,
        dim=cfg.model_params.hid_dim,
        num_layers=cfg.model_params.mp_layers,
        dropout=cfg.model_params.dropout,
        norm_type=cfg.model_params.norm_type,
        act=cfg.model_params.act,
        act_first=cfg.model_params.act_first,
        residual_type=cfg.model_params.residual_type,
        mp_kwargs=cfg.model_params.get("mp_kwargs", None),
        norm_kwargs=cfg.model_params.get("norm_kwargs", None),
    )


def build_pred_head(cfg: DictConfig):
    return PredictionHeadModule(
        dim_in=cfg._shared.pred_head_dim_in,
        dim_out=cfg._shared.dim_out,
        num_layers=cfg.model_params.pred_head_layers,
        dim_inner=cfg.model_params.hid_dim,
    )
