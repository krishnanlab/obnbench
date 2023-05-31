from typing import List

import torch
import torch.nn as nn
import torch_geometric.nn as pygnn

from model_layers.misc import RawFeatNorm


class BaseFeatureEncoder(nn.Module):

    def __init__(
        self,
        dim_feat: int,
        dim_encoder: int = 128,
        layers: int = 1,
        raw_dropout: float = 0.0,
        raw_bn: bool = True,
        raw_bn_affine: bool = True
    ):
        super().__init__()

        self.dim_feat = dim_feat
        self.dim_encoder = dim_encoder

        self.raw_bn = RawFeatNorm(dim_feat, affine=raw_bn_affine) if raw_bn else None
        self.raw_dropout = nn.Dropout(raw_dropout)

        if layers > 0:
            self.enc = pygnn.MLP(
                in_channels=dim_feat,
                out_channels=dim_encoder,
                hidden_channels=dim_encoder,
                num_layers=layers,
                norm=None,
                act="relu",
                plain_last=True,
            )
        else:
            self.enc = nn.Identity()

        fe_name = self.__class__.__name__.replace("FeatureEncoder", "")
        self._raw_feat_name = f"rawfeat_{fe_name}"

    def get_raw_features(self, batch) -> torch.Tensor:
        return getattr(batch, self._raw_feat_name)

    # def preprocess(self, *args, **kwargs):
    #     pass

    def forward(self, batch):
        x = self.get_raw_features(batch)
        if self.raw_bn is not None:
            x = self.raw_bn(x)
        x = self.raw_dropout(x)
        batch.x = self.enc(x)
        return batch


class CompoasedFeatureEncoder(BaseFeatureEncoder):

    def __init__(self, *args, fe_list: List[BaseFeatureEncoder], **kwargs):
        super().__init__(*args, **kwargs)
        if len(fe_list) < 1:
            raise ValueError("Empty sub-feature-encoder list.")
        self.feature_encoders = nn.ModuleList(fe_list)

    def get_raw_features(self, batch):
        xs = [enc(batch).x for enc in self.featture_encoders]
        return torch.cat(xs, dim=1)


class OneHotLogDegFeatureEncoder(BaseFeatureEncoder):
    ...


class ConstantFeatureEncoder(BaseFeatureEncoder):
    ...


class RandomNormalFeatureEncoder(BaseFeatureEncoder):
    ...


class OrbitalFeatureEncoder(BaseFeatureEncoder):
    ...


class SVDFeatureEncoder(BaseFeatureEncoder):
    ...


class LapEigMapFeatureEncoder(BaseFeatureEncoder):
    ...


class RandomWalkDiagFeatureEncoder(BaseFeatureEncoder):
    ...


class RandProjGaussianFeatureEncoder(BaseFeatureEncoder):
    ...


class RandProjSparseFeatureEncoder(BaseFeatureEncoder):
    ...


class EmbLINE1FeatureEncoder(BaseFeatureEncoder):
    ...


class EmbLINE2FeatureEncoder(BaseFeatureEncoder):
    ...


class EmbNode2vecFeatureEncoder(BaseFeatureEncoder):
    ...


class EmbWalkletsFeatureEncoder(BaseFeatureEncoder):
    ...


class LabelReuseFeatureEncoder(BaseFeatureEncoder):
    ...
