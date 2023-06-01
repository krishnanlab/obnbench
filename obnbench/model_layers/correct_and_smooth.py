from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.utils import scatter


class PatchedCorrectAndSmooth(CorrectAndSmooth):
    """Correct and smooth with improved forward to simplify calling.

    Normalized adjacency used for propagation is computed on-the-fly upon
    calling the forward function. Furthermore, three different normalization
    schemes ('left', 'right', and 'sym') can be selected independently for the
    correction and smoothing steps. Optionally, the computed normalized
    adjacencies can be cached for future computation.

    """

    def __init__(
        self,
        *args,
        norm_correct: str = "sym",
        norm_smooth: str = "left",
        cache: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.norm_correct = norm_correct
        self.norm_smooth = norm_smooth
        self.cache = cache

    @property
    def cached_correct_edge_weight(self) -> Optional[Tensor]:
        return getattr(self, "_cached_correct_edge_weight", None)

    @property
    def cached_smooth_edge_weight(self) -> Optional[Tensor]:
        return getattr(self, "_cached_smooth_edge_weight", None)

    def forward(
        self,
        y_soft: Tensor,
        y_true: Tensor,
        mask: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ):
        correct_edge_weight, smooth_edge_weight = self._get_adjs(
            y_soft.shape[0],
            edge_index,
            edge_weight,
        )
        y_soft = self.correct(y_soft, y_true, mask, edge_index, correct_edge_weight)
        y_soft = self.smooth(y_soft, y_true, mask, edge_index, smooth_edge_weight)
        return y_soft

    def _get_adjs(
        self,
        num_nodes: int,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Try to load from cache first
        correct_edge_weight = self._cached_correct_edge_weight
        smooth_edge_weight = self._cached_smooth_edge_weight

        # Compute if cache not available
        if correct_edge_weight is None or smooth_edge_weight is None:
            if edge_weight is None:
                edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)

            row, col = edge_index
            deg_inv = scatter(
                edge_weight,
                col,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            ).pow_(-1)

            args = (row, col, edge_weight, deg_inv)
            correct_edge_weight = self._get_norm_edge_weight(*args, self.norm_correct)
            if self.norm_smooth == self.norm_correct:
                smooth_edge_weight = correct_edge_weight
            else:
                smooth_edge_weight = self._get_norm_edge_weight(*args, self.norm_smooth)

            if self.cache:
                self._cached_correct_edge_weight = correct_edge_weight
                self._cached_smooth_edge_weight = smooth_edge_weight

        return correct_edge_weight, smooth_edge_weight

    @staticmethod
    def _get_norm_edge_weight(
        row: Tensor,
        col: Tensor,
        edge_weight: Tensor,
        deg_inv: Tensor,
        norm_type: str,
    ) -> Tensor:
        if norm_type == "sym":
            deg_inv_sqrt = deg_inv.pow(0.5)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif norm_type == "left":
            edge_weight = deg_inv[row] * edge_weight
        elif norm_type == "right":
            edge_weight = edge_weight * deg_inv[col]
        else:
            raise ValueError(f"Unknown norm_type {norm_type!r}")
        return edge_weight
