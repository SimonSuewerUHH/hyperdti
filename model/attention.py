from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
)
from torch_geometric.utils import softmax


class DrugProteinAttention(MessagePassing):
    """
    Bipartiter GAT-Conv, das Drugs → Proteine und Proteine → Drugs
    in zwei Attention‐Schritten durchführt, mit Residual, LayerNorm und Dropout.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.1,
            bias: bool = True,
            **kwargs,
    ):
        super().__init__(node_dim=0, aggr='add', **kwargs)

        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Attention-Parameter
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        # Norm & Dropout für Residual-Pfad
        self.norm_src = nn.LayerNorm(heads * out_channels)
        self.norm_dst = nn.LayerNorm(heads * out_channels)
        self.res_dropout = nn.Dropout(dropout)

        # Bias für den Output
        total_dim = heads * out_channels if concat else out_channels
        if bias:
            self.bias = Parameter(torch.Tensor(total_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(
            self,
            x_src: Tensor,
            x_dst: Tensor,
            edge_index: Adj,
    ) -> Tuple[Tensor, Tensor]:
        # 1) Lineare Projektion und Aufteilen in Heads
        H, C = self.heads, self.out_channels
        h_src = x_src.view(-1, H, C)  # [N_src, H, C]
        h_dst = x_dst.view(-1, H, C)  # [N_dst, H, C]

        # 2) Attention-Coefficients vorbereiten
        alpha_src = (h_src * self.att_src).sum(dim=-1)  # [N_src, H]
        alpha_dst = (h_dst * self.att_dst).sum(dim=-1)  # [N_dst, H]

        # --- Drug → Protein ---
        alpha = (alpha_src, alpha_dst)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=None)
        out_dst = self.propagate(
            edge_index, x=(h_src, h_dst), alpha=alpha,
            size=(h_src.size(0), h_dst.size(0))
        )  # [N_dst, H, C]

        # Heads zusammenführen
        if self.concat:
            out_dst = out_dst.view(-1, H * C)
            res_dst = x_dst.new_zeros(x_dst.size(0), H * C)
        else:
            out_dst = out_dst.mean(dim=1)
            res_dst = x_dst.new_zeros(x_dst.size(0), C)

        # 3) Residual + LayerNorm + Aktivierung
        res_dst = self.res_dropout(out_dst)
        dst = self.norm_dst(res_dst + (h_dst.view_as(res_dst)))
        dst = F.relu(dst)
        if self.bias is not None:
            dst = dst + self.bias

        # --- Protein → Drug (reverse edges) ---
        rev_edge_index = edge_index[[1, 0]]
        alpha = (alpha_dst, alpha_src)
        alpha = self.edge_updater(rev_edge_index, alpha=alpha, edge_attr=None)
        out_src = self.propagate(
            rev_edge_index, x=(h_dst, h_src), alpha=alpha,
            size=(h_dst.size(0), h_src.size(0))
        )  # [N_src, H, C]

        if self.concat:
            out_src = out_src.view(-1, H * C)
            res_src = x_src.new_zeros(x_src.size(0), H * C)
        else:
            out_src = out_src.mean(dim=1)
            res_src = x_src.new_zeros(x_src.size(0), C)

        res_src = self.res_dropout(out_src)
        src = self.norm_src(res_src + (h_src.view_as(res_src)))
        src = F.relu(src)
        if self.bias is not None:
            src = src + self.bias

        return src, dst

    def edge_update(
            self,
            alpha_j: Tensor,
            alpha_i: Optional[Tensor],
            edge_attr: Optional[Tensor],
            index: Tensor,
            ptr: Optional[Tensor],
    ) -> Tensor:
        # Summiere Src- und Dst-Anteile
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        # LeakyReLU
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # Softmax über eingehende Kanten
        alpha = softmax(alpha, index, num_nodes=None)
        # Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        # gewichtete Summe der Nachbar-Repräsentationen
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.in_channels_src}->{self.out_channels}, '
                f'{self.in_channels_dst}->{self.out_channels}, '
                f'heads={self.heads})')
