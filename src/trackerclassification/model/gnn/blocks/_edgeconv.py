from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class EdgeConvBlock(MessagePassing):
    """
    Static EdgeConv-style block using PyG's MessagePassing.

    For each edge i->j:
        m_ij = MLP([x_i, x_j, edge_attr_ij])
    Node update:
        h_i = max_j m_ij
    """

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int) -> None:
        super().__init__(aggr="max")  # DGCNN uses max-aggregation
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,              # (N, Fin)
        edge_index: torch.Tensor,     # (2, E)
        edge_attr: torch.Tensor,      # (E, Fe)
    ) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(
        self,
        x_i: torch.Tensor,            # (E, Fin), receiver features
        x_j: torch.Tensor,            # (E, Fin), sender features
        edge_attr: torch.Tensor,      # (E, Fe)
    ) -> torch.Tensor:
        h_ij = torch.cat((x_i, x_j, edge_attr), dim=-1)  # (E, 2*Fin+Fe)
        return self.mlp(h_ij)                            # (E, Fout)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        # aggr_out: (N, Fout) after max over neighbors
        return aggr_out