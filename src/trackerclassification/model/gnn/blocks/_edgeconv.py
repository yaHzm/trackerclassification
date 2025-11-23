import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class EdgeConvBlock(MessagePassing):
    """
    EdgeConv without torch-cluster.

    For each graph in the batch:
        - builds a k-NN graph in pure PyTorch
        - applies MLP on concat(x_i, x_j - x_i)
        - aggregates with max over neighbors
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        super().__init__(aggr="max")
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (N, C_in) node features
            batch: (N,)      graph ids for each node
        Returns:
            (N, C_out) updated node features
        """
        assert x is not None, "EdgeConv.forward: x is None"
        assert batch is not None, "EdgeConv.forward: batch is None"

        edge_index = self._build_knn_graph(x, batch)  # [2, E]
        out = self.propagate(edge_index=edge_index, x=x)
        return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Message function:
            m_ij = MLP( concat(x_i, x_j - x_i) )
        """
        out = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(out)

    def _build_knn_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Build a k-NN graph in pure PyTorch (per graph in the batch).

        Args:
            x:     (N, C)
            batch: (N,)
        Returns:
            edge_index: (2, E)
        """
        device = x.device
        edge_index_list = []
        offset = 0

        for b in batch.unique(sorted=True):
            mask = (batch == b)
            x_b = x[mask]                # [Nb, C]
            Nb = x_b.size(0)
            if Nb == 0:
                continue

            dist = torch.cdist(x_b, x_b)  # [Nb, Nb]
            dist.fill_diagonal_(float("inf"))

            k_eff = min(self.k, max(Nb - 1, 1))
            knn = dist.topk(k_eff, largest=False).indices  # [Nb, k_eff]

            row = torch.arange(Nb, device=device).view(-1, 1).expand_as(knn)
            col = knn

            edge_index_b = torch.stack(
                [row.reshape(-1) + offset, col.reshape(-1) + offset],
                dim=0,
            )  # [2, Nb * k_eff]

            edge_index_list.append(edge_index_b)
            offset += Nb

        if not edge_index_list:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        return torch.cat(edge_index_list, dim=1)