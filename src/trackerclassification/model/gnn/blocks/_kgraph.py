from torch_geometric.utils import dense_to_sparse
import torch

def full_edge_index(n: int):
    adj = torch.ones((n, n), dtype=torch.bool)
    adj.fill_diagonal_(False)
    edge_index, _ = dense_to_sparse(adj)
    return edge_index