from typing import Any, Dict, List

import torch
from torch import Tensor
from torch_geometric.data import Data, Batch

# Assuming you already have this somewhere in your codebase:
# def full_edge_index(num_nodes: int) -> Tensor:
#     ...

class PyGTrackingAffinityCollator:
    """
    Data collator for the new 'LED grouping' task.

    This collator:
      - Converts a list of dataset samples into a single PyG `Batch`
        with all LED nodes concatenated.
      - Extracts tracker and LED labels per node and concatenates them.
      - Builds binary edge labels indicating whether two LEDs
        belong to the same tracker.

    Expected input (per sample from `TrackingDataset`):
        {
            "x": Tensor of shape (N_i, 2), float32   # LED coordinates
            "y": Tensor of shape (N_i, 2), int64     # [tracker_id, led_index]
            # "warning": optional, ignored here
        }

    Output:
        {
            "data": Batch
                - .x:         (sum_i N_i, 2)
                - .edge_index (2, E_total)
                - .edge_attr  (E_total, 3)  # dist_sq, dir_x, dir_y
            "tracker_labels": LongTensor of shape (sum_i N_i,)  # per-node tracker id
            "led_labels":     LongTensor of shape (sum_i N_i,)  # per-node led index (kept for debugging)
            "edge_labels":    LongTensor of shape (E_total,)    # 1 if same tracker, else 0
        }
    """

    def _get_edge_features(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Compute edge features (dist_sq, dir_x, dir_y) for each edge i->j.

        Args:
            x:          (N, 2) node coordinates
            edge_index: (2, E) edge list (i -> j)

        Returns:
            edge_attr:  (E, 3) tensor:
                            [ dist_sq,
                              dir_x_normalized,
                              dir_y_normalized ]
        """
        row, col = edge_index  # i, j

        rel = x[col] - x[row]             # (E, 2)
        dist_sq = (rel ** 2).sum(dim=-1)  # (E,)
        dist = dist_sq.sqrt().clamp_min(1e-8)

        dir_norm = rel / dist.unsqueeze(-1)   # (E, 2)

        edge_attr = torch.cat([
            dist_sq.unsqueeze(-1),   # (E, 1)
            dir_norm,                # (E, 2)
        ], dim=-1)

        return edge_attr

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        data_list: List[Data] = []
        tracker_labels_list: List[Tensor] = []
        led_labels_list: List[Tensor] = []

        for sample in features:
            x: Tensor = sample["x"]  # (N_i, 2), float32
            y: Tensor = sample["y"]  # (N_i, 2), int64: [tracker_id, led_index]

            num_nodes = x.size(0)
            edge_index = full_edge_index(num_nodes)  # (2, E_i)

            # Build a PyG Data object for this sample.
            data = Data(
                x=x,
                edge_index=edge_index,
            )
            data_list.append(data)

            # Split labels into tracker id and LED index.
            tracker_labels_list.append(y[:, 0])  # (N_i,)
            led_labels_list.append(y[:, 1])      # (N_i,)

        # Concatenate over all samples in the batch.
        batch = Batch.from_data_list(data_list)

        # Compute edge_attr for the batched graph.
        if batch.edge_index is not None:
            batch.edge_attr = self._get_edge_features(batch.x, batch.edge_index)

        # Concatenate node-wise labels in the same order as nodes in the Batch.
        tracker_labels = torch.cat(tracker_labels_list, dim=0).long()  # (N_total,)
        led_labels = torch.cat(led_labels_list, dim=0).long()          # (N_total,)

        # Build edge labels: 1 if two LEDs belong to the same tracker, else 0.
        row, col = batch.edge_index  # (E_total,)
        edge_labels = (tracker_labels[row] == tracker_labels[col]).long()  # (E_total,)

        return {
            "data": batch,
            "tracker_labels": tracker_labels,
            "led_labels": led_labels,
            "edge_labels": edge_labels,
            "labels": edge_labels,  # <– HF Trainer will use this for label_ids / metrics
        }


from typing import Any, Dict, List

import torch
from torch_geometric.data import Data, Batch

from ..model.gnn.blocks import full_edge_index


class PyGTrackingDataCollator:
    """
    Data collator for tracker LED samples.

    This collator:
      - Converts a list of dataset samples into a single PyG `Batch`
        with all LED nodes concatenated.
      - Extracts tracker and LED labels per node and concatenates them
        into flat 1D tensors.

    Expected input (per sample from `TrackingDataset`):
        {
            "x": Tensor of shape (N_i, 2), float32   # LED coordinates
            "y": Tensor of shape (N_i, 2), int64     # [tracker_id, led_index]
            # "warning": optional, ignored here
        }

    Output (batch fed into the model's `forward`):
        {
            "data": Batch           # PyG Batch, with .x of shape (sum_i N_i, 2)
            "tracker_labels": LongTensor of shape (sum_i N_i,)
            "led_labels": LongTensor of shape (sum_i N_i,)
        }
    """
    def _get_edge_features(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute edge features (dist_sq, dir_x, dir_y) for each edge i->j.

        Args:
            x:          (N, 2) node coordinates
            edge_index: (2, E) edge list (i -> j)

        Returns:
            edge_attr:  (E, 3) tensor:
                            [ dist_sq,
                            dir_x_normalized,
                            dir_y_normalized ]
        """
        row, col = edge_index  # i, j

        rel = x[col] - x[row]             # (E, 2)
        dist_sq = (rel ** 2).sum(dim=-1)  # (E,)
        dist = dist_sq.sqrt().clamp_min(1e-8)

        dir_norm = rel / dist.unsqueeze(-1)   # (E, 2)

        edge_attr = torch.cat([
            dist_sq.unsqueeze(-1),   # (E, 1)
            dir_norm,                # (E, 2)
        ], dim=-1)

        return edge_attr

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        data_list: List[Data] = []
        tracker_labels_list: List[torch.Tensor] = []
        led_labels_list: List[torch.Tensor] = []
        
        for sample in features:
            x: torch.Tensor = sample["x"]          # (N_i, 2), float32
            y: torch.Tensor = sample["y"]          # (N_i, 2), int64

            edge_index = full_edge_index(x.size(0))

            # Build a PyG Data object for this sample.
            # We use x directly as node features (coordinates).
            data_list.append(Data(x=x, edge_index=edge_index))

            # Split labels into tracker id and LED index.
            tracker_labels_list.append(y[:, 0])    # (N_i,)
            led_labels_list.append(y[:, 1])        # (N_i,)

        # Concatenate over all samples in the batch.
        batch = Batch.from_data_list(data_list)

        if batch.edge_index is not None:
            batch.edge_attr = self._get_edge_features(batch.x, batch.edge_index)

        tracker_labels = torch.cat(tracker_labels_list, dim=0).long()
        led_labels = torch.cat(led_labels_list, dim=0).long()

        return {
            "data": batch,
            "tracker_labels": tracker_labels,
            "led_labels": led_labels,
        }