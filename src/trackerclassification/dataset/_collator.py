from __future__ import annotations
from typing import Any, Dict, List, Mapping

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

from ..utils.typing import Matrix_Nx2_f


class TrackingDataCollator:
    """
    A collator class to batch samples of tracker data for PyTorch DataLoader.
    """
    def __call__(self, batch: List[Mapping[str, Matrix_Nx2_f]]) -> Dict[str, Any]:
        """
        Collate a batch of samples into a single batch for training.

        Args:
            batch (List[Mapping[str, Matrix_Nx2_f]]): A list of sample dictionaries to collate. The input dictionaries
                                             look like this: 
                                                {
                                                    "x": torch.Tensor of shape (N, 2),
                                                    "y": torch.Tensor of shape (N, 2),
                                                    "warning": Optional[str],
                                                }

        Returns:
            Dict[str, Any]: A dictionary containing the batched data. If all samples have the same shape,
                            the data is stacked into tensors; otherwise, lists of tensors are returned.
        """
        batch = [{k: v for k, v in b.items() if k != "warning"} for b in batch]

        x_batch = [b["x"] for b in batch]
        y_batch = [b["y"] for b in batch]

        x_shapes = {tuple(x.shape) for x in x_batch}
        y_shapes = {tuple(y.shape) for y in y_batch}
        if len(x_shapes) == 1 and len(y_shapes) == 1:
            return {"x": torch.stack(x_batch, 0), "y": torch.stack(y_batch, 0)}

        return {"x": x_batch, "y": y_batch}
    







from typing import Any, Dict, List

import torch
from torch_geometric.data import Data, Batch


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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        data_list: List[Data] = []
        tracker_labels_list: List[torch.Tensor] = []
        led_labels_list: List[torch.Tensor] = []
        
        for sample in features:
            x: torch.Tensor = sample["x"]          # (N_i, 2), float32
            y: torch.Tensor = sample["y"]          # (N_i, 2), int64

            # Build a PyG Data object for this sample.
            # We use x directly as node features (coordinates).
            data_list.append(Data(x=x))

            # Split labels into tracker id and LED index.
            tracker_labels_list.append(y[:, 0])    # (N_i,)
            led_labels_list.append(y[:, 1])        # (N_i,)

        # Concatenate over all samples in the batch.
        batch = Batch.from_data_list(data_list)
        tracker_labels = torch.cat(tracker_labels_list, dim=0).long()
        led_labels = torch.cat(led_labels_list, dim=0).long()

        return {
            "data": batch,
            "tracker_labels": tracker_labels,
            "led_labels": led_labels,
        }