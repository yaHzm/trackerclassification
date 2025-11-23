from __future__ import annotations
from typing import Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch_geometric.data import Data, Batch

from .._base import ModelBase


class GraphNeuralNetworkBase(ModelBase, ABC):
    def __init__(self, in_dim: int, num_unique_ids: int, num_leds: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_unique_ids = num_unique_ids
        self.num_leds = num_leds

    def _ensure_batch(self, data: Data | Batch) -> Batch:
        return data if isinstance(data, Batch) else Batch.from_data_list([data])

    def heads(self, feat_dim: int) -> nn.ModuleDict:
        return nn.ModuleDict({
            "tracker": nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2), nn.ReLU(),
                nn.Linear(feat_dim // 2, self.num_unique_ids)
            ),
            "led": nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2), nn.ReLU(),
                nn.Linear(feat_dim // 2, self.num_leds)
            ),
        })

    @abstractmethod
    def forward(self, data: Data | Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        pass