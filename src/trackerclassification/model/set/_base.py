# models/set/_base.py (for example)

from __future__ import annotations
from typing import Dict, Optional
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from .._base import ModelBase


class SetModelBase(ModelBase, ABC):
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
                nn.Linear(feat_dim // 2, self.num_unique_ids),
            ),
            "led": nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2), nn.ReLU(),
                nn.Linear(feat_dim // 2, self.num_leds),
            ),
        })

    def _build_output(
        self,
        tracker_logits: torch.Tensor,
        led_logits: torch.Tensor,
        tracker_labels: Optional[torch.Tensor] = None,
        led_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        loss: Optional[torch.Tensor] = None
        if tracker_labels is not None and led_labels is not None:
            loss_tracker = F.cross_entropy(tracker_logits, tracker_labels)
            loss_led = F.cross_entropy(led_logits, led_labels)
            loss = loss_tracker + loss_led

        logits = torch.cat([tracker_logits, led_logits], dim=-1)

        return {
            "loss": loss,
            "logits": logits,
            "tracker_logits": tracker_logits,
            "led_logits": led_logits,
        }

    @abstractmethod
    def forward(
        self,
        data: Data | Batch,
        tracker_labels: Optional[torch.Tensor] = None,
        led_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        pass