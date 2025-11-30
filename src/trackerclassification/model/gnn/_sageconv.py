import torch
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from typing import Optional

import logging
LOGGER = logging.getLogger(__name__)

from ._base import GraphNeuralNetworkBase    


class SAGEConvModel(GraphNeuralNetworkBase):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        num_unique_ids: int,
        num_leds: int,
        dropout: float = 0.2,
    ):
        super().__init__(in_dim=2, num_unique_ids=num_unique_ids, num_leds=num_leds)

        convs = []
        last_dim = in_dim
        for h in hidden_dims:
            convs.append(SAGEConv(last_dim, h))
            last_dim = h
        self.convs = nn.ModuleList(convs)

        self.dropout = nn.Dropout(dropout)

        heads = self.heads(last_dim)
        self.tracker_head = heads["tracker"]
        self.led_head = heads["led"]

    def forward(
        self,
        data: Data | Batch,
        tracker_labels: Optional[torch.Tensor] = None,
        led_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        x, edge_index = data.x, data.edge_index

        edge_weight: Optional[torch.Tensor] = None
        if getattr(data, "edge_attr", None) is not None:
            dist_sq: torch.Tensor = data.edge_attr[:, 0]
            edge_weight = torch.exp(-dist_sq)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.dropout(x)

        tracker_logits = self.tracker_head(x)  # (N, C1)
        led_logits = self.led_head(x)  # (N, C2)

        if tracker_labels is not None and led_labels is not None:
            tracker_loss = F.cross_entropy(tracker_logits, tracker_labels)
            led_loss = F.cross_entropy(led_logits, led_labels)
            loss = tracker_loss + led_loss
        
        logits = torch.cat([tracker_logits, led_logits], dim=-1)

        return {
            "loss": loss if tracker_labels is not None and led_labels is not None else None,
            "logits": logits,
            "tracker_logits": tracker_logits,
            "led_logits": led_logits,
        }