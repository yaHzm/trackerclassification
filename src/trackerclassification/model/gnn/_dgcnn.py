from typing import Tuple, Sequence
import torch
from torch import nn
from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

LOGGER = logging.getLogger(__name__)

from ._base import GraphNeuralNetworkBase
from .blocks import EdgeConvBlock           


class DGCNNModel(GraphNeuralNetworkBase):
    def __init__(
        self,
        in_dim: int,
        num_unique_ids: int,
        num_leds: int,
        hidden_dims: Sequence[int],
        edge_dim: int = 3,
    ) -> None:
        super().__init__(in_dim=in_dim, num_unique_ids=num_unique_ids, num_leds=num_leds)

        convs = []
        last_dim = in_dim
        for h in hidden_dims:
            convs.append(EdgeConvBlock(last_dim, edge_dim=edge_dim, out_dim=h))
            last_dim = h
        self.convs = nn.ModuleList(convs)

        self.feat_dim = sum(hidden_dims)
        heads = self.heads(self.feat_dim)
        self.tracker_head = heads["tracker"]
        self.led_head = heads["led"]

    def forward_gnn(self, data: Data | Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self._ensure_batch(data)
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        feats = []
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)  # MessagePassing handles batching
            feats.append(h)

        h_cat = torch.cat(feats, dim=-1)
        tracker_logits = self.tracker_head(h_cat)
        led_logits = self.led_head(h_cat)
        return tracker_logits, led_logits

    def forward(
        self,
        data: Data | Batch,
        tracker_labels: Optional[torch.Tensor] = None,
        led_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        tracker_logits, led_logits = self.forward_gnn(data)

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