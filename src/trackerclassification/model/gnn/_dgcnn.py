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
            loss = 0.8 * loss_tracker + 0.2 * loss_led

        logits = torch.cat([tracker_logits, led_logits], dim=-1)

        return {
            "loss": loss,
            "logits": logits,
            "tracker_logits": tracker_logits,
            "led_logits": led_logits,
        }
          


class SingleDGCNNModel(GraphNeuralNetworkBase):
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
        #led_logits = self.led_head(h_cat)
        return tracker_logits, None

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
            #loss_led = F.cross_entropy(led_logits, led_labels)
            #loss = 0.8 * loss_tracker + 0.2 * loss_led

        #logits = torch.cat([tracker_logits, led_logits], dim=-1)

        return {
            "loss": loss_tracker,
            "logits": tracker_logits,
            # "tracker_logits": tracker_logits,
            # "led_logits": led_logits,
        }
    






class AffinityDGCNNModel(GraphNeuralNetworkBase):
    def __init__(
        self,
        in_dim: int,
        num_unique_ids: int,
        num_leds: int,
        hidden_dims: Sequence[int],
        edge_dim: int = 3,
        edge_hidden_dim: int = 128,
    ) -> None:
        super().__init__(in_dim=in_dim, num_unique_ids=num_unique_ids, num_leds=num_leds)

        convs = []
        last_dim = in_dim
        for h in hidden_dims:
            convs.append(EdgeConvBlock(last_dim, edge_dim=edge_dim, out_dim=h))
            last_dim = h
        self.convs = nn.ModuleList(convs)

        self.feat_dim = sum(hidden_dims)

        # Edge affinity head: takes [h_i, h_j, |h_i - h_j|] → scalar logit
        edge_in_dim = 3 * self.feat_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, 1),
        )

    def forward_gnn(self, data: Data | Batch) -> torch.Tensor:
        batch = self._ensure_batch(data)
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        feats = []
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            feats.append(h)

        h_cat = torch.cat(feats, dim=-1)  # (N_total, feat_dim)
        return h_cat

    def forward(
        self,
        data: Data | Batch,
        edge_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch = self._ensure_batch(data)
        h = self.forward_gnn(batch)  # (N_total, feat_dim)

        row, col = batch.edge_index  # (E_total,)
        h_i = h[row]
        h_j = h[col]
        h_diff = torch.abs(h_i - h_j)

        edge_feat = torch.cat([h_i, h_j, h_diff], dim=-1)  # (E_total, 3*feat_dim)
        edge_logits = self.edge_mlp(edge_feat).squeeze(-1)  # (E_total,)

        loss = None
        if edge_labels is not None:
            # edge_labels: (E_total,), 0/1
            edge_labels = edge_labels.float()
            # TODO: you can add pos_weight here if you want to rebalance
            loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)

            # later: add size penalty term and combine into total loss

        return {
            "loss": loss,
            "edge_logits": edge_logits,
            "node_embeddings": h,
        }
          

