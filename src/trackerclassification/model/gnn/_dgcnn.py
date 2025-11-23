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
    """
    DGCNN-style GNN for per-LED classification:
      - input: Data/Batch with .x (node features, e.g. 2D coords) and .batch
      - output: (tracker_logits, led_logits), each [N_nodes, num_classes]
    """

    def __init__(
        self,
        in_dim: int,
        num_trackers: int,
        num_leds: int,
        k: int = 20,
        hidden_dims: Sequence[int] = (64, 128, 256),
    ) -> None:
        super().__init__(in_dim=in_dim, num_trackers=num_trackers, num_leds=num_leds)

        self.k = k

        # EdgeConv stack (we’ll concat intermediate features à la DGCNN)
        convs = []
        last_dim = in_dim
        for h in hidden_dims:
            convs.append(EdgeConvBlock(last_dim, h, k=k))
            last_dim = h
        self.convs = nn.ModuleList(convs)

        # Final feature dim is concat of all EdgeConv outputs
        self.feat_dim = sum(hidden_dims)

        # Shared heads from base class
        heads = self.heads(self.feat_dim)
        self.tracker_head: nn.Module = heads["tracker"]
        self.led_head: nn.Module = heads["led"]

    def forward_gnn(self, data: Data | Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self._ensure_batch(data)

        assert hasattr(batch, "x"), f"Batch has no x attribute, keys: {batch.keys if hasattr(batch, 'keys') else dir(batch)}"
        assert batch.x is not None, "batch.x is None in forward_gnn"
        assert hasattr(batch, "batch"), "Batch has no batch vector"

        x = batch.x
        batch_vec = batch.batch

        feats = []
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, batch_vec)
            feats.append(h)

        h_cat = torch.cat(feats, dim=-1)
        tracker_logits = self.tracker_head(h_cat)
        led_logits = self.led_head(h_cat)
        return tracker_logits, led_logits

    # HF-compatible forward
    def forward(
        self,
        data: Data | Batch,
        tracker_labels: Optional[torch.Tensor] = None,
        led_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        tracker_logits, led_logits = self.forward_gnn(data)

        loss = None
        if tracker_labels is not None and led_labels is not None:
            loss_tracker = F.cross_entropy(tracker_logits, tracker_labels)
            loss_led = F.cross_entropy(led_logits, led_labels)
            loss = loss_tracker + loss_led

        # `logits` is what HFTrainer hands to compute_metrics
        # Here I just concatenate; you can also return a tuple and handle it.
        logits = torch.cat([tracker_logits, led_logits], dim=-1)

        return {
            "loss": loss,
            "logits": logits,
            "tracker_logits": tracker_logits,
            "led_logits": led_logits,
        }