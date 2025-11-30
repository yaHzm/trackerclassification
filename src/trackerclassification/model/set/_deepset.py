from __future__ import annotations
from typing import Optional, Dict

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

from ._base import SetModelBase


class DeepSetModel(SetModelBase):
    """
    DeepSets-style per-LED classifier:
      - input: Data/Batch with .x (N, 2) and .batch
      - output: tracker_logits, led_logits (N, C)
    """

    def __init__(
        self,
        in_dim: int,
        num_unique_ids: int,
        num_leds: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(in_dim=in_dim, num_unique_ids=num_unique_ids, num_leds=num_leds)

        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # φ: per-point embedding
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ρ: combine local + global context
        self.rho = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        heads = self.heads(hidden_dim)
        self.tracker_head: nn.Module = heads["tracker"]
        self.led_head: nn.Module = heads["led"]

    def forward(
        self,
        data: Data | Batch,
        tracker_labels: Optional[torch.Tensor] = None,
        led_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        batch = self._ensure_batch(data)
        x = batch.x                                  # (N_total, in_dim)
        batch_vec = getattr(batch, "batch", None)

        if batch_vec is None:
            batch_vec = x.new_zeros(x.size(0), dtype=torch.long)

        # Per-point embedding
        h = self.phi(x)                              # (N, H)
        h = self.dropout(h)

        # Global context per graph
        g = global_mean_pool(h, batch_vec)           # (B, H)
        g_expanded = g[batch_vec]                    # (N, H)

        # Combine local + global
        z = torch.cat([h, g_expanded], dim=-1)       # (N, 2H)
        z = self.rho(z)                              # (N, H)
        z = self.dropout(z)

        tracker_logits = self.tracker_head(z)        # (N, num_unique_ids)
        led_logits = self.led_head(z)                # (N, num_leds)

        return self._build_output(
            tracker_logits=tracker_logits,
            led_logits=led_logits,
            tracker_labels=tracker_labels,
            led_labels=led_labels,
        )