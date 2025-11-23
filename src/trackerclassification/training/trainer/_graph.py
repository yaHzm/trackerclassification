from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer as HFTrainer
from torch_geometric.data import Batch


class GraphHFTrainer(HFTrainer):
    """HF Trainer that:
       - pulls `Batch` from inputs["data"]
       - computes two CE losses (tracker/led)
       - returns both logits + labels for metrics
    """

    def compute_loss(self, model, inputs: Dict[str, Batch], return_outputs=False):
        batch = inputs["data"]  # torch_geometric.data.Batch
        tracker_logits, led_logits = model(batch)  # (N, Ct), (N, Cl)

        loss_t = F.cross_entropy(tracker_logits, batch.y_tracker)
        loss_l = F.cross_entropy(led_logits, batch.y_led)
        loss = loss_t + loss_l

        if return_outputs:
            outputs = {"tracker_logits": tracker_logits, "led_logits": led_logits}
            return loss, outputs
        return loss

    @torch.no_grad()
    def prediction_step(
        self, model, inputs: Dict[str, Any], prediction_loss_only: bool, ignore_keys=None
    ) -> Tuple[torch.Tensor | None, Any, Any]:
        model.eval()
        batch = inputs["data"]
        tracker_logits, led_logits = model(batch)
        loss = None
        if not prediction_loss_only:
            # pack logits and labels as tuples so compute_metrics can handle both heads
            logits = (tracker_logits.detach().cpu(), led_logits.detach().cpu())
            labels = (batch.y_tracker.detach().cpu(), batch.y_led.detach().cpu())
            return loss, logits, labels
        return loss, None, None