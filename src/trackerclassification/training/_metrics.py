from __future__ import annotations
from typing import Dict, Any

import numpy as np
from transformers import EvalPrediction


class TrackingMetrics:
    """
    Evaluation metrics for the tracker/LED classification task.
    """

    def __init__(self, num_trackers: int, num_leds: int) -> None:
        self._num_trackers = num_trackers
        self._num_leds = num_leds

    def _extract_logits(self, predictions: Any) -> np.ndarray:
        """
        HF can pass predictions as:
          - a single ndarray
          - a tuple/list of ndarrays (e.g. (logits, tracker_logits, led_logits))
        We want the main logits array of shape (N, num_trackers + num_leds).
        """
        # Case 1: tuple/list → pick first 2D array inside
        if isinstance(predictions, (tuple, list)):
            shapes = []
            for p in predictions:
                arr = np.asarray(p)
                shapes.append(arr.shape)
                if arr.ndim == 2:
                    return arr
            raise ValueError(
                f"Could not find 2D logits in predictions tuple/list. "
                f"Shapes seen: {shapes}"
            )

        # Case 2: single array-like
        arr = np.asarray(predictions)
        if arr.ndim == 2:
            return arr

        raise ValueError(
            f"Predictions is not 2D: shape={arr.shape}, type={type(predictions)}"
        )

    def _extract_labels(self, label_ids: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        HF can pass labels as:
          - a single (N, 2) array
          - a single (2, N) array
          - a tuple/list of two (N,) arrays
        We want:
          tracker_true: (N,)
          led_true:     (N,)
        """
        # Case 1: tuple/list of two arrays
        if isinstance(label_ids, (tuple, list)):
            if len(label_ids) != 2:
                raise ValueError(
                    f"Expected 2 label arrays (tracker, led), got {len(label_ids)}"
                )
            t = np.asarray(label_ids[0]).reshape(-1)
            l = np.asarray(label_ids[1]).reshape(-1)
            return t, l

        # Case 2: single array-like
        arr = np.asarray(label_ids)

        if arr.ndim == 2:
            # (N, 2) → columns are labels
            if arr.shape[1] == 2:
                tracker_true = arr[:, 0]
                led_true = arr[:, 1]
                return tracker_true, led_true
            # (2, N) → rows are labels
            if arr.shape[0] == 2:
                tracker_true = arr[0, :]
                led_true = arr[1, :]
                return tracker_true, led_true

        raise ValueError(
            f"Label ids have unexpected shape={arr.shape}, type={type(label_ids)}"
        )

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        logits = self._extract_logits(eval_pred.predictions)   # (N, num_trackers + num_leds)
        tracker_true, led_true = self._extract_labels(eval_pred.label_ids)  # both (N,)

        tracker_logits = logits[:, :self._num_trackers]
        led_logits = logits[:, self._num_trackers:]

        tracker_pred = tracker_logits.argmax(axis=-1)
        led_pred = led_logits.argmax(axis=-1)

        tracker_acc = (tracker_pred == tracker_true).mean().item()
        led_acc = (led_pred == led_true).mean().item()
        joint_acc = ((tracker_pred == tracker_true) & (led_pred == led_true)).mean().item()

        return {
            "tracker_accuracy": tracker_acc,
            "led_accuracy": led_acc,
            "joint_accuracy": joint_acc,
        }