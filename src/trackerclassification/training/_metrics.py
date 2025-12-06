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
    


from typing import Dict, Any

import numpy as np
from transformers import EvalPrediction


class AffinityMetrics:
    """
    Evaluation metrics for the LED grouping / affinity task.

    Interprets predictions as binary logits for edges:
      - prediction > threshold ⇒ same tracker (1)
      - prediction <= threshold ⇒ different tracker (0)

    Expects:
      - eval_pred.predictions: array-like of shape (E,) or (E, 1)
      - eval_pred.label_ids:   array-like of shape (E,) or (E, 1)
        with 0/1 ground truth edge labels.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def _extract_logits(self, predictions: Any) -> np.ndarray:
        """
        HF can pass predictions as:
          - a single ndarray
          - a tuple/list of ndarrays (e.g. (logits,) or (logits, extra_stuff, ...))

        We want a 1D array of logits of shape (E,).
        """
        # Case 1: tuple/list → pick first array-like
        if isinstance(predictions, (tuple, list)):
            # take the first element that is array-like
            for p in predictions:
                arr = np.asarray(p)
                if arr.ndim >= 1:
                    logits = arr
                    break
            else:
                raise ValueError(
                    f"Could not find 1D/2D logits in predictions. "
                    f"Got shapes: {[np.asarray(p).shape for p in predictions]}"
                )
        else:
            logits = np.asarray(predictions)

        if logits.ndim == 1:
            return logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            return logits[:, 0]

        raise ValueError(
            f"Predictions must be (E,) or (E, 1), got shape={logits.shape}"
        )

    def _extract_labels(self, label_ids: Any) -> np.ndarray:
        """
        HF can pass labels as:
          - a single 1D array: (E,)
          - a single 2D array: (E, 1) or (1, E)

        We want a 1D array of shape (E,) with 0/1 values.
        """
        arr = np.asarray(label_ids)

        if arr.ndim == 1:
            return arr.astype(np.int64)

        if arr.ndim == 2:
            # (E, 1)
            if arr.shape[1] == 1:
                return arr[:, 0].astype(np.int64)
            # (1, E)
            if arr.shape[0] == 1:
                return arr[0, :].astype(np.int64)

        raise ValueError(
            f"Label ids must be (E,) or (E, 1) / (1, E), got shape={arr.shape}"
        )

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        # 1) Extract logits + labels
        logits = self._extract_logits(eval_pred.predictions)   # (E,)
        labels = self._extract_labels(eval_pred.label_ids)     # (E,)

        # 2) Convert logits to probabilities and binary predictions
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        preds = (probs >= self.threshold).astype(np.int64)

        # 3) Basic counts
        assert preds.shape == labels.shape
        E = float(labels.size)

        tp = float(np.sum((preds == 1) & (labels == 1)))
        tn = float(np.sum((preds == 0) & (labels == 0)))
        fp = float(np.sum((preds == 1) & (labels == 0)))
        fn = float(np.sum((preds == 0) & (labels == 1)))

        # 4) Metrics with safe divisions
        accuracy = (tp + tn) / E if E > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        pos_rate = float(np.mean(labels == 1)) if E > 0 else 0.0

        return {
            "edge_accuracy": float(accuracy),
            "edge_precision": float(precision),
            "edge_recall": float(recall),
            "edge_f1": float(f1),
            "edge_pos_rate": float(pos_rate),
        }