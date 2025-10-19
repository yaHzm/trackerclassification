from __future__ import annotations
from typing import Any, Dict
import numpy as np
from torch.utils.data import Dataset
import torch

from .sample import Sample


class TrackingDataset(Dataset):
    def __init__(self, size: int, num_trackers: int, seed: int = 0) -> None:
        assert size >= 1, "dataset size must be >= 1"
        assert 1 <= num_trackers <= 27, "num_trackers must be between 1 and 27"

        self._size = int(size)
        self._num_trackers = int(num_trackers)
        self._seed = int(seed)

    def __len__(self) -> int:
        return self._size
    
    def _has_occlusion(self, sample: Sample) -> bool:
        """Return True if any LED from one tracker is covered by the triangle
        (first three vertices) of another tracker when viewed from the origin.
        Mirrors the MATLAB logic using a line from the camera (0,0,0) to the LED.
        leds_world: (T, 7, 3)
        """
        leds_coords = sample.get_world_coords()
        for t0 in range(self._num_trackers):
            q0 = leds_coords[t0, 0]
            q1 = leds_coords[t0, 1]
            q2 = leds_coords[t0, 2]
            e1 = q1 - q0
            e2 = q2 - q0
            for t1 in range(self._num_trackers):
                if t1 == t0:
                    continue
                for l in range(7):
                    cur_led = leds_coords[t1, l]
                    # Solve [e1 e2 -cur_led] * [r s t] = -q0
                    A = np.column_stack((e1, e2, -cur_led))
                    b = -q0
                    # Use least squares for robustness (singular/near-singular cases)
                    try:
                        res, *_ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    r, s, t = res
                    # Inside triangle and intersection in front of LED?
                    if (r >= 0.0) and (s >= 0.0) and (r + s <= 1.0) and (t >= 0.0) and (t <= 1.0):
                        return True
        return False
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(self._seed + idx)
        max_tries = 10000  
        for _ in range(max_tries):
            sample = Sample(size=self._num_trackers, rng=rng)
            if self._has_occlusion(sample):
                continue
            X, y = sample.get_data()
            X = torch.from_numpy(X)           # float32
            y = torch.from_numpy(y) 
            return {
                "x": X,                        # (N, 2) float32
                "y": y,                        # (N, 2) int64
            }

        # Fallback if we somehow never pass occlusion (rare)
        sample = Sample(size=self._num_trackers, rng=rng)
        X, y = sample.get_data()
        X = torch.from_numpy(X)           # float32
        y = torch.from_numpy(y) 
        return {
            "x": X,                        # (N, 2) float32
            "y": y,    
            "warning": "max_tries reached; occlusion filter not satisfied on this sample",
        }