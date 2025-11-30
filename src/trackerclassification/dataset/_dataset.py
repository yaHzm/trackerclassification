from __future__ import annotations
from typing import Any, Dict
import numpy as np
from torch.utils.data import Dataset
import torch

import logging
LOGGER = logging.getLogger(__name__)

from .sample import Sample, SampleVisualizer
from .tracker import TrackerBase


class TrackingDataset(Dataset):
    """
    A PyTorch Dataset class for generating samples of tracker data without occlusions.

    Attributes:
        size (int): The number of samples in the dataset.
        num_trackers (int): The number of trackers in each sample.
        TrackerClass (type[TrackerBase]): The tracker class to use in each sample.
        seed (int): The seed for the random number generator to ensure reproducibility.
    """
    def __init__(self, size: int, num_trackers: int, TrackerClass: type[TrackerBase], seed: int = 0) -> None:
        LOGGER.info("Initializing TrackingDataset with size=%d, num_trackers=%d, TrackerClass=%s, seed=%d", size, num_trackers, TrackerClass, seed)
        if not size >= 1:
            LOGGER.error("Dataset size must be >= 1, got %d", size)
            raise ValueError("Dataset size must be >= 1")
        self._size = int(size)
        self._num_trackers = int(num_trackers)
        self._TrackerClass = TrackerClass
        self._seed = int(seed)

    def __len__(self) -> int:
        return self._size
    
    def _has_occlusion(self, sample: Sample) -> bool:
        """
        Check if there are any occlusions between trackers in the sample, meaning if any LED from one tracker is occluded by the triangle formed by another tracker,
        i.e. it's vertices, when viewed from the camera origin.

        Args:
            sample (Sample): The sample to check for occlusions.

        Returns:
            bool: True if there is at least one occlusion, False otherwise.
        """
        leds_coords = sample.get_world_coords()
        if leds_coords.shape != (self._num_trackers, self._TrackerClass.num_leds(), 3):
            LOGGER.error("Expected leds_coords shape (%d, %d, 3), got %s", self._num_trackers, self._TrackerClass.num_leds(), leds_coords.shape)
            raise ValueError(f"Expected leds_coords shape ({self._num_trackers}, {self._TrackerClass.num_leds()}, 3), got {leds_coords.shape}")
        for t0 in range(self._num_trackers):
            q0 = leds_coords[t0, 0]
            q1 = leds_coords[t0, 1]
            q2 = leds_coords[t0, 2]
            e1 = q1 - q0
            e2 = q2 - q0
            for t1 in range(self._num_trackers):
                if t1 == t0:
                    continue
                for l in range(self._TrackerClass.num_leds()):
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
        """
        Generate one synthetic tracker sample without occlusion. The output dictionary looks as follows, with N = num_trackers * leds_per_tracker  : 
                {    
                    "x": Tensor of shape (N, 2), float32    
                        LED coordinates for this sample       

                    "y": Tensor of shape (N, 2), int64
                        Integer labels per LED:
                            y[:, 0] = tracker ID   (0 .. num_trackers-1)
                            y[:, 1] = LED index    (0 .. leds_per_tracker-1)
                }
        
        Args:
            idx (int): The index of the sample to generate.

        Returns:
            Dict[str, Any]: A dictionary containing the sample data. If occlusion filtering fails (rare), an additional
                            "warning" field is included.

        """
        rng = np.random.default_rng(self._seed + idx)
        max_tries = 10000  
        for _ in range(max_tries):
            sample = Sample(num_trackers=self._num_trackers, TrackerClass=self._TrackerClass, rng=rng)
            if self._has_occlusion(sample):
                continue
            X, y = sample.get_data()
            X = torch.from_numpy(X)           
            y = torch.from_numpy(y) 

            X = X - X.mean(dim=0, keepdim=True)
            scale = X.norm(dim=1).max()
            if scale > 0:
                X = X / scale

            return {
                "x": X,                        # (N, 2) float32
                "y": y,                        # (N, 2) int64
            }

        # Fallback if we somehow never pass occlusion (rare)
        sample = Sample(num_trackers=self._num_trackers, TrackerClass=self._TrackerClass, rng=rng)
        X, y = sample.get_data()
        X = torch.from_numpy(X)
        y = torch.from_numpy(y) 

        X = X - X.mean(dim=0, keepdim=True)
        scale = X.norm(dim=1).max()
        if scale > 0:
            X = X / scale

        return {
            "x": X,                        # (N, 2) float32
            "y": y,                        # (N, 2) int64
            "warning": "max_tries reached; occlusion filter not satisfied on this sample",
        }
    
    @classmethod
    def visualize(cls) -> None:
        """
        Visualize a random sample from the dataset using the SampleVisualizer.
        """
        SampleVisualizer.main()