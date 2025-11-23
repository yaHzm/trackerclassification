from __future__ import annotations
import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from ..tracker import TrackerBase, TrackerCodeBase, CameraIntrinsics
from ...utils.typing import Tensor_TxLx3_f, Matrix_Nx2_f
    

class Sample:
    """
    A sample of tracker data consisting of multiple trackers, each with multiple LEDs.
    """
    def __init__(self, num_trackers: int, TrackerClass: type[TrackerBase], rng: np.random.Generator) -> None:
        """
        Initialize a sample of tracker data consisting of num_trackers trackers, each of type TrackerType.

        Args:
            num_trackers (int): The number of trackers in the sample.
            TrackerClass (type[TrackerBase]): The type of the tracker to instantiate.
            rng (np.random.Generator): The random number generator used for sampling.

        Raises:
            ValueError: If num_trackers is less than 1 or exceeds the maximum unique IDs for the tracker type.
        """
        self._num_trackers = int(num_trackers)
        self._TrackerClass = TrackerClass
        self._rng = rng
        self._trackers = self._instantiate_trackers()

        if not num_trackers >= 1:
            LOGGER.error("Number of trackers must be >= 1, got %d", num_trackers)
            raise ValueError("Number of trackers must be >= 1")
        if not num_trackers <= self._TrackerClass.num_unique_ids():
            LOGGER.error("Number of trackers %d exceeds max unique ids %d for tracker type %s", num_trackers, self._TrackerClass.num_unique_ids(), self._TrackerClass)
            raise ValueError(f"Number of trackers {num_trackers} exceeds max unique ids {self._TrackerClass.num_unique_ids()} for tracker type {self._TrackerClass}")

    def _get_tracker_ids(self) -> list[int]:
        """
        Randomly select unique tracker ids for the sample.

        Returns:
            list[int]: A list of unique tracker ids
        """
        chosen_ids = self._rng.choice(
            self._TrackerClass.num_unique_ids(), 
            size=self._num_trackers, 
            replace=False)
        return chosen_ids
    
    def _instantiate_trackers(self) -> list[TrackerBase]:
        """
        Instantiate the tracker objects for the sample.

        Returns:
            list[TrackerBase]: A list of tracker instances with sampled poses and geometries and unique codes
        """
        tracker_ids = self._get_tracker_ids()
        trackers = [
            self._TrackerClass.from_id(id, self._rng)
            for id in tracker_ids
        ]
        return trackers

    def get_world_coords(self) -> Tensor_TxLx3_f:
        """
        Get the 3D world coordinates of all LEDs from all trackers in the sample.

        Returns:
            Tensor_TxLx3_f: A tensor of shape (T, L, 3) containing the 3D coordinates of the LEDs in world coordinates for T trackers with L LEDs each
        """
        leds_world_coords = np.stack([tracker.get_leds_world_coords() for tracker in self._trackers], axis=0)
        return leds_world_coords
    
    def get_trackers(self) -> list[TrackerBase]:
        """
        Get the list of tracker instances in the sample.

        Returns:
            list[TrackerBase]: The list of tracker instances
        """
        return self._trackers
    
    def get_data(self) -> tuple[Matrix_Nx2_f, Matrix_Nx2_f]:
        """
        Get the sample data as a dictionary containing the projected 2D LED coordinates and their corresponding labels.

        Returns:
            tuple[Matrix_Nx2_f, Matrix_Nx2_f]: A tuple containing:
                - 'X': An array of shape (N, 2) containing the 2D pixel coordinates of the N valid LEDs
                - 'y': An array of shape (N, 2) containing the labels for each valid LED, where each label is a tuple (tracker_id, led_index)
        """
        leds_projected, valid_mask = CameraIntrinsics.project_sample(self.get_world_coords(), L=self._TrackerClass.num_leds()) # (T,L,2), (T,L)
        T, L, D = leds_projected.shape

        if T != self._num_trackers or L != self._TrackerClass.num_leds() or D != 2:
            LOGGER.error("Expected leds_projected shape (%d,%d,2), got %s", self._num_trackers, self._TrackerClass.num_leds(), leds_projected.shape)
            raise ValueError(f"Expected leds_projected shape ({self._num_trackers},{self._TrackerClass.num_leds()},2), got {leds_projected.shape}")

        tracker_ids = np.asarray([tr.id for tr in self._trackers], dtype=np.int64)
        ids_grid = np.repeat(tracker_ids[:, None], L, axis=1)
        led_idx_grid = np.broadcast_to(np.arange(L, dtype=np.int64), (T, L))

        X = leds_projected[valid_mask]                                # (N,2)
        y_ids = ids_grid[valid_mask]                                  # (N,)
        y_idx = led_idx_grid[valid_mask]                              # (N,)
        y = np.stack([y_ids, y_idx], axis=1)                          # (N,2)

        X = X.astype(np.float32, copy=False)
        y = y.astype(np.int64,   copy=False)
        
        return X, y