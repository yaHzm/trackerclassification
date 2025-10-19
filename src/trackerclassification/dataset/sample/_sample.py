from __future__ import annotations
import numpy as np

from ..tracker import Tracker, TrackerPose, TrackerGeometry, TrackerCode, CameraIntrinsics
    

class Sample:
    def __init__(self, size: int, rng: np.random.Generator) -> None:
        assert size >= 1, "sample size must be >= 1"
        assert 1 <= size <= 27, "sample size must be between 1 and 27"

        self._size = int(size)
        self._rng = rng
        self._trackers = self._instantiate_trackers()

    def _get_tracker_codes(self) -> list[TrackerCode]:
        chosen_ids = self._rng.choice(27, size=self._size, replace=False)
        tracker_codes: list[TrackerCode] = [TrackerCode.from_id(i) for i in chosen_ids]
        return tracker_codes
    
    def _instantiate_trackers(self) -> list[Tracker]:
        codes = self._get_tracker_codes()
        trackers = [
            Tracker(
                code=code,
                pose=TrackerPose.sample(self._rng),
                geometry=TrackerGeometry.from_code(code),
            )
            for code in codes
        ]
        return trackers

    def get_world_coords(self) -> np.ndarray:
        leds_world_coords = np.stack([tracker.get_leds_world_coords() for tracker in self._trackers], axis=0)
        return leds_world_coords
    
    def get_trackers(self) -> list[Tracker]:
        return self._trackers
    
    def get_data(self) -> dict:
        leds_projected, valid_mask = CameraIntrinsics.project_sample(self.get_world_coords()) # (T,7,2), (T,7)

        T, N, D = leds_projected.shape
        assert N == 7 and D == 2, "Expected shape (T,7,2)."

        tracker_ids = np.asarray([tr.id for tr in self._trackers], dtype=np.int64)
        ids_grid = np.repeat(tracker_ids[:, None], N, axis=1)
        led_idx_grid = np.broadcast_to(np.arange(N, dtype=np.int64), (T, N))

        X = leds_projected[valid_mask]                                # (N,2)
        y_ids = ids_grid[valid_mask]                                  # (N,)
        y_idx = led_idx_grid[valid_mask]                              # (N,)
        y = np.stack([y_ids, y_idx], axis=1)                    # (N,2)

        X = X.astype(np.float32, copy=False)
        y = y.astype(np.int64,   copy=False)
        
        return X, y