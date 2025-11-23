from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel
import numpy as np

from ._pose import TrackerPose
from ...utils.registry import RegistryMeta
from ...utils.typing import Matrix_Lx3_f, Vector_3_f


class TrackerBase(ABC, metaclass=RegistryMeta["TrackerBase"]):
    CodeClass: type[TrackerCodeBase]
    GeometryClass: type[TrackerGeometryBase]

    def __init__(self, code: TrackerCodeBase, pose: TrackerPose, geometry: TrackerGeometryBase) -> None:
        self._code = code
        self._pose = pose
        self._geometry = geometry

    @property
    def code(self) -> TrackerCodeBase:
        return self._code

    @property
    def pose(self) -> TrackerPose:
        return self._pose

    @property
    def geometry(self) -> TrackerGeometryBase:
        return self._geometry

    @classmethod
    def from_id(cls, id: int, rng: np.random.Generator) -> TrackerBase:
        """
        Create a Tracker instance from a given TrackerCode id by sampling a random pose and generating the corresponding geometry.

        Parameters:
            id (int): The unique tracker id within the range [0, num_unique_ids] with num_unique_ids defined by the tracker type
            rng (np.random.Generator): The random number generator used for sampling the pose

        Returns:
            TrackerBase: The corresponding tracker instance
        """
        code = cls.CodeClass.from_id(id)
        pose = TrackerPose.sample(rng)
        geometry = cls.GeometryClass.from_code(code)
        return cls(code=code, pose=pose, geometry=geometry)

    def get_leds_world_coords(self) -> Matrix_Lx3_f:
        """
        Get the 3D coordinates of the LEDs in world coordinates.

        Returns:
            Matrix_Lx3_f: An array of shape (L, 3) containing the 3D coordinates of the LEDs in world coordinates, with L being the number of LEDs on the tracker
        """
        R, t = self.pose.R, self.pose.t
        leds_tracker = self.geometry.as_array()
        leds_tracker_centered = leds_tracker - self.geometry.center    # shift so center is at (0,0,0)
        leds_world = (R @ leds_tracker_centered.T).T + t
        return leds_world

    @property
    def id(self) -> int:
        """
        Get the unique id of the tracker based on its code.

        Returns:
            int: The unique id of the tracker
        """
        return self.code.to_id()
    
    @classmethod
    def num_leds(cls) -> int:
        """
        Get the number of LEDs on the tracker.

        Returns:
            int: The number of LEDs on the tracker
        """
        return cls.GeometryClass.num_leds()
    

    @classmethod
    def num_unique_ids(cls) -> int:
        """
        Get the number of unique IDs for the tracker.

        Returns:
            int: The number of unique IDs for the tracker
        """
        return cls.CodeClass.num_unique_ids()


class TrackerCodeBase(ABC, BaseModel):
    @classmethod
    @abstractmethod
    def from_id(cls, id: int) -> TrackerCodeBase:
        """
        Create a TrackerCode from a unique id within the range [0, num_unique_ids] with num_unique_ids defined by the tracker type.

        Parameters:
            id (int): one-digit id in [0, num_unique_ids]

        Returns:
            TrackerCodeBase: The corresponding TrackerCode class
        
        Raises:
            ValueError: If id is not in [0, num_unique_ids]
        """
        pass
    
    @abstractmethod
    def to_id(self) -> int:
        """
        Get the unique id of the tracker (in [0, num_unique_ids]) based on its code, with num_unique_ids defined by the encodation.

        Returns:
            int: The unique id of the tracker
        """
        pass

    @classmethod
    @abstractmethod
    def num_unique_ids(cls) -> int:
        """Get the number of unique IDs for the tracker."""
        pass


class TrackerGeometryBase(ABC, BaseModel):
    @classmethod
    @abstractmethod
    def num_leds(cls) -> int:
        """
        Get the number of LEDs on the tracker.

        Returns:
            int: The number of LEDs on the tracker
        """
        pass

    @property
    @abstractmethod
    def center(self) -> Vector_3_f:
        """
        Get the center of the tracker geometry, calculated as the mean of all vertex positions.

        Returns:
            Vector_3_f: The 3D coordinates of the center of the tracker
        """
        pass

    @classmethod
    @abstractmethod
    def from_code(cls, code: TrackerCodeBase) -> TrackerGeometryBase:
        """
        Create a TrackerGeometry from a given TrackerCode.

        Parameters:
            code (TrackerCodeBase): The tracker code defining the arrangement of the LEDs on the sides

        Returns:
            TrackerGeometryBase: The corresponding geometry of the tracker
        """
        pass

    @abstractmethod
    def as_array(self) -> Matrix_Lx3_f:
        """
        Get the 3D coordinates of all LEDs in the tracker's local coordinate system.

        Returns:
            Matrix_Lx3_f: An array of shape (L, 3) containing the 3D coordinates of the LEDs in the tracker's local coordinate system, with L being the number of LEDs on the tracker
        """
        pass