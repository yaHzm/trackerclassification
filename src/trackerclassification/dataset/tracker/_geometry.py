from __future__ import annotations
from pydantic import ConfigDict
import numpy as np
import math
from typing_extensions import override

from ._base import TrackerGeometryBase
from ._code import V4TrackerCode
from ...config import (
    L, 
    TRAFO_BE, 
    RS_VAL_E,
)
from ...utils.typing import Vector_3_f, Matrix_7x3_f, check_dtypes


class V4TrackerGeometry(TrackerGeometryBase):
    """
    Representation of a tracker's LED geometry. 
    It defines the 3D coordinates of the LEDs in the tracker's local coordinate system. The local coordinate system is defined 
    such that the tracker lies in the XY plane with its normal pointing in +Z direction.

    The geometry consists of 7 LEDs in total:
        - 3 LEDs at the corners (indices 0, 1, 2)
        - 2 LEDs on side 0 (indices 3, 4)
        - 1 LED on side 1 (index 5)
        - 1 LED on side 2 (index 6)

    Attributes:
        vertex0 (Vector_3_f): 3D coordinates of LED at corner 0
        vertex1 (Vector_3_f): 3D coordinates of LED at corner 1
        vertex2 (Vector_3_f): 3D coordinates of LED at corner 2
        side0_a (Vector_3_f): 3D coordinates of LED on side 0 (first LED)
        side0_b (Vector_3_f): 3D coordinates of LED on side 0 (second LED)
        side1 (Vector_3_f): 3D coordinates of LED on side 1
        side2 (Vector_3_f): 3D coordinates of LED on side 2
    """
    vertex0: Vector_3_f
    vertex1: Vector_3_f
    vertex2: Vector_3_f
    side0_a: Vector_3_f
    side0_b: Vector_3_f
    side1: Vector_3_f
    side2: Vector_3_f

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @override
    @classmethod
    def num_leds(cls) -> int:
        return 7

    @override
    @property
    def center(self) -> Vector_3_f:
        """
        Get the center of the tracker geometry, calculated as the mean of all vertex positions.

        Returns:
            Vector_3_f: The 3D coordinates of the center of the tracker
        """
        verts = self.as_array()[:3]
        return np.mean(verts, axis=0)

    @staticmethod
    def _get_vertices() -> tuple[Vector_3_f, Vector_3_f, Vector_3_f]:
        """
        Get the 3D coordinates of the triangle vertices in the tracker's local coordinate system.

        Returns:
            tuple[Vector_3_f, Vector_3_f, Vector_3_f]: tuple of 3D coordinates of the triangle vertices (v0, v1, v2)
        """
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([L,   0.0, 0.0])
        v2 = np.array([0.5 * L, 0.5 * math.sqrt(3.0) * L, 0.0])
        return v0, v1, v2
    
    @check_dtypes
    @staticmethod
    def _get_sides(code: V4TrackerCode, v0: Vector_3_f, v1: Vector_3_f, v2: Vector_3_f) -> tuple[Vector_3_f, Vector_3_f, Vector_3_f, Vector_3_f]:
        """
        Get the 3D coordinates of the LEDs on the sides of the triangle based on the given tracker code.

        Parameters:
            code (V4TrackerCode): The tracker code defining the arrangement of the LEDs on the sides
            v0 (Vector3f): 3D coordinates of vertex 0
            v1 (Vector3f): 3D coordinates of vertex 1
            v2 (Vector3f): 3D coordinates of vertex 2

        Returns:
            tuple[Vector3f,Vector3f,Vector3f,Vector3f]: 3D coordinates of the LEDs on side 0 (two LEDs), side 1, and side 2
        """
        sides = [
            (v0, v1 - v0, v2 - v0),  
            (v1, v2 - v1, v0 - v1),
            (v2, v0 - v2, v1 - v2),
        ]
        led_poses = []
        for side_idx in range(3):
            vertex, edge_direction_1, edge_direction_2 = sides[side_idx]
            side_code = code[side_idx]
            for candidate_id in range(3):
                place = (side_idx == 0 and candidate_id != side_code) or (side_idx != 0 and candidate_id == side_code)
                if place:
                    rs_B = TRAFO_BE @ RS_VAL_E[:, candidate_id]
                    led_position = vertex + rs_B[0] * edge_direction_1 + rs_B[1] * edge_direction_2
                    led_poses.append(led_position)
        return led_poses[0], led_poses[1], led_poses[2], led_poses[3]

    @override
    @classmethod
    def from_code(cls, code: V4TrackerCode) -> V4TrackerGeometry:
        v0, v1, v2 = cls._get_vertices()
        s0_a, s0_b, s1, s2 = cls._get_sides(code, v0, v1, v2)
        return cls(
            vertex0=v0, vertex1=v1, vertex2=v2,
            side0_a=s0_a, side0_b=s0_b,
            side1=s1, side2=s2
        )

    @override
    def as_array(self) -> Matrix_7x3_f:
        """
        Convert the tracker geometry to a NumPy array.
        The LEDs occur in the following order:
            - vertex0 (index 0)
            - vertex1 (index 1)
            - vertex2 (index 2)
            - side0_a (index 3)
            - side0_b (index 4) 
            - side1 (index 5)
            - side2 (index 6)

        Returns:
            Matrix_7x3_f: A 2D array representation of the tracker geometry with shape (7, 3)
        """
        return np.stack([
            self.vertex0, self.vertex1, self.vertex2,
            self.side0_a, self.side0_b,
            self.side1, self.side2
        ], axis=0) 