from __future__ import annotations
from pydantic import BaseModel, field_validator, ConfigDict
import numpy as np
import math

from ...config import (
    X_MIN, X_MAX,
    Y_MIN, Y_MAX,
    Z_MIN, Z_MAX,
    MAX_ROT_ANGLE_DEG
)
from ...utils import GeometryUtils
from ...utils.typing import Vector_3_f, Matrix_3x3_f


class TrackerPose(BaseModel):
    """
    Rigid transformation (R, t) that places the tracker in 3D space.
    Trackers can be sampled randomly within the predefined sample space. The translation t is the position of the tracker center in world coordinates.
    The rotation R is defined such that the local -Z axis of the tracker points towards the camera (located at the origin) with a small random tilt.

    Attributes:
        R (Matrix_3x3_f): Rotation matrix of shape (3, 3)
        t (Vector_3_f): Translation vector of shape (3,)
    """
    R: Matrix_3x3_f
    t: Vector_3_f
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("R")
    @classmethod
    def _check_R(cls, v: Matrix_3x3_f) -> Matrix_3x3_f:
        """
        Validation to ensure that R is a valid rotation matrix of shape (3, 3).

        Parameters:
            v (Matrix_3x3_f): The rotation matrix to validate

        Returns:
            Matrix_3x3_f: The original rotation matrix if validation passes

        Raises:
            ValueError: If R is not of shape (3, 3)
        """
        v = np.asarray(v, dtype=float)
        if v.shape != (3, 3):
            raise ValueError(f"R must be shape (3,3), got {v.shape}")
        return v

    @field_validator("t")
    @classmethod
    def _check_t(cls, v: Vector_3_f) -> Vector_3_f:
        """
        Validation to ensure that t is a valid translation vector of shape (3,).

        Parameters:
            v (Vector_3_f): The translation vector to validate
        
        Returns:
            Vector_3_f: The original translation vector if validation passes

        Raises:
            ValueError: If t is not of shape (3,)
        """
        v = np.asarray(v, dtype=float)
        if v.shape != (3,):
            raise ValueError(f"t must be shape (3,), got {v.shape}")
        return v
    
    @classmethod
    def sample(cls, rng: np.random.Generator) -> TrackerPose:
        """
        Sample a random tracker pose within the predefined bounds. The tracker will face towards the camera (at the origin) with a small random tilt.

        Parameters:
            rng (np.random.Generator): Random number generator

        Returns:
            TrackerPose: A randomly sampled tracker pose
        """
        # sample tracker center randomly within sample space
        t = np.array([
            rng.uniform(X_MIN, X_MAX),
            rng.uniform(Y_MIN, Y_MAX),
            rng.uniform(Z_MIN, Z_MAX),
        ], dtype=float)

        # get direction vector from camera (at origin) to tracker center
        dir_vec = GeometryUtils.normalize(t) 

        # rotate local -Z to point along dir_vec (face the camera)
        R_face = GeometryUtils.rotate_negZ_to(dir_vec)

        # add a random tilt (0..max_angle) around an axis ⟂ dir_vec
        max_angle_rad = math.radians(MAX_ROT_ANGLE_DEG)
        R_tilt = GeometryUtils.random_tilt_about(dir_vec, max_angle_rad, rng)

        # compose rotation
        R = R_tilt @ R_face
        return cls(R=R, t=t)