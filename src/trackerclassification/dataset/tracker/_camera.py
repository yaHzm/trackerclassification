from __future__ import annotations
import math
import numpy as np

from ...config import WIDTH, HEIGHT, HFOV_DEG
from ...utils.typing import Matrix_Tx7_b, Tensor_Tx7x3_f, Tensor_Tx7x2_f, Matrix_Nx3_f, Matrix_Nx2_f, check_dtypes


class CameraIntrinsics:
    """
    Representation of Camera intrinsics for a pinhole camera model to project 3D points in camera coordinates to 2D pixel coordinates on
    a projected image plane.
    """
    @classmethod
    def aspect(cls) -> float:
        """
        Get the aspect ratio (width / height) of the camera.
        
        Returns:
            float: The aspect ratio of the camera
        """
        return WIDTH / HEIGHT

    @classmethod
    def hfov_deg(cls) -> float:
        """
        Get the horizontal field of view (in degrees) of the camera.

        Returns:
            float: The horizontal field of view in degrees
        """
        return HFOV_DEG

    @classmethod
    def vfov_deg(cls) -> float:
        """
        Get the vertical field of view (in degrees) of the camera, calculated from the horizontal field of view and the aspect ratio with the formula:
        tan(v/2) = tan(h/2) * (H/W)
        
        Returns:
            float: The vertical field of view in degrees
        """
        h = math.radians(HFOV_DEG)
        vfov = 2.0 * math.atan(math.tan(h * 0.5) * (HEIGHT / WIDTH))
        return math.degrees(vfov)
    
    @classmethod
    def fx(cls) -> float:
        """
        Get the focal length in pixels along the x-axis, calculated from the horizontal field of view with the formula:
        fx = (W/2) / tan(hfov/2)

        Returns:
            float: The focal length in pixels along the x-axis
        """
        h = math.radians(HFOV_DEG)
        return (WIDTH * 0.5) / math.tan(h * 0.5)
    
    @classmethod
    def fy(cls) -> float:
        """
        Get the focal length in pixels along the y-axis, calculated from the vertical field of view with the formula:
        fy = (H/2) / tan(vfov/2)

        Returns:
            float: The focal length in pixels along the y-axis
        """
        v = math.radians(cls.vfov_deg())
        return (HEIGHT * 0.5) / math.tan(v * 0.5)

    @classmethod
    def cx(cls) -> float:
        """
        Get the x-coordinate of the principal point (optical center) in pixels.

        Returns:
            float: The x-coordinate of the principal point in pixels
        """
        return WIDTH * 0.5
    
    @classmethod
    def cy(cls) -> float:
        """
        Get the y-coordinate of the principal point (optical center) in pixels.

        Returns:
            float: The y-coordinate of the principal point in pixels
        """
        return HEIGHT * 0.5

   
    @classmethod
    @check_dtypes
    def _project(cls, flattened_leds_world: Matrix_Nx3_f) -> tuple[Matrix_Nx2_f, np.ndarray]:
        """
        Project a set of 3D points in world coordinates to 2D pixel coordinates on the projection plane.
        Also, checks if the points are valid (in front of the camera and within the image frame).   

        Parameters:
            flattened_leds_world (Matrix_Nx3_f): An array of shape (N, 3) containing the 3D coordinates of the points in world coordinates

        Returns:
            Tuple[Matrix_Nx2_f, np.ndarray]: A tuple containing:
                - An array of shape (N, 2) containing the 2D pixel coordinates of the points on the projection plane
                - A boolean array of shape (N,) indicating whether each point is valid (in front of the camera and within the image frame)

        Raises:
            AssertionError: If flattened_leds_world is not of shape (N, 3)
        """
        Z = flattened_leds_world[:, 2]

        valid = Z > 1e-6

        # Perspective divide
        x = flattened_leds_world[:, 0] / np.maximum(Z, 1e-6)
        y = flattened_leds_world[:, 1] / np.maximum(Z, 1e-6)

        # Pixel coords (x right, y down)
        u = cls.fx() * x + cls.cx()
        v = cls.fy() * (-y) + cls.cy()  # negate to flip to image y-down

        pixels = np.stack([u, v], axis=1)

        # In-frame test
        in_frame = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] <= WIDTH - 1) &
            (pixels[:, 1] >= 0) & (pixels[:, 1] <= HEIGHT - 1)
        )
        return pixels, (valid & in_frame)

    @classmethod
    @check_dtypes
    def project_sample(cls, leds_world: Tensor_Tx7x3_f) -> tuple[Tensor_Tx7x2_f, Matrix_Tx7_b]:
        """
        Project a sample of trackers from world coordinates to pixel coordinates on the projection plane. 

        Parameters:
            leds_world (Tensor_Tx7x3_f): An array of shape (T, 7, 3) containing the 3D coordinates of the LEDs in world coordinates for T trackers

        Returns:
            Tuple[Tensor_Tx7x2_f, Matrix_Tx7_b]: A tuple containing:
                - An array of shape (T, 7, 2) containing the 2D pixel coordinates of the LEDs on the projection plane
                - A boolean array of shape (T, 7) indicating whether each LED is valid (in front of the camera and within the image frame)

        Raises:
            AssertionError: If leds_world is not of shape (T, 7, 3)
        """
        T = leds_world.shape[0]

        flattened_leds_world = leds_world.reshape(T * 7, 3)
        leds_projected, valid_mask = cls._project(flattened_leds_world)

        return leds_projected.reshape(T, 7, 2), valid_mask.reshape(T, 7)