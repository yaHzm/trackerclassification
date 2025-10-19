from __future__ import annotations
import math
import numpy as np


class GeometryUtils:
    @staticmethod
    def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > eps else np.zeros_like(v)

    @staticmethod
    def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
        a = GeometryUtils.normalize(axis)
        if np.allclose(a, 0.0):
            return np.eye(3)
        x, y, z = a
        K = np.array([[0, -z, y],
                      [z, 0, -x],
                      [-y, x, 0]], dtype=float)
        c = math.cos(angle); s = math.sin(angle)
        return np.eye(3) + s * K + (1.0 - c) * (K @ K)

    @staticmethod
    def rotate_negZ_to(dir_vec: np.ndarray) -> np.ndarray:
        u = np.array([0.0, 0.0, -1.0], dtype=float)
        d = GeometryUtils.normalize(dir_vec)
        axis = np.cross(u, d)
        # dot(u, d) = -d_z  (since u = [0,0,-1])
        cosang = float(-d[2])
        cosang = max(-1.0, min(1.0, cosang))
        angle = math.acos(cosang)
        return GeometryUtils.rodrigues(axis, angle)

    @staticmethod
    def random_tilt_about(dir_vec: np.ndarray, max_angle_rad: float, rng) -> np.ndarray:
        d = GeometryUtils.normalize(dir_vec)
        help_vec = np.array([rng.random(), rng.random(), 0.0], dtype=float)
        axis = np.cross(help_vec, d)
        if np.linalg.norm(axis) < 1e-12:
            # fallback axes if nearly parallel
            axis = np.cross(np.array([1.0, 0.0, 0.0]), d)
            if np.linalg.norm(axis) < 1e-12:
                axis = np.cross(np.array([0.0, 1.0, 0.0]), d)
        angle = rng.uniform(0.0, max_angle_rad)
        return GeometryUtils.rodrigues(axis, angle)
