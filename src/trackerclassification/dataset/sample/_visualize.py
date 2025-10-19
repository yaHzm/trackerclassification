from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

from ...config import (
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    Z_MAX,
    WIDTH,
    HEIGHT,
)
from ..tracker import CameraIntrinsics
from ._sample import Sample


class SampleVisualizer:
    def __init__(self):
        self._fig = plt.figure(figsize=(8, 6))
        self._ax3d = self._fig.add_subplot(1,1,1, projection='3d')
        self._initialize_sample()

    def _initialize_sample(self):
        rng = np.random.default_rng()
        self._sample = Sample(3, rng)
        self._cmap = plt.cm.get_cmap("tab20", max(3, 1))

    def _visualize_axes(self):
        self._ax3d.set_xlim(X_MIN, X_MAX)
        self._ax3d.set_ylim(Y_MIN, Y_MAX)
        self._ax3d.set_zlim(0, Z_MAX)
        self._ax3d.set_xlabel('X')
        self._ax3d.set_ylabel('Y')
        self._ax3d.set_zlabel('Z')
        self._ax3d.set_title("3D scene → projection → 2D")
    
    def _visualize_sample_centers(self) -> None:
        for i, tr in enumerate(self._sample.get_trackers()):
            color = self._cmap(i)
            center = tr.pose.t
            self._ax3d.scatter(
                [center[0]], [center[1]], [center[2]],
                s=80, depthshade=False, c=[color]
            )
        camera = np.array([0.0, 0.0, 0.0], dtype=float)
        self._ax3d.scatter(
            [camera[0]], [camera[1]], [camera[2]],
            marker="^", s=30
        )

    def _visualize_tracker_leds(self) -> None:
        for i, tr in enumerate(self._sample.get_trackers()):
            color = self._cmap(i)
            leds_world = tr.get_leds_world_coords()
            self._ax3d.scatter(
                leds_world[:, 0], leds_world[:, 1], leds_world[:, 2],
                s=20,
                marker="o",    
                facecolors='none',
                edgecolors=[color],    
                linewidths=1.8,
                depthshade=False
            )
    
    def _visualize_triangle_edges(self):
        for i, tr in enumerate(self._sample.get_trackers()):
            color = self._cmap(i)
            leds_world = tr.get_leds_world_coords()   
            v = leds_world[:3, :]
            loop = np.vstack([v, v[0:1]])            
            self._ax3d.plot(loop[:,0], loop[:,1], loop[:,2],
                            linestyle='-', linewidth=1, color=color, alpha=0.9)
    
    def _visualize_projection_plane(self, z_plane: float, face_alpha: float = 0.18, with_rays: bool = True) -> None:
        assert z_plane > 0.0, "z_plane must be > 0."

        hfov = math.radians(CameraIntrinsics.hfov_deg())
        vfov = math.radians(CameraIntrinsics.vfov_deg())
        half_w = z_plane * math.tan(hfov * 0.5)
        half_h = z_plane * math.tan(vfov * 0.5)

        X = np.array([[-half_w, +half_w],
                    [-half_w, +half_w]], dtype=float)
        Y = np.array([[-half_h, -half_h],
                    [+half_h, +half_h]], dtype=float)
        Z = np.full_like(X, z_plane, dtype=float)

        self._ax3d.plot_surface(X, Y, Z, linewidth=0, antialiased=False,
                                color="gold", alpha=face_alpha)

        corners = np.array([
            [-half_w, -half_h, z_plane],
            [ +half_w, -half_h, z_plane],
            [ +half_w, +half_h, z_plane],
            [ -half_w, +half_h, z_plane],
        ], dtype=float)

        if with_rays:
            extend_factor = 300 / z_plane
            self._visualize_camera_rays_to_plane(corners, extend_factor)
    
    def _visualize_camera_rays_to_plane(self, plane_corners: np.ndarray, extend_factor: float):
        cam = np.array([0.0, 0.0, 0.0])
        for C in plane_corners:
            seg = np.vstack([cam, C])
            self._ax3d.plot(seg[:,0], seg[:,1], seg[:,2],
                            linestyle='-', linewidth=1.2, color='k', alpha=0.7)
            C2 = extend_factor * C
            seg2 = np.vstack([C, C2])
            self._ax3d.plot(seg2[:,0], seg2[:,1], seg2[:,2],
                            linestyle='--', linewidth=1.0, color='k', alpha=0.7)
            
    def _project_to_plane_z(self, points_world: np.ndarray, z_plane: float) -> np.ndarray:
        Z = np.maximum(points_world[:, 2], 1e-9)
        s = (z_plane / Z)[:, None]
        out = np.empty_like(points_world)
        out[:, :2] = points_world[:, :2] * s
        out[:, 2]  = z_plane
        return out
            
    def _visualize_projection_paths(self, z_plane: float, frac: float,
                                    s_start: float = 28.0, s_end: float = 10.0):
        frac = float(np.clip(frac, 0.0, 1.0))
        # current marker size (shrinking)
        s_now = ((1.0 - frac) * s_start + frac * s_end) / 2

        for i, tr in enumerate(self._sample.get_trackers()):
            color = self._cmap(i)

            leds_world = tr.get_leds_world_coords()                        # (7,3)
            leds_on_plane = self._project_to_plane_z(leds_world, z_plane)  # (7,3)
            current = leds_world * (1.0 - frac) + leds_on_plane * frac     # (7,3)

            # --- AHEAD PATH: draw from CURRENT --> FINAL (so the path is in front of the point)
            for k in range(leds_world.shape[0]):
                seg = np.vstack([current[k], leds_on_plane[k]])
                # draw the ahead segment; draw AFTER scatter for clarity, or use higher zorder
                self._ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                                linestyle='-', linewidth=1.1, color=color, alpha=0.75, zorder=5)

            # moving LED markers (shrinking)
            self._ax3d.scatter(current[:, 0], current[:, 1], current[:, 2],
                            s=s_now,
                            marker='o',
                            c=[color],
                            edgecolors=[color],
                            linewidths=1.6,
                            depthshade=False,
                            zorder=6)
            
    def _ease_in_out_cubic(self, t: float) -> float:
        t = np.clip(t, 0.0, 1.0)
        return 4*t*t*t if t < 0.5 else 1 - pow(-2*t + 2, 3)/2
    
    def _animate_view_rotation(self, frac: float,
                            elev_start: float = 30.0, azim_start: float = -60.0,
                            elev_end: float = -90.0,  azim_end: float = 0.0):
        f = self._ease_in_out_cubic(frac)
        elev = (1.0 - f) * elev_start + f * elev_end
        azim = (1.0 - f) * azim_start + f * azim_end
        self._ax3d.view_init(elev=elev, azim=azim)
            
    def _animate_zoom_to_plane(self, z_plane: float, frac: float, pad_ratio: float = 0.08):
        # easing for smoother zoom
        f = self._ease_in_out_cubic(frac)

        # global bounds (start) → plane bounds (end)
        x0, x1 = X_MIN, X_MAX
        y0, y1 = Y_MIN, Y_MAX
        z0, z1 = 0.0, Z_MAX

        # plane half extents from intrinsics
        hfov = math.radians(CameraIntrinsics.hfov_deg())
        vfov = math.radians(CameraIntrinsics.vfov_deg())
        half_w = z_plane * math.tan(hfov * 0.5)
        half_h = z_plane * math.tan(vfov * 0.5)

        # pad around plane rectangle for nice framing
        px = half_w * pad_ratio
        py = half_h * pad_ratio
        # z window: thin slab around plane
        pz = max(1e-3, z_plane * 0.02)
        x0_t, x1_t = -half_w - px,  half_w + px
        y0_t, y1_t = -half_h - py,  half_h + py
        z0_t, z1_t = z_plane - pz,  z_plane + pz

        # interpolate limits
        def lerp(a, b): return (1.0 - f)*a + f*b
        self._ax3d.set_xlim(lerp(x0, x0_t), lerp(x1, x1_t))
        self._ax3d.set_ylim(lerp(y0, y0_t), lerp(y1, y1_t))
        self._ax3d.set_zlim(lerp(z0, z0_t), lerp(z1, z1_t))

    def _visualize_plane_2d_pixels(self):
        self._fig.clf()
        ax2d = self._fig.add_subplot(1,1,1)

        # project all LEDs (T,7,2), valid mask (T,7)
        leds_world = self._sample.get_world_coords()              # (T,7,3)
        pixels, valid = CameraIntrinsics.project_sample(leds_world)  # (T,7,2), (T,7)

        pixels_rot = np.empty_like(pixels)
        pixels_rot[..., 0] = HEIGHT - pixels[..., 1]   # u' = HEIGHT - v
        pixels_rot[..., 1] = pixels[..., 0]  

        # draw image frame
        ax2d.add_patch(plt.Rectangle((0, 0), WIDTH, HEIGHT, fill=False, linewidth=1.5))
        # plot per tracker with color
        for i, tr in enumerate(self._sample.get_trackers()):
            color = self._cmap(i)
            pts = pixels_rot[i][valid[i]]
            if pts.size:
                ax2d.scatter(pts[:,0], pts[:,1], s=18, c=[color], edgecolors=[color], linewidths=1.4)
        ax2d.set_xlim(0, WIDTH)
        ax2d.set_ylim(HEIGHT, 0)  
        ax2d.set_aspect('equal')
        ax2d.set_xlabel("v (px)")
        ax2d.set_ylabel("u (px)")
        ax2d.set_title("Projection on image plane (pixels)")
        self._fig.canvas.draw_idle()

    def _update_frame(self, frame: int):
        # timings
        t0 = 10   # centers
        t1 = 20   # LEDs
        t2 = 30   # triangle edges ON
        t3 = 40   # plane ON + camera rays
        t4 = 55   # edges & rays OFF
        t5 = 60   # start projection animation (with shrinking dots + ahead paths)
        t6 = 90   # finish projection
        t7a = 94  # start rotating camera to top-down
        t7b = 104 # finish rotation
        t8  = 120 # finish zoom, then switch to 2D

        z_plane = 25.0

        # before 2D switch, we still render 3D
        if frame < t8:
            self._ax3d.clear()
            self._visualize_axes()

            # centers
            if frame >= t0:
                self._visualize_sample_centers()
            # LEDs (static)
            if frame >= t1:
                self._visualize_tracker_leds()
            # triangle edges in [t2, t4)
            if t2 <= frame < t4:
                self._visualize_triangle_edges()
            # plane + rays in [t3, t4)
            if frame >= t3:
                self._visualize_projection_plane(z_plane=25.0, with_rays=True)
            # projection animation [t5, t6]
            if frame >= t5:
                frac = (frame - t5) / max(1, (t6 - t5))
                frac = float(np.clip(frac, 0.0, 1.0))
                self._visualize_projection_paths(z_plane=z_plane, frac=frac,
                                                s_start=28.0, s_end=10.0)

           # --- rotation phase (t7a → t7b) ---
            if t7a <= frame < t7b:
                f_rot = (frame - t7a) / max(1, (t7b - t7a))
                self._animate_view_rotation(frac=f_rot,
                                            elev_start=30.0, azim_start=-60.0,
                                            elev_end=90.0,  azim_end=0.0)

            # --- zoom phase (t7b → t8) ---
            if t7b <= frame < t8:
                # keep the final facing-plane orientation fixed while zooming
                self._ax3d.view_init(elev=90.0, azim=0.0)
                f_zoom = (frame - t7b) / max(1, (t8 - t7b))
                self._animate_zoom_to_plane(z_plane=z_plane, frac=f_zoom)

            return []

        # switch to 2D pixel view
        if frame >= t8:
            self._visualize_plane_2d_pixels()
            return []
    
    def _visualize(self):
        ani = animation.FuncAnimation(self._fig, self._update_frame, frames=130, interval=130, blit=False)
        plt.tight_layout()
        plt.show()

    @classmethod
    def main(cls):
        cls()._visualize()