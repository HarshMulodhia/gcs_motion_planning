"""PyVista Visualizer - COMPLETE with add_point_cloud() method"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PyVistaVisualizer:
    """PyVista-based scientific 3D visualization with full feature support"""

    def __init__(self, off_screen: bool = False):
        """Initialize PyVista visualizer.

        Parameters
        ----------
        off_screen : bool, optional
            If True, create an off-screen plotter suitable for headless/X-less
            environments (e.g., CI, servers). When False, use the default
            on-screen plotter behavior.
        """
        self.plotter = None
        self.connected = False
        self.off_screen = off_screen

        try:
            import pyvista as pv

            self.pv = pv
            self.connected = True
            logger.info("✓ PyVista available")
        except ImportError as e:
            logger.warning(f"PyVista not available: {e}")
            self.pv = None
            self.connected = False

    def add_trajectory(
        self,
        waypoints: np.ndarray,
        name: str = "trajectory",
        color: str = "red",
        line_width: float = 2.0,
    ):
        """Add trajectory to visualization.

        Args:
            waypoints: (N, 3) array of 3D points.
            name: Name for the trajectory.
            color: Color name (red, blue, green, etc.).
            line_width: Line width.
        """
        if not self.connected or self.pv is None:
            logger.warning("PyVista not available, skipping trajectory")
            return

        try:
            if self.plotter is None:
                # Respect off_screen flag for headless environments
                self.plotter = self.pv.Plotter(off_screen=self.off_screen)

            waypoints = np.asarray(waypoints, dtype=float)
            if waypoints.ndim != 2 or waypoints.shape[1] != 3:
                logger.warning(f"Invalid waypoints shape: {waypoints.shape}")
                return

            num_points = waypoints.shape[0]
            if num_points < 2:
                logger.warning(
                    f"Need at least 2 waypoints for trajectory '{name}', "
                    f"got {num_points}"
                )
                return

            # Create PolyData from waypoints
            poly = self.pv.PolyData(waypoints)

            # VTK-style connectivity for a single polyline:
            # [N, 0, 1, 2, ..., N-1] (size = N + 1)
            lines = np.hstack(
                [[num_points], np.arange(num_points, dtype=np.int64)]
            )
            poly.lines = lines

            # Add to plotter
            self.plotter.add_mesh(
                poly,
                color=color,
                line_width=line_width,
                label=name,
            )

            logger.debug(f"✓ Added trajectory: {name}")
        except Exception as e:
            logger.warning(f"Error adding trajectory: {e}")

    def add_sphere(
        self,
        center: np.ndarray,
        radius: float = 0.1,
        color: str = "blue",
        name: str = "sphere",
    ):
        """Add sphere to visualization.

        Args:
            center: Center position (3,).
            radius: Sphere radius.
            color: Color name.
            name: Sphere name.
        """
        if not self.connected or self.pv is None:
            logger.warning("PyVista not available, skipping sphere")
            return

        try:
            if self.plotter is None:
                self.plotter = self.pv.Plotter(off_screen=self.off_screen)

            center = np.asarray(center, dtype=float)
            sphere = self.pv.Sphere(radius=radius, center=center)
            self.plotter.add_mesh(sphere, color=color, label=name)
            logger.debug(f"✓ Added sphere: {name}")
        except Exception as e:
            logger.warning(f"Error adding sphere: {e}")

    def add_box(
        self,
        center: np.ndarray,
        size: np.ndarray,
        color: str = "orange",
        name: str = "box",
    ):
        """Add box to visualization.

        Args:
            center: Box center (3,).
            size: Box dimensions [dx, dy, dz].
            color: Color name.
            name: Box name.
        """
        if not self.connected or self.pv is None:
            logger.warning("PyVista not available, skipping box")
            return

        try:
            if self.plotter is None:
                self.plotter = self.pv.Plotter(off_screen=self.off_screen)

            center = np.asarray(center, dtype=float)
            size = np.asarray(size, dtype=float)

            # Create box with proper bounds
            box = self.pv.Box(
                bounds=[
                    center[0] - size[0] / 2,
                    center[0] + size[0] / 2,
                    center[1] - size[1] / 2,
                    center[1] + size[1] / 2,
                    center[2] - size[2] / 2,
                    center[2] + size[2] / 2,
                ]
            )
            self.plotter.add_mesh(
                box,
                color=color,
                opacity=0.7,
                label=name,
            )
            logger.debug(f"✓ Added box: {name}")
        except Exception as e:
            logger.warning(f"Error adding box: {e}")

    def add_point_cloud(
        self,
        points: np.ndarray,
        name: str = "cloud",
        color: str = "lightblue",
        point_size: float = 5.0,
    ):
        """Add point cloud to visualization - **FIXED METHOD**.

        Args:
            points: (N, 3) array of 3D points.
            name: Name for the point cloud.
            color: Color name (lightblue, red, green, etc.).
            point_size: Point size in pixels.
        """
        if not self.connected or self.pv is None:
            logger.warning("PyVista not available, skipping point cloud")
            return

        try:
            if self.plotter is None:
                self.plotter = self.pv.Plotter(off_screen=self.off_screen)

            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3:
                logger.warning(
                    f"Invalid points shape: {points.shape}, expected (N, 3)"
                )
                return

            # Create PolyData from points
            cloud = self.pv.PolyData(points)

            # Add to plotter with point rendering
            self.plotter.add_mesh(
                cloud,
                color=color,
                point_size=point_size,
                render_points_as_spheres=True,
                label=name,
            )

            logger.debug(
                f"✓ Added point cloud: {name} ({len(points)} points)"
            )
        except Exception as e:
            logger.warning(f"Error adding point cloud: {e}")

    def add_arrows(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        name: str = "arrows",
        color: str = "white",
        scale: float = 1.0,
    ):
        """Add arrow field to visualization.

        Args:
            origins: (N, 3) array of arrow origins.
            directions: (N, 3) array of arrow directions.
            name: Name for the arrows.
            color: Color name.
            scale: Arrow scale factor.
        """
        if not self.connected or self.pv is None:
            logger.warning("PyVista not available, skipping arrows")
            return

        try:
            if self.plotter is None:
                self.plotter = self.pv.Plotter(off_screen=self.off_screen)

            origins = np.asarray(origins, dtype=float)
            directions = np.asarray(directions, dtype=float)

            if origins.shape != directions.shape:
                logger.warning(
                    "Origins and directions must have same shape, "
                    f"got {origins.shape} and {directions.shape}"
                )
                return

            # Create vectors mesh
            vectors = self.pv.PolyData(origins)
            vectors["vectors"] = directions * scale

            # Add arrows
            self.plotter.add_mesh(
                vectors.arrows,
                color=color,
                label=name,
            )

            logger.debug(f"✓ Added arrow field: {name}")
        except Exception as e:
            logger.warning(f"Error adding arrows: {e}")

    def set_background(self, color: str = "white"):
        """Set background color.

        Args:
            color: Color name.
        """
        if not self.connected or self.plotter is None:
            return

        try:
            self.plotter.set_background(color)
        except Exception as e:
            logger.warning(f"Error setting background: {e}")

    def export(self, filename: str = "pyvista_visualization.png"):
        """Export visualization as PNG screenshot.

        Args:
            filename: Output filename.
        """
        if not self.connected or self.plotter is None:
            logger.warning("PyVista not available, skipping export")
            print("⚠️ PyVista export skipped (not available)")
            return

        try:
            self.plotter.screenshot(filename)
            logger.info(f"✓ Exported screenshot to {filename}")
            print(f"✓ Exported screenshot to {filename}")
        except Exception as e:
            logger.warning(f"Error exporting screenshot: {e}")
            print(f"⚠️ Could not export screenshot: {e}")

    def save_html(self, filename: str = "visualization.html"):
        """Save visualization as HTML file.

        Args:
            filename: Output filename.
        """
        if not self.connected or self.plotter is None:
            logger.warning("PyVista not available, skipping HTML export")
            print("⚠️ PyVista HTML export skipped (not available)")
            return

        try:
            # Try to export HTML with trame
            try:
                self.plotter.export_html(filename)
                logger.info(f"✓ HTML visualization saved to {filename}")
                print(f"✓ HTML visualization saved to {filename}")
            except ImportError:
                # trame not installed - gracefully skip
                logger.warning(
                    "trame_vtk not installed, HTML export unavailable"
                )
                print(
                    "⚠️ HTML export requires: "
                    "pip install 'pyvista[jupyter]'"
                )
                print(" Skipping HTML export (screenshot saved instead)")
        except Exception as e:
            logger.warning(f"Error saving HTML: {e}")
            print(f"⚠️ Could not save HTML: {e}")

    def show(self, title: str = "3D Visualization"):
        """Display the visualization.

        Args:
            title: Window title.
        """
        if not self.connected or self.plotter is None:
            logger.warning("PyVista not available, cannot show visualization")
            return

        try:
            self.plotter.show(title=title)
        except Exception as e:
            logger.warning(f"Error showing visualization: {e}")

    def is_connected(self) -> bool:
        """Check if PyVista is connected.

        Returns:
            True if connected, False otherwise.
        """
        return self.connected

    def reset_camera(self):
        """Reset camera to fit all objects."""
        if self.plotter is None:
            return

        try:
            self.plotter.reset_camera()
            logger.debug("✓ Camera reset")
        except Exception as e:
            logger.warning(f"Error resetting camera: {e}")

    def set_view(
        self,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        reset: bool = True,
    ):
        """Set camera azimuth/elevation for a nicer view.

        Works with code like: ``pv_vis.set_view(azimuth=45, elevation=30)``.

        Args:
            azimuth: Azimuth angle in degrees, or None.
            elevation: Elevation angle in degrees, or None.
            reset: If True, call ``reset_camera()`` after changing the view.
        """
        if not self.connected or self.plotter is None:
            return

        try:
            cam = self.plotter.camera

            if azimuth is not None:
                cam.azimuth = float(azimuth)
            if elevation is not None:
                cam.elevation = float(elevation)

            if reset:
                self.plotter.reset_camera()
        except Exception as e:
            logger.warning(f"Error setting camera view: {e}")

    def __del__(self):
        """Cleanup."""
        try:
            if self.plotter is not None:
                self.plotter.close()
        except Exception:
            # Suppress all exceptions in destructor
            pass
