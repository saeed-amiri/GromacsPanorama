"""
Tools for computing averages of APBS analysis results.
They were statimehotd in the module apbs_analysis_average_pot_plots.py.
"""

import numpy as np


# tools for method: analyse_potential
def calculate_center(grid_points: list[int]
                     ) -> tuple[int, int, int]:
    """Calculate the center of the box in grid units"""
    center_x: int = grid_points[0] // 2
    center_y: int = grid_points[1] // 2
    center_z: int = grid_points[2] // 2
    return center_x, center_y, center_z


# tools for method: cut_average_from_surface
def calculate_grid_sphere_intersect_radius(radius: float,
                                           center_z: int,
                                           sphere_grid_range: np.ndarray,
                                           z_grid_spacing: float,
                                           ) -> np.ndarray:
    """Get radius of the intersection between the sphere and the
    grid plane"""
    radius_index: np.ndarray = np.zeros(len(sphere_grid_range))
    for i, z_index in enumerate(sphere_grid_range):
        height: float = \
            np.abs(center_z - z_index) * z_grid_spacing
        if height <= radius:
            radius_index[i] = np.sqrt(radius**2 - height**2)
    return radius_index


def find_inidices_of_surface(interset_radius: np.ndarray,
                             radii_list: list[np.ndarray]
                             ) -> np.ndarray:
    """Find the indices of the surface by finding the index of the
    closest radius to the intersection radius"""
    cut_indices: np.ndarray = np.zeros(len(interset_radius))
    for i, radius in enumerate(interset_radius):
        cut_indices[i] = np.argmin(np.abs(radii_list[i] - radius)) + 5
    return cut_indices


def find_indices_of_diffuse_layer(radii_list: list[np.ndarray],
                                  threshold: float) -> np.ndarray:
    """Find the indices of the diffuse layer by finding the index
    of the radial average which is less than the threshold"""
    cut_indices: np.ndarray = np.ones(len(radii_list)) * -1
    for i, radii in enumerate(radii_list):
        cut_indices[i] = np.argmin(radii - threshold <= 0)
    return cut_indices


# tools for method: process_layer
def calculate_max_radius(center_xyz: tuple[float, float, float],
                         grid_spacing: list[float]
                         ) -> float:
    """Calculate the maximum radius for the radial average"""
    return min(center_xyz[:2]) * min(grid_spacing)


def create_distance_grid(grid_points: list[int],
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the distance grid"""
    x_space = np.linspace(0, grid_points[0] - 1, grid_points[0])
    y_space = np.linspace(0, grid_points[1] - 1, grid_points[1])
    z_space = np.linspace(0, grid_points[2] - 1, grid_points[2])

    grid_x, grid_y, grid_z = \
        np.meshgrid(x_space, y_space, z_space, indexing='ij')
    return grid_x, grid_y, grid_z


def compute_distance(grid_spacing: list[float],
                     grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
                     center_xyz: tuple[float, float, float],
                     ) -> np.ndarray:
    """Calculate the distances from the center of the box"""
    return np.sqrt((grid_xyz[0] - center_xyz[0])**2 +
                   (grid_xyz[1] - center_xyz[1])**2 +
                   (grid_xyz[2] - center_xyz[2])**2) * grid_spacing[0]
