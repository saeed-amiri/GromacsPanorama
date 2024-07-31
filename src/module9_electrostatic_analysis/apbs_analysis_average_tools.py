"""
Tools for computing averages of APBS analysis results.
They were statimehotd in the module apbs_analysis_average_pot_plots.py.
"""

import numpy as np


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
