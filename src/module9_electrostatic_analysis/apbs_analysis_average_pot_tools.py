"""
Tools for computing averages of APBS analysis results.
They were statimehotd in the module apbs_analysis_average_pot_plots.py.
"""
import os
import typing

import numpy as np
import pandas as pd

from common import logger
from common import xvg_to_dataframe
from common.colors_text import TextColor as bcolors


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
def process_layer(center_xyz: tuple[int, int, int],
                  grid_spacing: list[float],
                  grid_points: list[int],
                  data_arr: np.ndarray,
                  bulk_averaging: bool,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """process the layer
    The potential from the surface until the end of the diffuse layer
    """
    max_radius: float = calculate_max_radius(
        center_xyz, grid_spacing)
    # Create the distance grid
    grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray] = \
        create_distance_grid(grid_points)
    # Calculate the distances from the center of the box
    distances: np.ndarray = compute_distance(
            grid_spacing, grid_xyz, center_xyz)
    radii, radial_average = calculate_radial_average(
            data_arr,
            distances,
            grid_spacing,
            max_radius,
            grid_xyz[2],
            interface_low_index=center_xyz[2],
            interface_high_index=center_xyz[2],
            lower_index_bulk=0,
            bulk_averaging=bulk_averaging
            )
    return radii, np.asanyarray(radial_average)


def calculate_max_radius(center_xyz: tuple[float, float, float],
                         grid_spacing: list[float]
                         ) -> float:
    """Calculate the maximum radius for the radial average"""
    return min(center_xyz[:2]) * min(grid_spacing)


def create_distance_grid(grid_points: list[int],
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the distance grid"""
    x_space: np.ndarray = \
        np.linspace(0, grid_points[0] - 1, grid_points[0])
    y_space: np.ndarray = \
        np.linspace(0, grid_points[1] - 1, grid_points[1])
    z_space: np.ndarray = \
        np.linspace(0, grid_points[2] - 1, grid_points[2])

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


def calculate_radial_average(data_arr: np.ndarray,
                             distances: np.ndarray,
                             grid_spacing: list[float],
                             max_radius: float,
                             grid_z: np.ndarray,
                             interface_low_index: int,
                             interface_high_index: int,
                             lower_index_bulk: int,
                             bulk_averaging: bool,
                             ) -> tuple[np.ndarray, list[float]]:
    """Calculate the radial average of the potential"""
    # pylint: disable=too-many-arguments
    radii: np.ndarray = np.arange(0, max_radius, grid_spacing[0])
    radial_average: list[float] = []

    for radius in radii:
        mask = create_mask(distances,
                           radius,
                           grid_spacing,
                           grid_z,
                           interface_low_index,
                           interface_high_index,
                           lower_index_bulk,
                           bulk_averaging
                           )
        if np.sum(mask) > 0:
            avg_potential = np.mean(data_arr[mask])
            radial_average.append(avg_potential)
        else:
            radial_average.append(0)

    return radii, radial_average


def create_mask(distances: np.ndarray,
                radius: float,
                grid_spacing: list[float],
                grid_z: np.ndarray,
                interface_low_index: int,
                interface_high_index: int,
                low_index_bulk: int,
                bulk_averaging: bool,
                ) -> np.ndarray:
    """Create a mask for the radial average"""
    # pylint: disable=too-many-arguments
    shell_thickness: float = grid_spacing[0]
    shell_condition: np.ndarray = (distances >= radius) & \
                                  (distances < radius + shell_thickness)

    if bulk_averaging:
        z_condition: np.ndarray = create_mask_bulk(
            grid_z, interface_low_index, low_index_bulk)
    else:
        z_condition = create_mask_interface(
            grid_z, interface_low_index, interface_high_index)

    return shell_condition & z_condition


def create_mask_bulk(grid_z: np.ndarray,
                     interface_low_index: int,
                     low_index_bulk: int,
                     ) -> np.ndarray:
    """Create a mask for the radial average from the bulk"""
    z_condition: np.ndarray = (grid_z <= interface_low_index) & \
                              (grid_z >= low_index_bulk)
    return z_condition


def create_mask_interface(grid_z: np.ndarray,
                          interface_low_index: int,
                          interface_high_index: int
                          ) -> np.ndarray:
    """Create a mask for the radial average from the interface"""
    z_condition: np.ndarray = (grid_z >= interface_low_index) & \
                              (grid_z <= interface_high_index)
    return z_condition


# Other tools
def get_arr_from_dict(data_dict: dict[typing.Any,
                                      int | float | np.int64 | np.float64],
                      key_i: str,
                      ) -> np.ndarray:
    """Get the array from the dictionary"""
    if key_i not in data_dict:
        raise KeyError(f'The key {key_i} is not in the dictionary.')
    return np.array(list(data_dict[key_i].values()))


def get_average_z_palce_nanoparticle(log: logger.logging.Logger,
                                     fname: str = 'COR_COM.xvg'
                                     ) -> float:
    """
    Get the average z place of the nanoparticle from the uncentered
    trajectory
    """
    if not os.path.exists(fname):
        log.info(msg := f'\tFile `{fname}` does not exist!, return `0.0`\n')
        print(f'{bcolors.WARNING}{msg}{bcolors.ENDC}')
        return 0.0
    df: pd.DataFrame = xvg_to_dataframe.XvgParser(fname, log).xvg_df
    try:
        return df['COR_Z'].mean()
    except KeyError:
        log.error(
            msg := '\tThe column `COR_Z` does not exist!, return `0.0`\n')
        print(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
        return 0.0


def compute_z_offset_nanoparticles(box_size_z: float,
                                   log: logger.logging.Logger,
                                   ) -> float:
    """
    Compute the offset of the nanoparticles after centering the
    nanoparticle in the box
    """
    box_center: float = box_size_z / 2 / 10.0  # Convert to nm
    np_com_z: float = get_average_z_palce_nanoparticle(log)
    return box_center - np_com_z
