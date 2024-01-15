"""finding the interface and plotting some of them in a few frames.

It should also save the the interface location for further analsyis.

The class identifies the water molecules that are located at the
surface of a simulated box, typically containing a mixture of water,
oil, and possibly nanoparticles. It usesa mesh grid approach to
analyze the water molecules' positions and determines which ones are
at the highest z-coordinate within each mesh cell.
"""

import typing
import random
import multiprocessing
from dataclasses import dataclass
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator

from common import logger
from common import cpuconfig
from common import plot_tools
from common import file_writer
from common.colors_text import TextColor as bcolors
from module3_analysis_aqua.com_plotter import ComPlotter, OutputConfig


MeshInfo: tuple = \
    namedtuple('MeshInfo', ['x_mesh', 'y_mesh', 'mesh_size', 'z_threshold'])


@dataclass
class PlotConfig:
    """set the output configurations"""
    fout_suffix: str = 'surface.png'
    nr_fout: int = 100  # Numbers pics in case the list is empty
    indices: typing.Optional[list[int]] = None


@dataclass
class ParamConfig:
    """sett the constant parameters"""
    oil_top_ratio: float = 2/3  # Where form top for sure should be oil
    mesh_nr: float = 100.  # Number of meshes in each directions
    z_threshold: float = 120.


@dataclass
class ComputationConfig(PlotConfig, ParamConfig):
    """set all the configs"""


class GetSurface:
    """find the surface of the water"""

    info_msg: str = 'Message from GetSurface:\n'  # Meesage in methods to log

    compute_config: "ComputationConfig"
    surface_waters: dict[int, np.ndarray]  # Water at the surface, include np
    locz_arr: np.ndarray  # Average location of the surface (z component)

    def __init__(self,
                 water_arr: np.ndarray,
                 box_dims: dict[str, float],
                 log: logger.logging.Logger,
                 compute_config: "ComputationConfig" = ComputationConfig()
                 ) -> None:
        self.compute_config = compute_config
        self.locz_arr = self.get_water_surface(water_arr, box_dims, log)
        self.plot_water_surfaces(box_dims)
        self._write_msg(log)

    def get_water_surface(self,
                          water_arr: np.ndarray,
                          box_dims: dict[str, float],
                          log: logger.logging.Logger
                          ) -> np.ndarray:
        """
        mesh the box and find resides with highest z value in them
        """
        # To save a snapshot to see the system
        x_mesh: np.ndarray  # Mesh grid in x and y
        y_mesh: np.ndarray  # Mesh grid in x and y
        output_config = OutputConfig(
            to_png=True,
            out_suffix='com_output.png',
            to_xyz=True,
            xyz_suffix='com_output.xyz')
        ComPlotter(com_arr=water_arr[:-2],
                   index=2,
                   log=log,
                   output_config=output_config)
        # self.compute_config.z_threshold = \
        # self.get_interface_z_threshold(box_dims)
        x_mesh, y_mesh, self.mesh_size = self._get_xy_grid(box_dims)
        surface_indices: dict[int, list[np.int64]] = \
            self._get_surface_topology(water_arr[:-2], x_mesh, y_mesh, log)
        self.surface_waters: dict[int, np.ndarray] = \
            self.get_xyz_arr(water_arr[:-2], surface_indices)
        locz_df: pd.DataFrame = self.get_interface_z()
        file_writer.write_xvg(locz_df, log, fname := 'contact.xvg')
        self.info_msg += (f'\tThe dataframe saved to `{fname}`\n')
        return locz_df['interface_z'].to_numpy()

    def get_interface_z_threshold(self,
                                  box_dims: dict[str, float]
                                  ) -> float:
        """find the threshold of water highest point"""
        z_threshold: float = \
            box_dims['z_hi'] * self.compute_config.oil_top_ratio
        self.info_msg += \
            ('\tThe oil top ratio was set to '
             f'`{self.compute_config.oil_top_ratio:.3f}`\n'
             f'\tThe z threshold is set to `{z_threshold:.3f}`\n')
        return z_threshold

    def _get_xy_grid(self,
                     box_dims: dict[str, float]
                     ) -> tuple[np.ndarray, np.ndarray, float]:
        """return the mesh grid for the box"""
        mesh_size: float = \
            (box_dims['x_hi']-box_dims['x_lo'])/self.compute_config.mesh_nr
        self.info_msg += \
            f'\tThe number of meshes is `{self.compute_config.mesh_nr**2}`\n'
        x_mesh: np.ndarray  # Mesh grid in x and y
        y_mesh: np.ndarray  # Mesh grid in x and y
        x_mesh, y_mesh = np.meshgrid(
            np.arange(
                box_dims['x_lo'], box_dims['x_hi'] + mesh_size, mesh_size),
            np.arange(
                box_dims['y_lo'], box_dims['y_hi'] + mesh_size, mesh_size))
        return x_mesh, y_mesh, mesh_size

    def _get_surface_topology(self,
                              water_arr: np.ndarray,
                              x_mesh: np.ndarray,
                              y_mesh: np.ndarray,
                              log: logger.logging.Logger
                              ) -> dict[int, list[np.int64]]:
        """get max water in each time frame"""
        cpu_info = cpuconfig.ConfigCpuNr(log)
        n_cores: int = min(cpu_info.cores_nr, water_arr.shape[0])

        results: list[list[np.int64]]
        max_indices: dict[int, list[np.int64]] = {}

        mesh_info: "MeshInfo" = MeshInfo(x_mesh,
                                         y_mesh,
                                         self.mesh_size,
                                         self.compute_config.z_threshold)

        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(
                self._process_single_frame,
                [(i_frame, frame, mesh_info)
                 for i_frame, frame in enumerate(water_arr)])
        for i_frame, result in enumerate(results):
            max_indices[i_frame] = result

        return max_indices

    def _process_single_frame(self,
                              i_frame: int,  # index of the frame
                              frame: np.ndarray,  # One frame of water com traj
                              mesh_info: "MeshInfo",
                              ) -> list[np.int64]:
        """Process a single frame to find max water indices
        *keep the `i_frame` for debuging
        """
        # pylint: disable=unused-argument

        max_z_indices: list[np.int64] = []
        min_z_threshold: float = mesh_info.z_threshold / 2
        xyz_i = frame.reshape(-1, 3)
        for (i, j), _ in np.ndenumerate(mesh_info.x_mesh):
            # Define the boundaries of the current mesh element
            x_min_mesh = mesh_info.x_mesh[i, j]
            x_max_mesh = mesh_info.x_mesh[i, j] + mesh_info.mesh_size
            y_min_mesh = mesh_info.y_mesh[i, j]
            y_max_mesh = mesh_info.y_mesh[i, j] + mesh_info.mesh_size

            # Select atoms within the current mesh element based on XY
            ind_in_mesh = np.where((xyz_i[:, 0] >= x_min_mesh) &
                                   (xyz_i[:, 0] < x_max_mesh) &
                                   (xyz_i[:, 1] >= y_min_mesh) &
                                   (xyz_i[:, 1] < y_max_mesh) &
                                   (xyz_i[:, 2] < mesh_info.z_threshold) &
                                   (xyz_i[:, 2] > min_z_threshold))
            if len(ind_in_mesh[0]) > 0:
                max_z = np.argmax(frame[2::3][ind_in_mesh])
                max_z_indices.append(ind_in_mesh[0][max_z])
        return max_z_indices

    def get_xyz_arr(self,
                    water_arr: np.ndarray,  # All the water residues
                    surface_indices: dict[int, list[np.int64]]
                    ) -> dict[int, np.ndarray]:
        """
        return the surface residues for each time frame as a np.ndarry
        """
        surface_waters: dict[int, np.ndarray] = {}
        for i_frame, indices in surface_indices.items():
            frame = water_arr[i_frame]
            i_arr: np.ndarray = np.zeros((len(indices), 3))
            for i, index in enumerate(indices):
                residue_ind = int(index*3)
                i_arr[i] = frame[residue_ind:residue_ind+3]
            surface_waters[i_frame] = i_arr
            del i_arr
        return surface_waters

    def get_interface_z(self) -> pd.DataFrame:
        """Creates a dataframe of the z component of the interface."""
        loc_z: dict[int, np.float64] = {}
        for i_frame, water_arr in self.surface_waters.items():
            loc_z[i_frame] = np.mean(water_arr[:, 2])
        loc_z_df = pd.DataFrame(
            list(loc_z.items()), columns=['i_frame', 'interface_z'])
        return loc_z_df

    def plot_water_surfaces(self,
                            box_dims: dict[str, float]
                            ) -> None:
        """plot water surface in random selected frames"""
        selected_frames: dict[int, np.ndarray] = \
            self.get_selected_frames(self.compute_config.indices,
                                     self.compute_config.nr_fout)

        self.plot_surface(selected_frames,
                          box_dims,
                          fout_suffix=self.compute_config.fout_suffix)

    def get_selected_frames(self,
                            indices: typing.Optional[list[int]],
                            nr_fout: int
                            ) -> dict[int, np.ndarray]:
        """return the numbers of the wanted frames data"""
        if indices is None:
            indices = random.sample(list(self.surface_waters.keys()), nr_fout)
        self.info_msg += f'\tThe selected indices are:\n\t\t{indices}\n'
        return {key: self.surface_waters[key] for key in indices}

    def plot_surface(self,
                     selected_frames: dict[int, np.ndarray],
                     box_dims: dict[str, float],
                     fout_suffix: str
                     ) -> None:
        """plot the surface"""
        self.info_msg += f'\tThe suffix of the files is: `{fout_suffix}`\n'
        for frame, value in selected_frames.items():
            fig_i, ax_i = \
                plot_tools.mk_canvas((0, 200), height_ratio=5**0.5-1)
            scatter = ax_i.scatter(value[:, 0], value[:, 1], c=value[:, 2],
                                   s=15, label=f'frame: {frame}')
            # 'left', 'bottom', 'width', 'height'
            cbar_ax = fig_i.add_axes([.80, 0.15, 0.25, 0.7])
            cbar = fig_i.colorbar(scatter, cax=cbar_ax)
            desired_num_ticks = 5
            cbar.ax.yaxis.set_major_locator(MaxNLocator(desired_num_ticks))

            cbar.set_label('Z-Coordinate [A]', rotation=90)
            ax_i.set_xlabel('X_Coordinate [A]')
            ax_i.set_ylabel('Y_Coordinate [A]')

            ax_i.set_xlim(box_dims['x_lo'] - 7, box_dims['x_hi'] + 7)
            ax_i.set_ylim(box_dims['y_lo'] - 7, box_dims['y_hi'] + 7)
            plt.gca().set_aspect('equal')
            plot_tools.save_close_fig(
                fig_i, ax_i, fname=f'{frame}_{fout_suffix}', legend=False)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{GetSurface.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
