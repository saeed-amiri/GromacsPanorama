"""
Analysis the water phase
main porpose is getting the interface of water with minimum possible
reduction of surface from nanoparticle.
"""

import multiprocessing
import numpy as np

from common import logger
from common import cpuconfig
from common.colors_text import TextColor as bcolors
from module3_analysis_com.com_file_parser import GetCom
from module3_analysis_com.com_plotter import ComPlotter


class GetSurface:
    """find the surface of the water"""

    info_msg: str = 'Message from GetSurface:\n'  # Meesage in methods to log
    oil_top_ratio: float = 2/3  # Where form top for sure should be oil
    mesh_nr: float = 100.  # Number of meshes in each directions
    z_treshhold: float
    mesh_size: float

    def __init__(self,
                 water_arr: np.ndarray,
                 box_dims: dict[str, float],
                 log: logger.logging.Logger
                 ) -> None:
        self.get_water_surface(water_arr, box_dims, log)
        self._write_msg(log)

    def get_water_surface(self,
                          water_arr: np.ndarray,
                          box_dims: dict[str, float],
                          log: logger.logging.Logger
                          ) -> None:
        """
        mesh the box and find resides with highest z value in them
        """
        # To save a snapshot to see the system
        x_mesh: np.ndarray  # Mesh grid in x and y
        y_mesh: np.ndarray  # Mesh grid in x and y
        ComPlotter(com_arr=water_arr[:-2],
                   index=10,
                   log=log,
                   to_png=True,
                   to_xyz=True)
        self.z_treshhold = self.get_interface_z_treshhold(box_dims)
        x_mesh, y_mesh, self.mesh_size = self._get_xy_grid(box_dims)
        self._get_surface_topology(water_arr[:-2], x_mesh, y_mesh, log)

    def get_interface_z_treshhold(self,
                                  box_dims: dict[str, float]
                                  ) -> float:
        """find the treshhold of water highest point"""
        z_treshhold: float = box_dims['z_hi'] * self.oil_top_ratio
        self.info_msg += \
            (f'\tThe oil top ratio was set to `{self.oil_top_ratio:.3f}`\n'
             f'\tThe z treshold is set to `{z_treshhold:.3f}`\n')
        return z_treshhold

    def _get_xy_grid(self,
                     box_dims: dict[str, float]
                     ) -> tuple[np.ndarray, np.ndarray, float]:
        """return the mesh grid for the box"""
        mesh_size: float = \
            (box_dims['x_hi']-box_dims['x_lo'])/self.mesh_nr
        self.info_msg += f'\tThe number of meshes is `{self.mesh_nr**2}`\n'
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
        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(
                self._process_single_frame,
                [(frame, x_mesh, y_mesh, self.mesh_size, self.z_treshhold)
                 for frame in water_arr[:12]])
        for i_frame, result in enumerate(results):
            max_indices[i_frame] = result

        return max_indices

    @staticmethod
    def _process_single_frame(frame: np.ndarray,
                              x_mesh: np.ndarray,
                              y_mesh: np.ndarray,
                              mesh_size: float,
                              z_treshhold: float
                              ) -> list[np.int64]:
        """Process a single frame to find max water indices"""
        max_z_index: list[np.int64] = []
        xyz_i = frame.reshape(-1, 3)
        for i in range(x_mesh.shape[0]):
            for j in range(x_mesh.shape[1]):
                # Define the boundaries of the current mesh element
                x_min_mesh = x_mesh[i, j]
                x_max_mesh = x_mesh[i, j] + mesh_size
                y_min_mesh = y_mesh[i, j]
                y_max_mesh = y_mesh[i, j] + mesh_size

                # Select atoms within the current mesh element based on XY
                ind_in_mesh = np.where((xyz_i[:, 0] >= x_min_mesh) &
                                       (xyz_i[:, 0] < x_max_mesh) &
                                       (xyz_i[:, 1] >= y_min_mesh) &
                                       (xyz_i[:, 1] < y_max_mesh) &
                                       (xyz_i[:, 2] < z_treshhold))
                if len(ind_in_mesh[0]) > 0:
                    max_z = np.argmax(frame[2::3][ind_in_mesh])
                    max_z_index.append(ind_in_mesh[0][max_z])
        return max_z_index

    @staticmethod
    def _get_chunk_lists(arr_size: int,
                         n_cores: int
                         ) -> list[np.ndarray]:
        """chunk the main array"""
        integer_array: np.ndarray = np.arange(arr_size + 1)
        return np.array_split(integer_array, n_cores)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{GetSurface.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class AnalysisAqua:
    """get everything from water!"""
    def __init__(self,
                 parsed_com: "GetCom",
                 log: logger.logging.Logger
                 ) -> None:
        GetSurface(parsed_com.split_arr_dict['SOL'], parsed_com.box_dims, log)


if __name__ == "__main__":
    AnalysisAqua(GetCom(), log=logger.setup_logger("aqua_log"))
