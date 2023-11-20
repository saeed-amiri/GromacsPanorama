"""
Analysis the water phase
main porpose is getting the interface of water with minimum possible
reduction of surface from nanoparticle.
"""

import os
import warnings
import multiprocessing
import pandas as pd
import numpy as np

from common import logger
from common import cpuconfig
from common import xvg_to_dataframe
from common import static_info as stinfo
from common.colors_text import TextColor as bcolors
from common.com_file_parser import GetCom
from module3_analysis_aqua.com_plotter import ComPlotter
from module3_analysis_aqua.surface_plotter import SurfPlotter


class GetSurface:
    """find the surface of the water"""

    info_msg: str = 'Message from GetSurface:\n'  # Meesage in methods to log

    oil_top_ratio: float = 2/3  # Where form top for sure should be oil
    mesh_nr: float = 100.  # Number of meshes in each directions
    z_treshhold: float
    mesh_size: float

    surface_waters: dict[int, np.ndarray]  # Water at the surface, include np

    def __init__(self,
                 water_arr: np.ndarray,
                 box_dims: dict[str, float],
                 log: logger.logging.Logger
                 ) -> None:
        self.np_com: np.ndarray = self._get_np_gmx(log)
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
                   index=2,
                   log=log,
                   to_png=True,
                   to_xyz=True)
        # self.z_treshhold = self.get_interface_z_treshhold(box_dims)
        self.z_treshhold = 120
        x_mesh, y_mesh, self.mesh_size = self._get_xy_grid(box_dims)
        surface_indices: dict[int, list[np.int64]] = \
            self._get_surface_topology(water_arr[:-2], x_mesh, y_mesh, log)
        self.surface_waters: dict[int, np.ndarray] = \
            self.get_xyz_arr(water_arr[:-2], surface_indices)

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
                [(i_frame,
                  frame,
                  x_mesh,
                  y_mesh,
                  self.mesh_size,
                  self.z_treshhold)
                 for i_frame, frame in enumerate(water_arr)])
        for i_frame, result in enumerate(results):
            max_indices[i_frame] = result

        return max_indices

    def _process_single_frame(self,
                              i_frame: int,  # index of the frame
                              frame: np.ndarray,  # One frame of water com traj
                              x_mesh: np.ndarray,
                              y_mesh: np.ndarray,
                              mesh_size: float,
                              z_treshhold: float
                              ) -> list[np.int64]:
        """Process a single frame to find max water indices"""
        max_z_index: list[np.int64] = []
        min_z_treshhold: float = \
            self.np_com[i_frame, 2] - stinfo.np_info['radius']
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
                                       (xyz_i[:, 2] < z_treshhold) &
                                       (xyz_i[:, 2] > min_z_treshhold))
                if len(ind_in_mesh[0]) > 0:
                    max_z = np.argmax(frame[2::3][ind_in_mesh])
                    max_z_index.append(ind_in_mesh[0][max_z])
        return max_z_index

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

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{GetSurface.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)

    def _get_np_gmx(self,
                    log: logger.logging.Logger
                    ) -> np.ndarray:
        """geeting the COM of the nanoparticles from gmx traj
        the file name is coord.xvg
        convert AA to nm
        """
        xvg_df: pd.Dataframe = \
            xvg_to_dataframe.XvgParser('coord.xvg', log).xvg_df
        xvg_arr: np.ndarray = xvg_df.iloc[:, 1:4].to_numpy()
        return xvg_arr * 10


class AnalysisAqua:
    """get everything from water!"""

    info_msg: str = '-Message from AnalysisAqua:\n'
    surface_waters: dict[int, np.ndarray]  # All the surface waters
    contact_df: pd.DataFrame  # Final dataframe contains contact info
    selected_frames: list[int]  # To plot same series of the frames
    np_source: str = 'coord'  # If want to get the data from gromacs

    def __init__(self,
                 parsed_com: "GetCom",
                 log: logger.logging.Logger
                 ) -> None:
        surface = GetSurface(parsed_com.split_arr_dict['SOL'],
                             parsed_com.box_dims,
                             log)
        self.surface_waters = surface.surface_waters
        if self.np_source == 'coord':
            self.np_com: np.ndarray = surface.np_com
        else:
            self.np_com = parsed_com.split_arr_dict['APT_COR']
        self._initiate(parsed_com.box_dims, log)
        self._write_xvg(self.contact_df, fname="contact.xvg", log=log)

        self._write_msg(log)

    def _initiate(self,
                  box_dims: dict[str, float],
                  log: logger.logging.Logger
                  ) -> None:
        """initiate surface analysing"""
        self.selected_frames = SurfPlotter(surf_dict=self.surface_waters,
                                           box_dims=box_dims,
                                           np_com=self.np_com,
                                           log=log).selected_frames
        np_r: float = stinfo.np_info['radius']
        np_radius: np.ndarray = np.full((self.np_com.shape[0], 1), np_r)
        self.info_msg += f'\tThe radius of the NP was set to `{np_r}`\n'
        surface_water_under_np: dict[int, np.ndarray] = \
            self.drop_water_inside_radius(np_radius,
                                          box_dims,
                                          'under_r.png',
                                          log)
        interface_z_r: np.ndarray = \
            self.get_interface_z(surface_water_under_np)
        contact_r: np.ndarray = self.calc_contact_r(interface_z_r)
        self.drop_water_inside_radius(contact_r,
                                      box_dims,
                                      'contact_r.png',
                                      log)
        contact_angle: np.ndarray = self.calc_contact_angles(contact_r)
        self.contact_df = self.mk_df(contact_r, contact_angle, interface_z_r)
        self.info_msg += \
            (f'\tThe average of contact angle is: `{np.mean(contact_angle)}`\n'
             f'\tThe std of contact angle is: `{np.std(contact_angle)}`\n')

    def drop_water_inside_radius(self,
                                 radius: np.ndarray,
                                 box_dims: dict[str, float],
                                 fout_suffix: str,
                                 log: logger.logging.Logger
                                 ) -> dict[int, np.ndarray]:
        """Drop the water under the nanoparticle.
        The com of NP is known, and the radius of the NP is also known
        with some error. We calculate the contact radius of the np.
        First, we drop the water under np with radius r, then calculate
        the contact radius, and then from initial surface_waters, we
        drop the residues under the contact radius!
        """
        l_xy: tuple[float, float] = (box_dims['x_hi'] - box_dims['x_lo'],
                                     box_dims['y_hi'] - box_dims['y_lo'])
        surface_waters_under_r: dict[int, np.ndarray] = {}
        for frame, waters in self.surface_waters.items():
            np_radius = radius[frame]
            np_com_i = self.np_com[frame]
            dx_or = waters[:, 0] - np_com_i[0]
            dx_in = dx_or - (l_xy[0] * np.round(dx_or/l_xy[0]))
            dy_or = waters[:, 1] - np_com_i[1]
            dy_in = dy_or - (l_xy[1] * np.round(dy_or/l_xy[1]))
            distances: np.ndarray = np.sqrt(dx_in**2 + dy_in**2)
            outside_circle_mask: np.ndarray = distances < np_radius
            surface_waters_under_r[frame] = waters[~outside_circle_mask]

        SurfPlotter(surf_dict=surface_waters_under_r,
                    np_com=self.np_com,
                    box_dims=box_dims,
                    log=log,
                    indices=self.selected_frames,
                    fout_suffix=fout_suffix)
        return surface_waters_under_r

    def get_interface_z(self,
                        surface_water: dict[int, np.ndarray]
                        ) -> np.ndarray:
        """return the average of the z values for each frame"""
        return np.array(
            [np.mean(frame[:, 2]) for frame in surface_water.values()]
            ).reshape(-1, 1)

    def calc_contact_r(self,
                       interface_z_r: np.ndarray
                       ) -> np.ndarray:
        """calculate the contact radius based on the np center of mass,
        average interface location and radius of the np
        the hypothesis is the np com is under the interface!
        """
        under_water: bool = False
        r_contact = np.zeros(interface_z_r.shape)
        r_np_squre: float = stinfo.np_info['radius']**2
        for i, frame in enumerate(interface_z_r):
            deep = np.abs(self.np_com[i, 2] - frame)
            if (h_prime := r_np_squre - deep**2) >= 0:
                r_contact[i] = np.sqrt(h_prime)
                under_water = True
            else:
                r_contact[i] = np.nan
        r_contact += np.std(r_contact)
        if under_water:
            self.info_msg += \
                '\tIn one or more frames np is under the interface\n'
        return r_contact

    def calc_contact_angles(self,
                            contact_r: np.ndarray
                            ) -> np.ndarray:
        """calculate contact angles from contact radius"""
        contact_angles = np.zeros(contact_r.shape)
        np_radius: float = stinfo.np_info['radius']
        for i, frame in enumerate(contact_r):
            deep = np.sqrt(np_radius**2 - frame**2)
            h_deep = deep + np_radius
            contact_angles[i] = np.degrees(np.arccos((h_deep/np_radius)-1))
        return contact_angles

    def mk_df(self,
              contact_r: np.ndarray,
              contact_angle: np.ndarray,
              interface_z: np.ndarray
              ) -> pd.DataFrame:
        """make a df with everthing in it"""
        columns: list[str] = \
            ['contact_radius', 'contact_angles', 'interface_z', 'np_com_z']
        if (l_i := len(contact_r)) == len(contact_angle) == len(interface_z):
            data = {
                'contact_radius': contact_r.ravel(),
                'contact_angles': contact_angle.ravel(),
                'interface_z': interface_z.ravel(),
                'np_com_z': self.np_com[:l_i, 2].ravel()
            }
            df_i = pd.DataFrame(data, columns=columns)
            return df_i
        raise ValueError("Lengths of input arrays do not match.")

    def _write_xvg(self,
                   df_i: pd.DataFrame,
                   log: logger.logging.Logger,
                   fname: str = 'df.xvg'
                   ) -> None:
        """
        Write the data into xvg format
        Raises:
            ValueError: If the DataFrame has no columns.
        """
        if df_i.columns.empty:
            log.error(msg := "\tThe DataFrame has no columns.\n")
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}\n')
        if df_i.empty:
            log.warning(
                msg := f"The df is empty. `{fname}` will not contain data.")
            warnings.warn(msg, UserWarning)

        columns: list[str] = df_i.columns.to_list()

        header_lines: list[str] = [
            f'# Written by {self.__module__}',
            f"# Current directory: {os.getcwd()}",
            '@   title "Contact information"',
            '@   xaxis label "Frame index"',
            '@   yaxis label "Varies"',
            '@TYPE xy',
            '@ view 0.15, 0.15, 0.75, 0.85',
            '@legend on',
            '@ legend box on',
            '@ legend loctype view',
            '@ legend 0.78, 0.8',
            '@ legend length 2'
        ]
        legend_lines: list[str] = \
            [f'@ s{i} legend "{col}"' for i, col in enumerate(df_i.columns)]

        with open(fname, 'w', encoding='utf8') as f_w:
            for line in header_lines + legend_lines:
                f_w.write(line + '\n')
            df_i.to_csv(f_w, sep=' ', index=True, header=None, na_rep='NaN')

        self.info_msg += (f'\tThe dataframe saved to `{fname}` '
                          f'with columns:\n\t`{columns}`\n')

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{AnalysisAqua.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    AnalysisAqua(GetCom(), log=logger.setup_logger("aqua.log"))
