"""
calculating rdf from nanoparticle segment center.
The nanoparticle com for this situation (x, y, z_interface)
The z valu is only used for droping the oda which are not at interface
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from common import logger
from common import static_info as stinfo
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors


@dataclass
class FilePaths:
    """path of the all the needed files"""
    coord_xvg: str = 'coord.xvg'
    box_xvg: str = 'box.xvg'
    contact_xvg: str = 'contact.xvg'


class RdfClculation:
    """calculate radial distribution function"""

    info_msg: str = 'Message from RdfCalculation:\n'

    def __init__(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 box_dims: dict[str, float],  # Dimension of the Box
                 log: logger.logging.Logger,
                 file_pathes: "FilePaths" = FilePaths()
                 ) -> None:
        self.file_pathes = file_pathes
        self.np_com: np.ndarray = self.get_xvg_gmx(file_pathes.coord_xvg, log)
        self.box_size: np.ndarray = self.get_xvg_gmx(file_pathes.box_xvg, log)
        self.initiate(amino_arr, box_dims, log)
        self._write_msg(log)

    def initiate(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 box_dims: dict[str, float],  # Dimension of the Box
                 log: logger.logging.Logger
                 ) -> None:
        """initiate the calculations"""
        box_xyz: tuple[float, float, float] = \
            (box_dims['x_hi'] - box_dims['x_lo'],
             box_dims['y_hi'] - box_dims['y_lo'],
             box_dims['z_hi'] - box_dims['z_lo'])
        contact_info: pd.DataFrame = \
            self.get_contact_info(self.file_pathes.contact_xvg, log)
        interface_oda: dict[int, np.ndarray] = \
            self.get_interface_oda(contact_info, amino_arr[:-2])
        oda_distances: dict[int, np.ndarray] = \
            self.calc_distance_from_np(interface_oda)
        bin_edges, rdf = self.calc_rdf(oda_distances, bin_width=0.9)
        plt.plot(bin_edges, rdf)
        plt.xlim(-10, box_xyz[0]/2)
        plt.show()

    def get_contact_info(self,
                         fname: str,
                         log: logger.logging.Logger
                         ) -> pd.DataFrame:
        """
        read the dataframe made by aqua analysing named "contact.xvg"
        """
        self.info_msg += f'\tReading `{fname}`\n'
        return xvg.XvgParser(fname, log).xvg_df

    def calc_distance_from_np(self,
                              interface_oda: dict[int, np.ndarray]
                              ) -> dict[int, np.ndarray]:
        """calculate the oda-np dictances by applying pbc"""
        oda_distances: dict[int, np.ndarray] = {}
        for frame, arr in interface_oda.items():
            distance_i = np.zeros((arr.shape[0], 1))

            dx_pbc = self._apply_pbc(arr, axis=0, frame=frame)
            dy_pbc = self._apply_pbc(arr, axis=1, frame=frame)
            dz_pbc = self._apply_pbc(arr, axis=2, frame=frame)

            distance_i = np.sqrt(dx_pbc**2 + dy_pbc**2 + dz_pbc**2)
            oda_distances[frame] = distance_i
        return oda_distances

    def _apply_pbc(self,
                   arr: np.ndarray,
                   axis: int,  # 0 -> x, 1 -> y, 2 -> z
                   frame: int  # Index of the frame
                   ) -> np.ndarray:
        """apply pbc to each axis"""
        dx_i = arr[:, axis] - self.np_com[frame, axis]
        dx_pbc = dx_i - (self.box_size[frame][axis] *
                         np.round(dx_i/self.box_size[frame][axis]))
        return dx_pbc

    def calc_rdf(self,
                 distances_dict: dict[int, np.ndarray],
                 bin_width: float
                 ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the radial distribution function (RDF) for each frame,
        accounting for the changing box size.
        """
        rdf_list = []
        bin_centers_list = []

        max_distance: float = self._get_max_distance(distances_dict)
        num_bins = int(max_distance / bin_width)

        for _, distances in distances_dict.items():
            volume = 4/3 * np.pi * max_distance**3
            rdf, bin_centers = self._calculate_histogram(
                distances, max_distance, num_bins, volume)
            rdf_list.append(rdf)
            bin_centers_list.append(bin_centers)

        # Average the RDF over all frames
        avg_rdf = np.mean(rdf_list, axis=0)
        avg_bin_centers = bin_centers_list[0]

        return avg_bin_centers, avg_rdf

    def _get_max_distance(self,
                          distances_dict: dict[int, np.ndarray]
                          ) -> float:
        """Get the maximum distance across all frames."""
        return max(self.get_max_distance_for_frame(frame) for
                   frame in distances_dict.keys())

    def get_max_distance_for_frame(self,
                                   frame: int
                                   ) -> float:
        """determine max distance for the given frame"""
        box_dimensions = self.box_size[frame]
        return float(np.max(box_dimensions) / 2)

    def _calculate_histogram(self,
                             distances: np.ndarray,
                             max_distance: float,
                             num_bins: int,
                             volume: float
                             ) -> tuple[np.ndarray, np.ndarray]:
        rdf_counts = np.zeros(num_bins)
        num_surfactants = len(distances)

        counts, bin_edges = \
            np.histogram(distances, bins=num_bins, range=(0, max_distance))
        rdf_counts += counts

        bin_volumes = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        rdf = rdf_counts / (bin_volumes * num_surfactants / volume)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return rdf, bin_centers

    @staticmethod
    def get_interface_oda(contact_info: pd.DataFrame,
                          amino_arr: np.ndarray
                          ) -> dict[int, np.ndarray]:
        """get the oda at interface"""
        interface_z: np.ndarray = \
            contact_info['interface_z'].to_numpy().reshape(-1, 1)
        np_radius: float = stinfo.np_info['radius']
        interface_oda: dict[int, np.ndarray] = {}
        for i_frame, frame in enumerate(amino_arr):
            xyz_i: np.ndarray = frame.reshape(-1, 3)
            ind_at_interface: list[int] = []
            ind_at_interface = np.where(
                (xyz_i[:, 2] <= interface_z[i_frame] + np_radius/2) &
                (xyz_i[:, 2] >= interface_z[i_frame] - np_radius/2))
            interface_oda[i_frame] = xyz_i[ind_at_interface[0]]
        return interface_oda

    def get_xvg_gmx(self,
                    fxvg: str,  # Name of the xvg file
                    log: logger.logging.Logger
                    ) -> np.ndarray:
        """geeting the COM of the nanoparticles from gmx traj
        the file name is coord.xvg
        convert AA to nm
        """
        xvg_df: pd.Dataframe = \
            xvg.XvgParser(fxvg, log).xvg_df
        xvg_arr: np.ndarray = xvg_df.iloc[:, 1:4].to_numpy()
        return xvg_arr * 10

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    pass
