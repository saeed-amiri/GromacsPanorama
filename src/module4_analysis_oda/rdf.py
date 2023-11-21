"""
calculating rdf from nanoparticle segment center.
The nanoparticle com for this situation (x, y, z_interface)
The z valu is only used for droping the oda which are not at interface
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from common import logger
from common import my_tools
from common import xvg_to_dataframe
from common import static_info as stinfo
from common.colors_text import TextColor as bcolors


class RdfClculation:
    """calculate radial distribution function"""

    info_msg: str = 'Message from RdfCalculation:\n'

    def __init__(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 box_dims: dict[str, float],  # Dimension of the Box
                 log: logger.logging.Logger
                 ) -> None:
        self.np_com: np.ndarray = self._get_xvg_gmx('coord.xvg', log)
        self.box_size: np.ndarray = self._get_xvg_gmx('box.xvg', log)
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
        contact_info: pd.DataFrame = self.get_contact_info(log)
        interface_oda: dict[int, np.ndarray] = \
            self.get_interface_oda(contact_info, amino_arr[:-2])
        oda_distances: dict[int, np.ndarray] = \
            self.calc_distance_from_np(interface_oda)
        bin_edges, rdf = self.calc_rdf_dy(oda_distances, bin_width=0.9)
        plt.plot(bin_edges, rdf)
        plt.xlim(-10, box_xyz[0]/2)
        plt.show()

    def get_contact_info(self,
                         log: logger.logging.Logger
                         ) -> pd.DataFrame:
        """
        read the dataframe made by aqua analysing named "contact.info"
        """
        my_tools.check_file_exist(fname := 'contact.info', log)
        self.info_msg += f'\tReading `{fname}`\n'
        return pd.read_csv(fname, sep=' ')

    def calc_distance_from_np(self,
                              interface_oda: dict[int, np.ndarray]
                              ) -> dict[int, np.ndarray]:
        """calculate the oda-np dictances by applying pbc"""
        oda_distances: dict[int, np.ndarray] = {}
        for frame, arr in interface_oda.items():
            distance_i = np.zeros((arr.shape[0], 1))

            dx_i = arr[:, 0] - self.np_com[:len(arr), 0]
            dx_pbc = dx_i - (self.box_size[frame][0] *
                             np.round(dx_i/self.box_size[frame][0]))

            dy_i = arr[:, 1] - self.np_com[:len(arr), 1]
            dy_pbc = dy_i - (self.box_size[frame][1] *
                             np.round(dy_i/self.box_size[frame][1]))

            dz_i = arr[:, 2] - self.np_com[:len(arr), 2]
            dz_pbc = dz_i - (self.box_size[frame][2] *
                             np.round(dz_i/self.box_size[frame][2]))

            distance_i = np.sqrt(dx_pbc**2 + dy_pbc**2 + dz_pbc**2)
            oda_distances[frame] = distance_i
        return oda_distances

    def calc_rdf(self,
                 distances_dict: dict[int, np.ndarray],
                 max_distance: float,
                 bin_width: float
                 ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the radial distribution function (RDF) for a set of
        distances.
        Returns:
        tuple: A tuple containing the bin centers and the RDF values.
        """
        # Determine the number of bins
        num_bins = int(max_distance / bin_width)
        volume: float = 4/3 * np.pi * max_distance**3

        rdf_counts: np.ndarray = np.zeros(num_bins)

        num_surfactants: int = \
            sum(len(distances) for distances in distances_dict.values())
        for distances in distances_dict.values():
            counts, _ = \
                np.histogram(distances, bins=num_bins, range=(0, max_distance))
            rdf_counts += counts
        # Normalize the RDF
        bin_edges = np.linspace(0, max_distance, num_bins + 1)
        bin_volumes: np.ndarray = \
            (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        rdf: np.ndarray = rdf_counts / (
            bin_volumes * len(distances_dict) * (num_surfactants / volume))

        bin_centers: np.ndarray = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, rdf

    def calc_rdf_dy(self,
                    distances_dict: dict[int, np.ndarray],
                    bin_width: float
                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the radial distribution function (RDF) for each frame,
        accounting for the changing box size.
        """
        rdf_list = []
        bin_centers_list = []

        for frame, distances in distances_dict.items():
            max_distance = self.get_max_distance_for_frame(frame)
            num_bins = int(max_distance / bin_width)
            volume = 4/3 * np.pi * max_distance**3

            rdf_counts = np.zeros(num_bins)
            num_surfactants = len(distances)

            # Calculate the histogram for the current frame
            counts, bin_edges = \
                np.histogram(distances, bins=num_bins, range=(0, max_distance))
            rdf_counts += counts

            # Calculate the volume of each bin
            bin_volumes = \
                (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

            # Normalize the RDF for the current frame
            rdf = rdf_counts / (bin_volumes * num_surfactants / volume)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Collect the RDF and bin centers
            rdf_list.append(rdf)
            bin_centers_list.append(bin_centers)

        # Average the RDF over all frames
        avg_rdf = np.mean(rdf_list, axis=0)
        avg_bin_centers = bin_centers_list[0]

        return avg_bin_centers, avg_rdf

    def get_max_distance_for_frame(self,
                                   frame: int
                                   ) -> float:
        """determine max distance for the given frame"""
        box_dimensions = self.box_size[frame]
        return np.max(box_dimensions) / 2

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

    def _get_xvg_gmx(self,
                     fxvg: str,  # Name of the xvg file
                     log: logger.logging.Logger
                     ) -> np.ndarray:
        """geeting the COM of the nanoparticles from gmx traj
        the file name is coord.xvg
        convert AA to nm
        """
        xvg_df: pd.Dataframe = \
            xvg_to_dataframe.XvgParser(fxvg, log).xvg_df
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
