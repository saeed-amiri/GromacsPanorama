"""
calculating rdf from nanoparticle segment center.
The nanoparticle com for this situation (x, y, z_interface)
The z valu is only used for droping the oda which are not at interface
"""

import numpy as np
import pandas as pd

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
        self.np_com: np.ndarray = self._get_np_gmx(log)
        self.initiate(amino_arr, box_dims, log)
        self._write_msg(log)

    def initiate(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 box_dims: dict[str, float],  # Dimension of the Box
                 log: logger.logging.Logger
                 ) -> None:
        """initiate the calculations"""
        contact_info: pd.DataFrame = self.get_contact_info(log)
        interface_oda: dict[int, np.ndarray] = \
            self.get_interface_oda(contact_info, amino_arr[:-2])
        oda_distances: dict[int, np.ndarray] = \
            self.calc_distance_from_np(interface_oda, box_dims)

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
                              interface_oda: dict[int, np.ndarray],
                              box_dims: dict[str, float]
                              ) -> dict[int, np.ndarray]:
        """calculate the oda-np dictances by applying pbc"""
        l_xy: tuple[float, float] = (box_dims['x_hi'] - box_dims['x_lo'],
                                     box_dims['y_hi'] - box_dims['y_lo'])
        oda_distances: dict[int, np.ndarray] = {}
        for frame, arr in interface_oda.items():
            distance_i = np.zeros((arr.shape[0], 1))
            dx_i = arr[:, 0] - self.np_com[:len(arr), 0]
            dx_pbc = dx_i - (l_xy[0] * np.round(dx_i/l_xy[0]))
            dy_i = arr[:, 1] - self.np_com[:len(arr), 1]
            dy_pbc = dy_i - (l_xy[1] * np.round(dy_i/l_xy[1]))
            distance_i = np.sqrt(dx_pbc**2 + dy_pbc**2)
            oda_distances[frame] = distance_i
        return oda_distances

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

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{RdfClculation.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)

if __name__ == '__main__':
    pass
