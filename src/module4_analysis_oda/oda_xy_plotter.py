"""
plot the center of the masses of the oda on a 2d plane to see the
density
"""

import typing
from dataclasses import dataclass, field
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.patches import Circle

from common import logger
from common import plot_tools
from common import static_info as stinfo
from common.colors_text import TextColor as bcolors
from common import xvg_to_dataframe as xvg

if typing.TYPE_CHECKING:
    from module4_analysis_oda.oda_density_around_np \
        import SurfactantDensityAroundNanoparticle


class PlotSurfactantComXY:
    """plot the seurfactants xy positions"""

    info_msg: str = 'Message from PlotSurfactantComXY:\n'

    def __init__(self,
                 amino_arr: np.ndarray,
                 log: logger.logging.Logger
                 ) -> None:
        self.amino_arr = amino_arr[:-2]
        self._initiate(log)
        self.write_msg(log)

    def _initiate(self,
                  log: logger.logging.Logger
                  ) -> None:
        """initiate the data and plotting"""
        self.process_data(log)

    def process_data(self,
                     log: logger.logging.Logger
                     ) -> None:
        """process data by split them based on the frames and than
        subtract the NP com from them, and finally applying pbc"""
        np_com: np.ndarray = self.parse_gmx_xvg(self.load_np_com_data(log))
        box: np.ndarray = self.parse_gmx_xvg(self.load_box_data(log))

        data_dict: dict[int, np.ndarray] = self.get_data_dict()
        centered_data: dict[int, np.ndarray] = \
            self.subtract_np_com(np_com, data_dict)
        pbc_data: dict[int, np.ndarray] = \
            self.apply_pbc(centered_data, np_com, box)
        for i, frame in pbc_data.items():
            plt.scatter(frame[0, :], frame[1, :])
        plt.show()

    def get_data_dict(self) -> dict[int, np.ndarray]:
        """split data framewise"""
        data_dict: dict[int, np.ndarray] = {}
        for i, frame in enumerate(self.amino_arr):
            data_dict[i] = frame.reshape(-1, 3)
        return data_dict

    def subtract_np_com(self,
                        np_com: np.ndarray,
                        data_dict: dict[int, np.ndarray]
                        ) -> dict[int, np.ndarray]:
        """subtract np com from the oda"""
        centered_data: dict[int, np.ndarray] = {}
        for key, value in data_dict.items():
            centered_data[key] = value - np_com[key, :]
        return centered_data

    def apply_pbc(self,
                  data_dict: dict[int, np.ndarray],
                  np_com: np.ndarray,
                  box: np.ndarray
                  ) -> dict[int, np.ndarray]:
        """apply pbc to all the oda with np as refrence"""
        pbc_data: dict[int, np.ndarray] = {}
        for k, arr in data_dict.items():
            arr_pbc: np.ndarray = np.zeros(arr.shape)
            for i in range(arr.shape[1]):
                dx_i = arr[:, i] - np_com[k, i]
                arr_pbc[:, i] = dx_i - (box[k, i] * np.round(dx_i/box[k, i]))
            pbc_data[k] = arr_pbc
        return pbc_data

    def load_np_com_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """
        Load and return the nanoparticle center of mass data from XVG
        file."""
        return xvg.XvgParser('coord.xvg', log).xvg_df

    def load_box_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """Load and return the box dimension data from XVG file."""
        return xvg.XvgParser('box.xvg', log).xvg_df

    @staticmethod
    def parse_gmx_xvg(np_com_df: pd.DataFrame
                      ) -> np.ndarray:
        """return the nanoparticle center of mass as an array"""
        unit_nm_to_angestrom: float = 10
        return np_com_df.iloc[:, 1:4].to_numpy() * unit_nm_to_angestrom

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
