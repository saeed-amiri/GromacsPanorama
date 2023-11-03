"""
Analysis the water phase
main porpose is getting the interface of water with minimum possible
reduction of surface from nanoparticle.
"""

import typing
import pandas as pd

from common import logger
from common.colors_text import TextColor as bcolors
from module3_analysis_com.com_file_parser import GetCom
from module3_analysis_com.com_plotter import ComPlotter


class GetSurface:
    """find the surface of the water"""

    info_msg: str = 'Message from GetSurface:\n'  # Meesage in methods to log
    oil_top_ratio: float = 2/3  # Where form top for sure should be oil

    def __init__(self,
                 water_df: pd.DataFrame,
                 box_dims: dict[str, float],
                 log: logger.logging.Logger
                 ) -> None:
        z_treshhold: float = self.get_interface_z_treshhold(box_dims)
        self.get_water_surface(water_df, z_treshhold, log)
        self._write_msg(log)

    def get_water_surface(self,
                          water_df: pd.DataFrame,
                          z_treshhold: float,  # Below this will be water
                          log: logger.logging.Logger
                          ) -> None:
        """
        mesh the box and find resides with highest z value in them
        """
        # To save a snapshot to see the system
        ComPlotter(
            com_arr=water_df[:-2], index=10, log=log, to_png=True, to_xyz=True)

    def get_interface_z_treshhold(self,
                                  box_dims: dict[str, float]
                                  ) -> float:
        """find the treshhold of water highest point"""
        z_treshhold: float = box_dims['z_hi'] * self.oil_top_ratio
        self.info_msg += \
            (f'\tThe oil top ratio was set to `{self.oil_top_ratio:.3f}`\n'
             f'\tThe z treshold is set to `{z_treshhold:.3f}`\n')
        return z_treshhold

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
