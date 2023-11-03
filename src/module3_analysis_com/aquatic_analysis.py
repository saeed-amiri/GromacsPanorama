"""
Analysis the water phase
main porpose is getting the interface of water with minimum possible
reduction of surface from nanoparticle.
"""

import typing
import pandas as pd

from common import logger
from module3_analysis_com.com_file_parser import GetCom


class GetSurface:
    """find the surface of the water"""
    
    info_msg: str = 'Message from GetSurface:\n'  # Meesage in methods to log

    def __init__(self,
                 water_df: pd.DataFrame,
                 log: logger.logging.Logger
                 ) -> None:
        self.get_water_surface(water_df, log)

    def get_water_surface(self,
                 water_df: pd.DataFrame,
                 log: logger.logging.Logger
                 ) -> None:
        """
        mesh the box and find resides with highest z value in them
        """
        print(water_df)
        


class AnalysisAqua:
    """get everything from water!"""
    def __init__(self,
                 parsed_com: "GetCom",
                 log: logger.logging.Logger
                 ) -> None:
        GetSurface(parsed_com.split_arr_dict['SOL'], log)


if __name__ == "__main__":
    AnalysisAqua(GetCom(), log=logger.setup_logger("aqua_log"))
