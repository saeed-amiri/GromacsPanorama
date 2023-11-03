"""
Analysis the water phase
main porpose is getting the interface of water with minimum possible
reduction of surface from nanoparticle.
"""

import typing

from common import logger
from module3_analys_com.com_file_parser import GetCom

class AnalysisAqua:
    """get everything from water!"""
    def __init__(self,
                 parsed_com: "GetCom",
                 log: logger.logging.Logger
                 ) -> None:
        print(parsed_com.__dict__)

if __name__ == "__main__":
    AnalysisAqua(GetCom(), log=logger.setup_logger("aqua_log"))
