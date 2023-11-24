"""
Analysing the surfactants behavior at the interface
rdf and other possible data
This script uses the out put of the module3_aqua_analysis/contact.info
"""

import sys

from module4_analysis_oda import rdf
from module4_analysis_oda.oda_density_around_np import \
    OdaInputConfig, SurfactantDensityAroundNanoparticle
from module4_analysis_oda.surfactant_density_plotter import \
    PlotConfig, SurfactantDensityPlotter
from common import logger
from common.com_file_parser import GetCom


class OdaAnalysis:
    """call all the other scripts here"""
    def __init__(self,
                 fname: str,  # Name of the com file
                 log: logger.logging.Logger
                 ) -> None:
        parsed_com = GetCom(fname)
        self.initiate(parsed_com, log)

    def initiate(self,
                 parsed_com: "GetCom",
                 log: logger.logging.Logger
                 ) -> None:
        """call the scripts"""
        # rdf.RdfClculation(parsed_com.split_arr_dict['AMINO_ODN'],
        # #               #   parsed_com.box_dims, log)
        oda_density = SurfactantDensityAroundNanoparticle(
            parsed_com.split_arr_dict['AMINO_ODN'], log)
        SurfactantDensityPlotter(oda_density, log)


if __name__ == '__main__':
    OdaAnalysis(fname=sys.argv[1],
                log=logger.setup_logger('oda_analysing.log'))
