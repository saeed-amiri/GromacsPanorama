"""
Analysing the surfactants behavior at the interface
rdf and other possible data
This script uses the out put of the module3_aqua_analysis/contact.info
"""

import sys

from module4_analysis_oda.rdf import RdfCalculationConfig, RdfClculation
from module4_analysis_oda.oda_density_around_np import \
    OdaInputFilesConfig, ParameterConfig, SurfactantDensityAroundNanoparticle
from module4_analysis_oda.oda_density_around_np_plotter import \
    DensityHeatMapConfig, DensityGraphConfig, Rdf2dGraphConfig, \
    SurfactantDensityPlotter
from module4_analysis_oda.oda_density_contrast import \
    InputFilesConfig, SurfactantsLocalizedDensityContrast
from module4_analysis_oda.oda_density_contrast_plotter import \
    GraphConfing, SurfactantContrastDensityPlotter
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
        # rdf_config = RdfCalculationConfig(
            # amino_arr=parsed_com.split_arr_dict['AMINO_ODN'],
            # box_dims=parsed_com.box_dims)
        # RdfClculation(log, rdf_config)
        params: "ParameterConfig" = ParameterConfig(number_of_regions=50)
        oda_density = SurfactantDensityAroundNanoparticle(
            parsed_com.split_arr_dict['AMINO_ODN'], log, param_config=params)
        SurfactantDensityPlotter(oda_density, log)
        # contrast = SurfactantsLocalizedDensityContrast(
        #     parsed_com.split_arr_dict['AMINO_ODN'], log)
        # SurfactantContrastDensityPlotter(contrast.number_density, log)
        


if __name__ == '__main__':
    OdaAnalysis(fname=sys.argv[1],
                log=logger.setup_logger('oda_analysing.log'))
