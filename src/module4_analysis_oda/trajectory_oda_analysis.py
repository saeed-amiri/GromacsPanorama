"""
Analysing the surfactants behavior at the interface
rdf and other possible data
This script uses the out put of the module3_aqua_analysis/contact.info
"""

import sys
from dataclasses import dataclass
from collections import namedtuple

from module4_analysis_oda.rdf import RdfCalculationConfig, RdfClculation
from module4_analysis_oda.oda_density_around_np import \
    ParameterConfig, SurfactantDensityAroundNanoparticle
from module4_analysis_oda.oda_density_around_np_plotter import \
    SurfactantDensityPlotter
from module4_analysis_oda.oda_density_contrast import \
    SurfactantsLocalizedDensityContrast
from module4_analysis_oda.oda_density_contrast_plotter import \
    SurfactantContrastDensityPlotter
from module4_analysis_oda.oda_xy_plotter import PlotSurfactantComXY
from module4_analysis_oda.density_overlay_plotter import OverlayPlotDensities

from common import logger
from common.com_file_parser import GetCom
from common.colors_text import TextColor as bcolors


FitTurns: "namedtuple" = \
    namedtuple('FitTurns', ['first_turn', 'midpoint', 'second_turn'])


@dataclass
class ComputationCalculations:
    """To select computions"""
    plain_rdf: bool = False
    contrast_density: bool = False
    density_rdf: bool = True
    overlay_plot: bool = True


class OdaAnalysis:
    """call all the other scripts here"""

    fit_turns: "FitTurns"

    def __init__(self,
                 fname: str,  # Name of the com file
                 log: logger.logging.Logger,
                 compute_config: "ComputationCalculations" =
                 ComputationCalculations()
                 ) -> None:
        self.parsed_com = GetCom(fname)
        self.compute_config = compute_config
        self.initiate(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> None:
        """call the scripts"""
        # pylint: disable=broad-exception-caught
        try:
            if self.compute_config.plain_rdf:
                self.plain_rdf_calculation(log)

            if self.compute_config.density_rdf:
                self.compute_plot_densities(log)

            if self.compute_config.contrast_density:
                self.compute_contrast_density(log)

            if self.compute_config.overlay_plot:
                self.plot_overlay_densities(log)

        except Exception as err:
            log.error(f"An error occurred during ODA analysis: {err}")

    def plain_rdf_calculation(self,
                              log: logger.logging.Logger
                              ) -> None:
        """calculate rdf (legecy)"""
        rdf_config = RdfCalculationConfig(
            amino_arr=self.parsed_com.split_arr_dict['AMINO_ODN'],
            box_dims=self.parsed_com.box_dims)
        RdfClculation(log, rdf_config)

    def compute_plot_densities(self,
                               log: logger.logging.Logger
                               ) -> None:
        """compute density and rdf"""
        for residue in ['AMINO_ODN', 'CLA']:
            params: "ParameterConfig" = \
                ParameterConfig(number_of_regions=50,
                                time_dependent_step=101,
                                xvg_output=f'{residue}_densities.xvg')
            oda_density = SurfactantDensityAroundNanoparticle(
                self.parsed_com.split_arr_dict[residue],
                log,
                param_config=params,
                residue=residue)
            SurfactantDensityPlotter(oda_density, log, residue=residue)
            if residue == 'AMINO_ODN':
                self.fit_turns = \
                    FitTurns(first_turn=oda_density.first_turn,
                             midpoint=oda_density.midpoint,
                             second_turn=oda_density.second_turn)

    def compute_contrast_density(self,
                                 log: logger.logging.Logger
                                 ) -> None:
        """compute the contrast density for the oda"""
        contrast = SurfactantsLocalizedDensityContrast(
            self.parsed_com.split_arr_dict['AMINO_ODN'], log)
        SurfactantContrastDensityPlotter(contrast.number_density, log)
        PlotSurfactantComXY(self.parsed_com.split_arr_dict['AMINO_ODN'], log)

    def plot_overlay_densities(self,
                               log: logger.logging.Logger
                               ) -> None:
        """
        Plot the overlay densities (density, rdf, ... for ODA and
        Cl.
        """
        if not self.compute_config.density_rdf:
            log.error(msg := ('\n\tError! Needs the turns points from '
                              '`SurfactantDensityAroundNanoparticle`\n'))
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
        OverlayPlotDensities(
            {'AMINO_ODN_densities.xvg': 'ODA', 'CLA_densities.xvg': 'Cl'},
            self.fit_turns,
            log)


if __name__ == '__main__':
    LOG = logger.setup_logger('oda_analysing.log')
    try:
        analysis = OdaAnalysis(fname=sys.argv[1], log=LOG)
    except IndexError:
        LOG.error(MSG := "No command line argument provided for the filename.")
        print(f"{bcolors.FAIL}{MSG}{bcolors.ENDC}")
        sys.exit(1)
    else:
        LOG.info(MSG := "ODA analysis is doen.")
        sys.exit(f"{bcolors.CAUTION}{MSG}{bcolors.ENDC}")
