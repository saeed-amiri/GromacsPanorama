"""
densities_around_np.py

This module is designed to calculate various properties such as
densities, radial distribution functions (rdf), and cumulative
distribution functions (cdf) for ions or other residues in the vicinity
of a nanoparticle. Unlike the 'oda_density_around_np.py' script in
module4, which focuses on two-dimensional analyses,
'densities_around_np.py' extends these computations to three dimensions.
This enhancement allows for a more comprehensive and spatially detailed
understanding of the distribution and interactions of ions or residues
around nanoparticles in a 3D space.

Functions:
    - calculate_density: Computes the density distribution of ions or
        residues around the nanoparticle in 3D.

    - compute_rdf: Calculates the radial distribution function, providing
    insights into the spatial distribution and local ordering of particles.

    - compute_cdf: Determines the cumulative distribution function,
    offering a cumulative perspective of particle distribution up to a
    certain radius.

This script facilitates a deeper exploration of the structural and
dynamic properties of systems involving nanoparticles and their
surrounding environment in three-dimensional space.
Jan 3, 2023
Saeed
"""

import typing
from dataclasses import dataclass

import numpy as np

from common import logger
from module6_charge_analysis import data_file_parser

if typing.TYPE_CHECKING:
    from module6_charge_analysis.data_file_parser import DataArrays


@dataclass
class InputFileConfigs:
    """set the name of the inputs files"""
    f_contact: str = 'contact.xvg'
    f_coord: str = 'coord.xvg'
    f_box: str = 'box.xvg'


@dataclass
class ParameterConfigs:
    """set the default parameters for the calculataion"""
    number_of_regions: int = 150
    time_dependent_step: int = 100
    xvg_output: str = 'densities.xvg'


class ResidueDensityAroundNanoparticle:
    """self explanatory"""

    info_msg: str = 'Messsges from ResidueDensityAroundNanoparticle:\n'
    input_config: "InputFileConfigs"
    param_config: "ParameterConfigs"
    res_name: str

    def __init__(self,
                 res_arr: np.ndarray,
                 log: logger.logging.Logger,
                 res_name: str,
                 input_config: "InputFileConfigs" = InputFileConfigs(),
                 param_config: "ParameterConfigs" = ParameterConfigs()
                 ) -> None:
        self.input_config = input_config
        self.param_config = param_config
        self.res_name = res_name
        self._initiate(res_arr, log, res_name)

    def _initiate(self,
                  res_arr: np.ndarray,
                  log: logger.logging.Logger,
                  res_name: str
                  ) -> None:
        """initiate computaitons from here"""
        parsed_data: "DataArrays" = data_file_parser.ParseDataFiles(
            input_config=self.input_config, log=log).data_arrays
        regions: list[float] = self.generate_regions(
            self.param_config.number_of_regions, parsed_data, update_msg=True)

    def generate_regions(self,
                         nr_of_regions: int,
                         parsed_data: "DataArrays",
                         update_msg: bool = True
                         ) -> list[float]:
        """divide the the area around the np for generating regions"""
        max_box_len: np.float64 = np.max(parsed_data.box[0][:2]) / 2
        if update_msg:
            self.info_msg += (
                f'\tThe number of regions is: `{nr_of_regions}`\n'
                f'\tThe half of the max length of box is `{max_box_len:.3f}`\n'
                f'\tThe bin size is: `{max_box_len/nr_of_regions:.3f}`\n'
            )
        return np.linspace(0, max_box_len, nr_of_regions).tolist()


if __name__ == '__main__':
    pass
