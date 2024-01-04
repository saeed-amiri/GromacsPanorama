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
from common.colors_text import TextColor as bcolors
from module6_charge_analysis import data_file_parser

if typing.TYPE_CHECKING:
    from module6_charge_analysis.data_file_parser import DataArrays


@dataclass
class InputFileConfigs:
    """Input file configurations"""
    f_contact: str = 'contact.xvg'
    f_coord: str = 'coord.xvg'
    f_box: str = 'box.xvg'


@dataclass
class ParameterConfigs:
    """default parameters for the calculataion"""
    number_of_regions: int = 150
    time_dependent_step: int = 100
    xvg_output: str = 'densities.xvg'


class Densities(typing.NamedTuple):
    """densities data structure"""
    density_per_region: dict[float, list[float]] = {}
    avg_density_per_region: dict[float, float] = {}
    rdf: dict[float, float] = {}


class ResidueDensityAroundNanoparticle:
    """Calculate densities, rdf, and cdf around a nanoparticle."""

    info_msg: str = 'Messsges from ResidueDensityAroundNanoparticle:\n'
    input_config: "InputFileConfigs"
    param_config: "ParameterConfigs"
    res_name: str
    res_arr: np.ndarray
    densities: "Densities"

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
        self.res_arr = res_arr[:-2]
        self.densities = self._initiate(log, res_name)
        self.write_msg(log)

    def _initiate(self,
                  log: logger.logging.Logger,
                  res_name: str
                  ) -> "Densities":
        """initiate computaitons from here"""
        density_per_region: dict[float, list[float]]
        avg_density_per_region: dict[float, float]
        num_res_in_radius: list[int]
        rdf: dict[float, float]

        parsed_data: "DataArrays" = data_file_parser.ParseDataFiles(
            input_config=self.input_config, log=log).data_arrays
        regions: list[float] = self.generate_regions(
            self.param_config.number_of_regions, parsed_data, update_msg=True)
        density_per_region, num_res_in_radius = \
            self.initiate_calculation(parsed_data, regions)
        avg_density_per_region = self.compute_avg_density(density_per_region)
        rdf = self.compute_rdf(density_per_region, num_res_in_radius)

        self.info_msg += f'\tDensities for residue `{res_name}` are computed\n'
        return Densities(density_per_region, avg_density_per_region, rdf)

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

    def initiate_calculation(self,
                             parsed_data: "DataArrays",
                             regions: list[float],
                             ) -> tuple[dict[float, list[float]],
                                        list[int]]:
        """initiate computaion by getting denisty"""
        # Initialize a dictionary to store densities for each region
        density_per_region: dict[float, list[float]] = \
            {region: [] for region in regions}
        num_res_in_radius: list[int] = []
        for i, frame_i in enumerate(self.res_arr):
            arr_i: np.ndarray = frame_i.reshape(-1, 3)
            np_com_i = parsed_data.np_com[i]
            box_i = parsed_data.box[i]
            distance: np.ndarray = \
                self._compute_pbc_distance(arr_i, np_com_i, box_i)
            density_per_region, count_in_radius = \
                self._compute_density_per_region(regions,
                                                 distance,
                                                 density_per_region)
            num_res_in_radius.append(count_in_radius)
        return density_per_region, num_res_in_radius

    @staticmethod
    def compute_avg_density(density_per_region: dict[float, list[float]]
                            ) -> dict[float, float]:
        """self explanatory"""
        avg_density_per_region: dict[float, float] = {}
        for region, densities in density_per_region.items():
            if densities:
                avg_density_per_region[region] = np.mean(densities)
            else:
                avg_density_per_region[region] = 0
        return avg_density_per_region

    def compute_rdf(self,
                    density_per_region: dict[float, list[float]],
                    num_oda: list[int]
                    ) -> dict[float, float]:
        """set the 2d rdf (g(r))"""
        max_radius_area: float = \
            max(item for item in density_per_region.keys())
        rdf: dict[float, float] = {}
        for region, densities in density_per_region.items():
            if not densities:
                rdf[region] = 0
                continue

            tmp = []
            for j, item in enumerate(densities):
                density: float = num_oda[j]/(np.pi * max_radius_area**2)
                tmp.append(item/density)
            rdf[region] = np.mean(tmp)
        return rdf

    def _compute_pbc_distance(self,
                              arr: np.ndarray,
                              np_com: np.ndarray,
                              box: np.ndarray
                              ) -> np.ndarray:
        """claculating the distance between the np and the surfactants
        at each frame and return an array
        Only considering 2d distance, in the XY plane
        """
        dx_i = arr[:, 0] - np_com[0]
        dx_pbc = dx_i - (box[0] * np.round(dx_i/box[0]))
        dy_i = arr[:, 1] - np_com[1]
        dy_pbc = dy_i - (box[1] * np.round(dy_i/box[1]))
        dz_i = arr[:, 2] - np_com[2]
        dz_pbc = dz_i - (box[2] * np.round(dz_i/box[2]))
        return np.sqrt(dx_pbc*dx_pbc + dy_pbc*dy_pbc + dz_pbc*dz_pbc)

    @staticmethod
    def _compute_density_per_region(regions: list[float],
                                    distance: np.ndarray,
                                    density_per_region:
                                    dict[float, list[float]]
                                    ) -> tuple[dict[float, list[float]], int]:
        """self explanatory"""
        # Here, the code increments the count in the appropriate bin
        # by 2. The increment by 2 is necessary because each pair
        # contributes to two interactions: one for each atom in the
        # pair being considered as the reference atom. In simpler
        # terms, for each pair of atoms, both atoms contribute to the
        # density at this distance, so the count for this bin is
        # increased by 2 to account for both contributions.
        count_in_radius = 2
        for ith in range(len(regions) - 1):
            r_inner = regions[ith]
            r_outer = regions[ith + 1]
            count = np.sum((distance >= r_inner) & (distance < r_outer))
            volume = 4 * np.pi * (r_outer**3 - r_inner**3) / 3
            density = count / volume
            density_per_region[r_outer].append(density)
            count_in_radius += count
        return density_per_region, count_in_radius

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    pass
