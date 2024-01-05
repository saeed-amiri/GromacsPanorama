"""
densities_around_np.py

Module to calculate densities, radial distribution functions (rdf),
and cumulative distribution functions (cdf) around nanoparticles in 3D.

Functions:
    - calculate_density: Computes the density distribution in 3D.
    - compute_rdf: Calculates the radial distribution function.
    - compute_cdf: Determines the cumulative distribution function.

This script enables exploration of structural and dynamic properties
involving nanoparticles in a 3D space.

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
    res_name: str
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
        return Densities(
            res_name, density_per_region, avg_density_per_region, rdf)

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
            if not densities:
                avg_density_per_region[region] = 0
            else:
                avg_density_per_region[region] = np.mean(densities)
        return avg_density_per_region

    def compute_rdf(self,
                    density_per_region: dict[float, list[float]],
                    num_oda: list[int]
                    ) -> dict[float, float]:
        """set the 3d rdf (g(r))
        The use of dr (div_r) in the normalization factor for the
        radial distribution function (RDF) calculation is important
        for a couple of reasons:
         Volume of Shells: In the RDF calculation, we are essentially
         counting the number of particle pairs within spherical shells
         of thickness dr at different radii. The volume of each shell
         is given by the formula for the volume of a spherical shell,
         which depends on dr. Specifically, the volume of a shell
         between radii r and r + dr is:
         4/3pi((r+dr)3-r3)4/3pi((r+dr)3-r3).

        Normalization Purpose: The normalization factor is used to
        convert the raw pair counts in each shell into a density.
        The RDF is a measure of density relative to an ideal gas at
        the same number density, so we need to account for how many
        pairs we would expect to find in each shell if the particles
        randomly distributed.
        This expected number depends on the volume of each shell (which
        includes dr) and the overall number density of particles.

        In the provided code, the normalization factor is calculated
        as follows:

        normalization_factor is initially set to:
         (N*(N-1)/2)*(4pidr^3)(N*(N-1)/2)*(4πdr^3).
        This accounts for the total number of unique pairs of particles
        (since each pair is counted twice in the double loop) and a
        volume scaling factor.
        rdf /= shell_volumes * number_density * normalization_factor:
        This line adjusts the RDF by dividing by the volume of each
        shell, the number density of particles, and the previously
        mentioned normalization factor. The term shell_volumes is the
        array of volumes of the individual shells, which is computed
        as:
         4/3pi((radii+dr)3-radii^3).

        By including dr in this way, we ensure that the RDF reflects
        the true spatial distribution of particles, normalized correctly
        for the volume in which they are counted. This normalization is
        crucial for comparing the RDF to theoretical models or RDFs
        from different systems.
        """
        max_radius_area: float = \
            max(item for item in density_per_region.keys())
        div_r: float = max_radius_area / len(density_per_region)
        total_valume: float = 4/3 * np.pi * div_r**3
        rdf: dict[float, float] = {}
        for region, densities in density_per_region.items():
            if not densities:
                rdf[region] = 0
                continue

            tmp = []
            for j, item in enumerate(densities):
                density: float = \
                    (num_oda[j] * (num_oda[j] - 1)/2) / total_valume
                tmp.append(item * density)
            rdf[region] = np.mean(tmp)
        return rdf

    def _compute_pbc_distance(self,
                              arr: np.ndarray,
                              np_com: np.ndarray,
                              box: np.ndarray
                              ) -> np.ndarray:
        """claculating the distance between the np and the surfactants
        at each frame and return an array
        """
        dx_i = arr[:, 0] - np_com[0]
        dx_pbc = dx_i - (box[0] * np.round(dx_i/box[0]))
        dy_i = arr[:, 1] - np_com[1]
        dy_pbc = dy_i - (box[1] * np.round(dy_i/box[1]))
        dz_i = arr[:, 2] - np_com[2]
        dz_pbc = dz_i - (box[2] * np.round(dz_i/box[2]))
        return np.sqrt(dx_pbc**2 + dy_pbc**2 + dz_pbc**2)

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
