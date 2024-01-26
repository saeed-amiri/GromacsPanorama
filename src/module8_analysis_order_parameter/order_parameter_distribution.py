"""
Order Parameter Analysis Procedure:

The analysis of order parameters in our Python script is structured to
proceed as follows:

Frame-by-Frame Analysis:
    For each frame in the simulation, we will conduct a separate
    analysis.

Residue Type Processing:
    Within each frame, we will categorize and process the data according
    to different residue types.

Identifying Residues in Bins:
    For each residue type, we'll identify the residues that fall into
    a specified bin. This step involves determining which residues'
    indices are within the boundaries of each bin.

Calculating Average Order Parameters:
    Once the relevant residues for each bin and frame are identified,
    we will calculate the average order parameter. This involves
    averaging the order parameters of all selected residues within each
    specific frame and bin.

Visulising the results:
    Also plot the different type of graphs for this distributions

Opt. by ChatGpt
Saeed
26 Jan 2024
"""

import typing
from dataclasses import dataclass, field

import numpy as np

from common import logger
from common.colors_text import TextColor as bcolors

if typing.TYPE_CHECKING:
    from common.com_file_parser import GetCom
    from module8_analysis_order_parameter.order_parameter_pickle_parser \
        import GetOorderParameter


@dataclass
class FileConfig:
    """set the name of the input file"""


@dataclass
class ParameterConfig:
    """set the prameters for the computations"""
    axis: str = 'z'
    residues_for_box_limit_check: list[str] = \
        field(default_factory=lambda: ['SOL', 'D10', 'CLA'])
    nr_bins: int = 50


@dataclass
class AllConfigs(FileConfig, ParameterConfig):
    """set all the inputs and parameters"""


class ComputeOPDistribution:
    """compute the order parameters distribution through on axis"""

    info_msg: str = 'Message from ComputeOPDistribution:\n'
    configs: AllConfigs

    def __init__(self,
                 com_arr: "GetCom",
                 orderp_arr: "GetOorderParameter",
                 log: logger.logging.Logger,
                 configs: AllConfigs = AllConfigs()
                 ) -> None:
        self.configs = configs
        self._initiate_calc(com_arr, orderp_arr, log)
        self._write_msg(log)

    def _initiate_calc(self,
                       com_arr: "GetCom",
                       orderp_arr: "GetOorderParameter",
                       log: logger.logging.Logger
                       ) -> None:
        """find the residues in the bin and calculate the average OP
        for each residue in each bin for all the frames"""
        self.get_bins(com_arr.split_arr_dict)

    def get_bins(self,
                 split_arr_dict: dict[str, np.ndarray]
                 ) -> None:
        """find the low and high of the box
        SOL, D10 and CLA must be checked
        """
        ax_lo: float  # Box lims in the asked direction
        ax_hi: float  # Box lims in the asked direction
        ax_ind: int = self._get_ax_index()
        ax_lo, ax_hi = self._get_box_lims(ax_ind, split_arr_dict)

    def _get_box_lims(self,
                      ax_ind: int,  # Index of the direction
                      split_arr_dict: dict[str, np.ndarray]
                      ) -> tuple[float, float]:
        """
        Find box limits in the asked direction using vectorized
        operations.
        """
        ax_lo: float = np.inf
        ax_hi: float = -np.inf

        for res in self.configs.residues_for_box_limit_check:
            # Concatenate all frames for a given residue type
            all_frames = np.concatenate(split_arr_dict[res][:-2])
            # Reshape and extract the axis of interest
            all_ax_values = all_frames.reshape(-1, 3)[:, ax_ind]

            # Find the min and max along the axis
            ax_lo = np.min((ax_lo, all_ax_values.min()))
            ax_hi = np.max((ax_hi, all_ax_values.max()))
        self.info_msg += ('\tThe min and of max of the box in the direction '
                          f'of `{self.configs.axis}` are {ax_lo:.3f} and '
                          f'`{ax_hi:.3f}`\n')
        return ax_lo, ax_hi

    def _get_ax_index(self) -> int:
        """set the index of the computation direction"""
        if (axis := self.configs.axis) == 'z':
            return 2
        if axis == 'y':
            return 1
        return 0

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ComputeOPDistribution.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print(f'{bcolors.OKGREEN}This script is calling within '
          f'`order_parameter_analysis.py`\n{bcolors.ENDC}')
