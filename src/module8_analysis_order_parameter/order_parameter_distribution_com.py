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
import pandas as pd
import matplotlib.pylab as plt

from common import logger, xvg_to_dataframe
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
    bin_from_file: bool = True
    bin_file_name: str = 'sol_density_trr.xvg'
    bin_file_column: str = 'Coordinate_nm'
    axis: str = 'z'
    residues_for_box_limit_check: list[str] = \
        field(default_factory=lambda: ['SOL', 'D10', 'CLA'])
    residues_for_compute_order_parameter: list[str] = \
        field(default_factory=lambda: ['SOL', 'D10', 'ODN'])
    nr_bins: int = 50


@dataclass
class AllConfigs(FileConfig, ParameterConfig):
    """set all the inputs and parameters"""


class ComputeOPDistribution:
    """compute the order parameters distribution through on axis"""

    info_msg: str = 'Message from ComputeOPDistribution:\n'
    configs: AllConfigs
    ax_ind: int  # Integer to set the dirction (change via configs)

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
        bins: np.ndarray = self.get_bins(com_arr.split_arr_dict, log)
        self.compute_orderp_mean(
            bins, com_arr.split_arr_dict, orderp_arr.split_arr_dict)

    def get_bins(self,
                 com_split_arr_dict: dict[str, np.ndarray],
                 log: logger.logging.Logger
                 ) -> np.ndarray:
        """
        If read bins from a density file from gromacs, just read the
        file and return the bins as an array, otherwise find the low
        and high of the box SOL, D10 and CLA must be checked.
        """
        # pylint: disable='broad-exception-caught'
        self.ax_ind: int = self._get_ax_index()
        if self.configs.bin_from_file:
            try:
                return self._bins_from_xvg(log)
            except FileNotFoundError:
                self.info_msg += (
                    '\tWarning: There is error in reading bins from the file\n'
                    '\t\tFalling back to compute the bins from com_pickle\n')
            except Exception as err:
                self.info_msg += \
                    f'\tProblem: `{err}` in getting bins from file\n'
        return self._compute_bins(com_split_arr_dict)

    def compute_orderp_mean(self,
                            bins: np.ndarray,
                            com_split_arr: dict[str, np.ndarray],
                            op_split_arr: dict[str, np.ndarray]
                            ) -> None:
        """get indices of each residues in bins for all the frames"""
        com: np.ndarray
        orderp: np.ndarray
        frames_sol: dict[int, list[float]] = \
            {ind: [] for ind in range(len(bins)-1)}
        frames_d10: dict[int, list[float]] = \
            {ind: [] for ind in range(len(bins)-1)}
        frames_odn: dict[int, list[float]] = \
            {ind: [] for ind in range(len(bins)-1)}

        for res in self.configs.residues_for_compute_order_parameter:
            for com, orderp in zip(com_split_arr[res][:-2],
                                   op_split_arr[res][:-2]):
                reshaped_com = com.reshape(-1, 3)
                reshaped_oprderp = orderp.reshape(-1, 3)
                axis_positions = reshaped_com[:, self.ax_ind]
                axis_orderp = reshaped_oprderp[:, self.ax_ind]
                for bin_i in range(len(bins) - 1):
                    indices = np.where((axis_positions >= bins[bin_i]) &
                                       (axis_positions < bins[bin_i+1]))[0]
                    if indices.size > 0:
                        mean_orderp = np.mean(axis_orderp[indices])
                        if res == 'SOL':
                            frames_sol[bin_i].append(mean_orderp)
                        elif res == 'D10':
                            frames_d10[bin_i].append(mean_orderp)
                        elif res == 'ODN':
                            frames_odn[bin_i].append(mean_orderp)
                    else:
                        mean_orderp = 0  # Handle empty bin

        print('frames_sol', [np.mean(item) for item in frames_sol.values()])
        print('frames_d10', [np.mean(item) for item in frames_d10.values()])
        print('frames_odn', [np.mean(item) for item in frames_odn.values()])
        plt.plot([np.mean(item) for item in frames_sol.values()])
        plt.plot([np.mean(item) for item in frames_d10.values()])
        plt.plot([np.mean(item) for item in frames_odn.values()])
        plt.show()

    def _compute_bins(self,
                      com_split_arr_dict: dict[str, np.ndarray]
                      ) -> np.ndarray:
        """compute the bins from the com_pickle"""
        ax_lo: float  # Box lims in the asked direction
        ax_hi: float  # Box lims in the asked direction
        ax_lo, ax_hi = self._get_box_lims(com_split_arr_dict)
        return np.linspace(ax_lo, ax_hi, self.configs.nr_bins + 1)

    def _bins_from_xvg(self,
                       log: logger.logging.Logger
                       ) -> np.ndarray:
        """read the density xvg file and return the bins
        since the gromacs default output are in nanometer here I
        convert them to Angastrom.
        """
        self.info_msg += \
            f'\tReading `{self.configs.bin_file_name}` to get bins\n'
        xvg_df: pd.DataFrame = \
            xvg_to_dataframe.XvgParser(fname=self.configs.bin_file_name,
                                       log=log,
                                       x_type=float,
                                       if_exit=False).xvg_df
        return xvg_df[self.configs.bin_file_column].to_numpy()*10

    def _get_box_lims(self,
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
            all_ax_values = all_frames.reshape(-1, 3)[:, self.ax_ind]

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
