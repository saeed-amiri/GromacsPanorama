"""
Computing the order parameter from the trajectory file.
The script reads the trajectory file frame by frame with MDAnalysis
and computes the order parameter for them.
The script computes the order paramters:
    - For water and oil molecules, it should be able to compute along
        the z-axis.
    - For the surfactant molecules, it should be able to compute along
        the y-axis, also only at the main interface; by the main
        interface, we mean the interface between the water and oil
        molecules where the must of surfactant molecules are located,
        and in case there is a nanoparticle in the system, the
        interface where the nanoparticle is located.

The steps will be as follows:
    - Read the trajectory file
    - Read the topology file
    - Select the atoms for the order parameter calculation
    - Compute the order parameter
    - Save the order parameter to a xvg file

Since we already have computed the interface location, we can use that
to compute the order parameter only at the interface.

input files:
    - trajectory file (centered on the NP if there is an NP in the system)
    - topology file
    - interface location file (contact.xvg)
output files:
    - order parameter xvg file along the z-axis
    - in the case of the surfactant molecules, we save a data file
        based on the grids of the interface, looking at the order
        parameter the value and save the average value of the order
        and contour plot of the order parameter.
        This will help us see the order parameter distribution in the
        case of an NP and see how the order parameter changes around
        the NP.
The prodidure is very similar to the computation of the rdf computation
with cosidering the actual volume in:
    read_volume_rdf_mdanalysis.py

12 April 2024
Saeed
Opt by VSCode CoPilot
"""

import os
import sys
import typing
from enum import Enum
from dataclasses import dataclass, field

import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MDAnalysis as mda

from common import logger, xvg_to_dataframe, my_tools, cpuconfig
from common.colors_text import TextColor as bcolors


class ResidueName(Enum):
    """Residue names for the residues in the trajectory"""
    # pylint: disable=invalid-name
    OIL = 'D10'
    WATER = 'SOL'
    SURFACTANT = 'ODN'


@dataclass
class OrderParameterConfig:
    """Order parameter dataclass"""
    resideu_name: typing.Union[None, ResidueName] = field(init=False)
    atom_selection: typing.Union[None, str, list[str]] = field(init=False)
    order_parameter_avg: float = field(init=False)
    order_parameter_std: float = field(init=False)
    order_parameter_data: np.ndarray = field(init=False)


@dataclass
class InterfaceConfig:
    """Interface configuration dataclass"""
    interface_location: float = 0.0
    interface_location_std: float = 0.0
    interface_location_data: np.ndarray = np.array([])


@dataclass
class InputFiles:
    """Input files dataclass"""
    box_file: str = 'box.xvg'
    tpr_file: str = field(init=False)
    trajectory_file: str = field(init=False)
    interface_location_file: str = 'contact.xvg'
    path_name: str = '/scratch/saeed/GÖHBP/PRE_DFG_7May24/single_np/15Oda/data'

    def __post_init__(self) -> None:
        """Post init function"""
        self.interface_location_file = \
            os.path.join(self.path_name, self.interface_location_file)
        self.box_file = os.path.join(self.path_name, self.box_file)


@dataclass
class ResiduesTails:
    """Residues tails atoms to compute the order parameter
    For water use average location of the hydrogen atoms as one tail
    """
    # pylint: disable=invalid-name
    OIL: dict[str, typing.Union[str, list[str]]] = \
        field(default_factory=lambda: ({'head': 'C1',
                                        'tail': 'C9'}))
    WATER: dict[str, typing.Union[str, list[str]]] = \
        field(default_factory=lambda: ({'head': 'OH2',
                                        'tail': ['H1', 'H2']}))
    SURFACTANT: dict[str, typing.Union[str, list[str]]] = \
        field(default_factory=lambda: ({'head': 'C1',
                                        'tail': 'NH2'}))


@dataclass
class AllConfig(ResiduesTails):
    """Order parameter options dataclass"""
    residues_tails: ResiduesTails = field(default_factory=ResiduesTails)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    order_parameter: OrderParameterConfig = \
        field(default_factory=OrderParameterConfig)
    input_files: InputFiles = field(default_factory=InputFiles)

    # The residue names to compute the order parameter
    # use the ResidueName enum names
    selected_res: ResidueName = ResidueName.SURFACTANT


class OrderParameter:
    """Order parameter computation"""

    info_msg: str = 'Message from OrderParameter:\n'
    configs: AllConfig
    box: np.ndarray
    universe: "mda.coordinates.TRR.TRRReader"

    def __init__(self,
                 fname: str,  # trajectory file,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.configs.input_files.trajectory_file = fname
        self.initate(log)
        self.compute_order_parameter(log)
        self.write_msg(log)

    def initate(self,
                log: logger.logging.Logger
                ) -> None:
        """Initiate the order parameter computation"""
        self.read_xvg_files(log)
        self.set_tpr_fname(log)
        self.intiate_order_parameter()
        self.universe = self.load_trajectory(log)

    def compute_order_parameter(self,
                                log: logger.logging.Logger
                                ) -> None:
        """Compute the order parameter"""
        OrderParameterComputation(log, self.universe, self.configs)

    def read_xvg_files(self,
                       log: logger.logging.Logger
                       ) -> None:
        """Read the xvg files"""
        self._read_interface_location(log)
        self._read_box_file(log)

    def _read_interface_location(self,
                                 log: logger.logging.Logger
                                 ) -> None:
        """Read the interface location file"""
        self.configs.interface.interface_location_data = \
            xvg_to_dataframe.XvgParser(
                self.configs.input_files.interface_location_file,
                log).xvg_df['interface_z'].to_numpy()
        self.configs.interface.interface_location = \
            np.mean(self.configs.interface.interface_location_data)
        self.configs.interface.interface_location_std = \
            np.std(self.configs.interface.interface_location_data)
        self.info_msg += \
            (f'\tInterface location:'
             f'`{self.configs.interface.interface_location:.3f}` '
             f'+/- `{self.configs.interface.interface_location_std:.3f}`\n')

    def _read_box_file(self,
                       log: logger.logging.Logger
                       ) -> None:
        """Read the box file"""
        box_data = xvg_to_dataframe.XvgParser(
            self.configs.input_files.box_file, log).xvg_df
        self.box = np.array([box_data['XX'].to_numpy(),
                             box_data['YY'].to_numpy(),
                             box_data['ZZ'].to_numpy()])

    def set_tpr_fname(self,
                      log: logger.logging.Logger
                      ) -> None:
        """Read the trajectory file"""
        self.configs.input_files.tpr_file = my_tools.get_tpr_fname(
            self.configs.input_files.trajectory_file, log)

    def load_trajectory(self,
                        log: logger.logging.Logger
                        ) -> "mda.coordinates.TRR.TRRReader":
        """read the input file"""
        my_tools.check_file_exist(self.configs.input_files.trajectory_file,
                                  log,
                                  if_exit=True)
        try:
            return mda.Universe(self.configs.input_files.tpr_file,
                                self.configs.input_files.trajectory_file)
        except ValueError as err:
            log.error(msg := '\tThe input file is not correct!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}\n\t{err}\n')

    def intiate_order_parameter(self) -> None:
        """Initiate the order parameter"""
        self.configs.order_parameter.resideu_name = None
        self.configs.order_parameter.atom_selection = None
        self.configs.order_parameter.order_parameter_avg = 0.0
        self.configs.order_parameter.order_parameter_std = 0.0
        self.configs.order_parameter.order_parameter_data = np.array([])

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class OrderParameterComputation:
    """doing the math here"""

    info_msg: str = 'Message from OrderParameterComputation:\n'
    configs: AllConfig

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.write_msg(log)

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    OrderParameter(sys.argv[1],
                   logger.setup_logger('order_parameter_mda.log'))
