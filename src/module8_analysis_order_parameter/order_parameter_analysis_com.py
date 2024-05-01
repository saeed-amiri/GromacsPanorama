"""
Analyzing the Order Parameter Data

This documentation outlines the process of analyzing order parameter
data using two Python modules: `order_parameter_pickle_parser.py` and
`common.com_file_parser.py.`

Data Reading:
    The `order_parameter_pickle_parser.py` reads the order parameter data,
    which is a dictionary of np.ndarray.
    The `common.com_file_parser.py` reads the center of mass data, also
    a dictionary of np.ndarray.

Data Structure and Keys:
    Both datasets have similar keys and structures, with a minor
    difference in the center of mass (COM) file.
    The keys of the dictionaries represent different residues:
        1: SOL (Water)
        2: CLA (Chloride ions)
        3: POT (Potassium ions)
        4: ODN (Octadecylamine: ODA)
        5: D10 (Decane)
    In the COM data, key 0 is reserved for the ODN amino group.

Array Layout:
    COM File:
        Layout: | Time | NP_x | NP_y | NP_z | Res1_x | Res1_y | Res1_z
        | ... | ResN_x | ResN_y | ResN_z | Odn1_x | Odn1_y | Odn1_z |
        ... | OdnN_z |
    Order Parameter File:
        Layout: | Time | NP_Sx | NP_Sy | NP_Sz | Res1_Sx | Res1_Sy |
        Res1_Sz | ... | ResN_Sx | ResN_Sy | ResN_Sz |
        'S' denotes the order parameter.

Special Considerations:
    The order parameter for ions (CLA and POT) is set to zero, as
    defining an order parameter for single-atom residues is not
    meaningful.
    The values for the nanoparticle (NP) are also set to zero.
    This approach is chosen for ease of data manipulation and alignment
    between the two datasets.

Purpose of COM Data:
    The COM data is essential for determining the center of mass of
    each residue.
    This information is valuable for further analysis, particularly in
    understanding the spatial distribution and orientation of
    molecules in the system.

Opt. ChatGPT
Author: Saeed
Date: 25 January 2024
"""

from dataclasses import dataclass, field

from common import logger
from common.com_file_parser import GetCom
from common.colors_text import TextColor as bcolors
from module8_analysis_order_parameter.order_parameter_pickle_parser import \
    GetOorderParameter
from module8_analysis_order_parameter.order_parameter_distribution import \
    ComputeOPDistribution


@dataclass
class FileConfig:
    """set the names of the input files"""
    com_fname: str = 'com_pickle'
    orderp_fname: str = 'order_parameter_pickle'


@dataclass
class OPDistributionConfig:
    """parameters for the computing OP (order parameter) distribution
    of along one axis"""
    axis: str = 'z'
    nr_bins: int = 50


@dataclass
class AllConfig(FileConfig):
    """set all the configurations"""
    ditribution_config: OPDistributionConfig = \
        field(default_factory=OPDistributionConfig)


class AnalysisOrderParameter:
    """analysing the order parmeter by looking at their values spatially
    and ...
    """

    info_msg: str = 'Message from AnalysisOrderParameter:\n'
    configs: AllConfig
    com_data: GetCom
    orderp_data: GetOorderParameter

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.com_data, self.orderp_data = self.initiate_data(log)
        self.initiate_computation(log)
        self.write_msg(log)

    def initiate_data(self,
                      log: logger.logging.Logger
                      ) -> tuple[GetCom, GetOorderParameter]:
        """initiate getting and setting data"""
        com_data = GetCom(fname=self.configs.com_fname)
        orderp_data = GetOorderParameter(log=log)
        return com_data, orderp_data

    def initiate_computation(self,
                             log: logger.logging.Logger
                             ) -> None:
        """doing the calculations:
        finding the residues at  an intersted coordinates and using their
        index to get the order parameter tensors
        """
        self.compute_order_parameter_distribution(log)

    def compute_order_parameter_distribution(self,
                                             log: logger.logging.Logger
                                             ) -> None:
        """
        Segment the Simulation Box: Divide the simulation box along
            the chosen axis into small segments or bins. The number of
            bins depends on the level of granularity needs and the size
            of the simulation box.

        Calculate the Order Parameter for Each Bin: For each segment
            or bin, calculate the average order parameter of the
            particles within that bin.
        """
        ComputeOPDistribution(self.com_data, self.orderp_data, log)

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{AnalysisOrderParameter.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    AnalysisOrderParameter(log=logger.setup_logger("orderp_analysis.log"))
