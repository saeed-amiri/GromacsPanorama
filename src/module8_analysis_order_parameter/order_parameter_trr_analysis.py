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
___________________________________________
The equation for the order parameter is:
    S = 1/2 * <3cos^2(theta) - 1>
where theta is the angle between the vector and the z-axis.
it also can be computed along the other axis, for example, the y-axis.
The angle between the vector and the z-axis can be computed by the dot
product of the two vectors.
The dot product of two vectors is:
    a.b = |a| |b| cos(theta)
where a and b are two vectors, and theta is the angle between them.
The dot product of two unit vectors is:
    a.b = cos(theta)
where a and b are two unit vectors, and theta is the angle between them.

The angles between the vectors and all the three axes will be computed
and save into the array with the coordintated of the head atoms, thus,
the the final array will be a 2D array with index of the residues and
their tail coordinates and the angles along each axis:
    [residue_index, x, y, z, angle_x, angle_y, angle_z]
this array will be computed for each frame and saved as a list.

Afterwards, the further analysis will be done on the list of the angles
to compute the order parameter:
    - Compute the average order parameter for each frame
    - Compute the average order parameter for the whole trajectory
    - Compute the oreder parameter along one axis in different bins to
        see the distribution of the order parameter along the axis.
also plot:
    - The distribution of the order parameter along the axis
    - The superposion of all the order parameter along the axis in a
        contour plot of x-y plane and order parameter as the c-bar.
and save the data to the xvg file.
15 April 2024
"""

import sys

import numpy as np

import MDAnalysis as mda

from common.colors_text import TextColor as bcolors
from common import logger, xvg_to_dataframe, my_tools, cpuconfig

from module8_analysis_order_parameter.config_classes_trr import AllConfig
from module8_analysis_order_parameter.angle_computation_trr import \
    AngleProjectionComputation
from module8_analysis_order_parameter.surfactant_oreder_parameter import \
    AnalysisSurfactantOrderParameter


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
        self.get_number_of_cores(log)
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

    def get_number_of_cores(self,
                            log: logger.logging.Logger
                            ) -> None:
        """Get the number of cores for multiprocessing"""
        self.configs.n_cores = cpuconfig.ConfigCpuNr(log)

    def compute_order_parameter(self,
                                log: logger.logging.Logger
                                ) -> None:
        """Compute the order parameter"""
        tail_with_angle: list[np.ndarray] = AngleProjectionComputation(
            log, self.universe, self.configs).tail_with_angle
        OrderParameterAnalaysis(tail_with_angle, self.configs, log)

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


class OrderParameterAnalaysis:
    """ Analysis the order parameter with tails_with_angles
    computed from the AngleProjectionComputation
    """
    info_msg: str = 'Message from OrderParameterAnalaysis:\n'
    configs: AllConfig

    def __init__(self,
                 tail_with_angle: list[np.ndarray],
                 configs: AllConfig,
                 log: logger.logging.Logger
                 ) -> None:
        self.configs = configs
        self.analyze_order_parameter(tail_with_angle, log)

    def analyze_order_parameter(self,
                                tail_with_angle: list[np.ndarray],
                                log: logger.logging.Logger
                                ) -> None:
        """Analyze the order parameter based on the residue
        For the surfactant molecules, the order parameter of interest
        is along the z-axis, and the interface location
        For Oil and water molecules, the order parameter of interest
        is along the z-axis, but the changes from bulk to interface
        is more important and desireabel.
        """
        # get the type of the residue
        selected_res: str = self.configs.selected_res.name
        if selected_res == 'SURFACTANT':
            self.analysis_order_parameter_surfactant(tail_with_angle, log)

    def analysis_order_parameter_surfactant(self,
                                            tail_with_angle: list[np.ndarray],
                                            log: logger.logging.Logger
                                            ) -> None:
        """Analysis the order parameter for the surfactant molecules"""
        AnalysisSurfactantOrderParameter(tail_with_angle, self.configs, log)


if __name__ == '__main__':
    OrderParameter(sys.argv[1],
                   logger.setup_logger('order_parameter_mda.log'))
