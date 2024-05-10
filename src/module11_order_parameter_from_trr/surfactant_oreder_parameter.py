"""
# surfactant_order_parameter.py
This module contains the `AnalysisSurfactantOrderParameter` class which
is used for analyzing the order parameter of surfactant molecules in a
molecular dynamics simulation.

## Class: AnalysisSurfactantOrderParameter
### Attributes:
- `configs`: An instance of `AllConfig` class which holds all the
    configuration data.
- `tail_with_angle`: A list of numpy arrays that store the tail
    coordinates and their angle projections along all axes.
- `info_msg`: A string used to store messages from the
    `AnalysisSurfactantOrderParameter` class.

### Methods:
- `__init__(self, tail_with_angle, configs, log)`: Initializes the
    `AnalysisSurfactantOrderParameter` class with tail coordinates
    with angle projections, configurations, and a logger.
- `_compute_order_parameter(self, log)`: Computes the order parameter
    and writes the average order parameter to an xvg file.
- `compute_order_parameter_frames(self)`: Computes the order parameter
    for each frame and returns a dictionary where the keys are the
    frame numbers and the values are the order parameters.
- `compute_order_parameter(self, angle_frame)`: Computes the order
    parameter for a given angle frame and returns a numpy array of
    order parameters.
- `compute_order_parameter_for_a_residue(self, angle_res_i)`: Computes
    the order parameter for a single angle and returns the order
    parameter.
- `compute_frame_avg_order_parameter(self, order_parameter_frames)`:
    Computes the average order parameter for each frame and returns a
    dictionary where the keys are the frame numbers and the values are
    the average order parameters.

## Usage:
This module is used for analyzing the order parameter of surfactant
molecules in a molecular dynamics simulation. The
`AnalysisSurfactantOrderParameter` class is initialized with tail
coordinates with angle projections, configurations, and a logger.
The order parameter is computed using the `_compute_order_parameter`
method. The order parameter for each frame is computed using the
`compute_order_parameter_frames` method, and the average order parameter
for each frame is computed using the `compute_frame_avg_order_parameter`
method.
"""

import numpy as np
import pandas as pd

from common import logger
from common import my_tools
from common.colors_text import TextColor as bcolors

from module8_analysis_order_parameter.config_classes_trr import AllConfig
from module8_analysis_order_parameter.order_parameter_heat_map import \
    OrderParameterHeatMap


class AnalysisSurfactantOrderParameter:
    """Analysis the surfactant order parameter"""
    info_msg: str = 'Message from AnalysisSurfactantOrderParameter:\n'

    def __init__(self,
                 tail_with_angle: list[np.ndarray],
                 configs: AllConfig,
                 log: logger.logging.Logger
                 ) -> None:
        self.configs = configs
        self.tail_with_angle = tail_with_angle
        self._compute_order_parameter(log)
        self.write_msg(log)

    def _compute_order_parameter(self,
                                 log: logger.logging.Logger
                                 ) -> None:
        """Compute the order parameter"""
        order_parameter_frames: dict[int, np.ndarray] = \
            self.compute_order_parameter_frames()
        avg_order_parameter_frames: dict[int, np.ndarray] = \
            self.compute_frame_avg_order_parameter(order_parameter_frames)
        self.write_avg_xvg(avg_order_parameter_frames, log)
        OrderParameterHeatMap(self.tail_with_angle,
                              order_parameter_frames,
                              self.configs,
                              log)

    def compute_order_parameter_frames(self) -> dict[int, np.ndarray]:
        """Compute the order parameter for each frame"""
        # Compute the order parameter for each frame
        order_parameter_frames: dict[int, np.ndarray] = {}
        for frame, tail_angle in enumerate(self.tail_with_angle):
            angle_frame: np.ndarray = tail_angle[:, -3:]
            order_parameter_frames[frame] = \
                self.compute_order_parameter(angle_frame)
        return order_parameter_frames

    def compute_order_parameter(self,
                                angle_frame: np.ndarray
                                ) -> np.ndarray:
        """Compute the order parameter"""
        order_parameter: np.ndarray = np.zeros(angle_frame.shape)
        for i, angle_res_i in enumerate(angle_frame):
            order_parameter[i] += \
                self.compute_order_parameter_for_a_residue(angle_res_i)
        return order_parameter

    def compute_order_parameter_for_a_residue(self,
                                              angle_res_i: np.ndarray
                                              ) -> np.ndarray:
        """Compute the order parameter for a single residue"""
        return 0.5 * (3 * np.cos(angle_res_i)**2 - 1)

    def compute_frame_avg_order_parameter(self,
                                          order_parameter_frames:
                                          dict[int, np.ndarray]
                                          ) -> dict[int, np.ndarray]:
        """Compute the average order parameter for each frame"""
        avg_order_parameter_frames: dict[int, np.ndarray] = {}
        for frame, order_parameter_frame in order_parameter_frames.items():
            avg_order_parameter_frames[frame] = \
                np.mean(order_parameter_frame, axis=0)
        return avg_order_parameter_frames

    def write_avg_xvg(self,
                      avg_order_parameter_frames: dict[int, np.ndarray],
                      log
                      ) -> None:
        """Write the average order parameter to the xvg file"""
        avg_df: pd.DataFrame = self.make_avg_df(avg_order_parameter_frames)
        extra_msg: list[str] = self.make_extra_msg(avg_df)

        residue_name: str = self.configs.selected_res.name
        fname: str = f'order_parameter_{residue_name}.xvg'

        my_tools.write_xvg(df_i=avg_df,
                           log=log,
                           extra_msg=extra_msg,
                           fname=fname,
                           write_index=True,
                           x_axis_label='Frame index',
                           y_axis_label='Order parameter',
                           title='Order parameter')
        self.info_msg += \
            f'\tThe average order parameter is written to {fname}\n'

    def make_avg_df(self,
                    avg_order_parameter_frames: dict[int, np.ndarray]
                    ) -> pd.DataFrame:
        """Make the average dataframe"""
        columns: list[str] = ['order_parameter_x',
                              'order_parameter_y',
                              'order_parameter_z']
        avg_df: pd.DataFrame = pd.DataFrame.from_dict(
            avg_order_parameter_frames, columns=columns, orient='index')
        return avg_df

    def make_extra_msg(self,
                       avg_df: pd.DataFrame
                       ) -> list[str]:
        """Make the extra message
        add # at the beginning of each line
        """
        avg_df_mean = avg_df.mean(axis=0)
        avg_msg: str = '\t'.join([f'{avg}' for avg in avg_df_mean])
        extra_msg: list[str] = [
            f'# Interface location: '
            f'{self.configs.interface.interface_location:.3f} +/- '
            f'{self.configs.interface.interface_location_std:.3f}',
            f'# Mean order parameter: (x, y, z) = \n# {avg_msg}']
        return extra_msg

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
