"""
Plot a heat map of all the order parameters for a given system.
All the points of the heads in all frames will be plotted in a heat map.
and the their order parameter will be the color of the point.
a the end there will be three plots, one for each axis.

"""

import numpy as np
import matplotlib.pyplot as plt

from common import logger
from common.colors_text import TextColor as bcolors

from module8_analysis_order_parameter.config_classes_trr import AllConfig


class OrderParameterHeatMap:
    """Plot a heat map of all the order parameters for a given system"""

    info_msg: str = 'Message from OrderParameterHeatMap:\n'
    configs: AllConfig
    order_parameter_frames: dict[int, np.ndarray]
    tail_with_angle: list[np.ndarray]

    def __init__(self,
                 tail_with_angle: list[np.ndarray],
                 order_parameter_frames: dict[int, np.ndarray],
                 configs: AllConfig,
                 log: logger.logging.Logger
                 ) -> None:
        self.configs = configs
        self.order_parameter_frames = order_parameter_frames
        self.tail_with_angle = tail_with_angle
        self.plot_heat_map()
        self.info_msg += '\tThe heat map of order parameters is plotted\n'
        self.write_msg(log)

    def plot_heat_map(self) -> None:
        """Plot the heat map of order parameters"""
        order_parameters: np.ndarray = self.stack_dict_arrays()
        positions: np.ndarray = self.stack_list_arrays()

        interface_oda: np.ndarray = self.get_interface_oda(positions)
        _, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, ax_i in enumerate(axs):
            data: np.ndarray = np.hstack(
                (interface_oda, order_parameters[:, i, np.newaxis]*i))
            ax_i.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='hot')
            ax_i.set_title(f'Order Parameter for Axis {i}')
            del data

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        plt.savefig('heat_map_file.png', dpi=300, bbox_inches='tight')
        plt.close()

    def get_interface_oda(self,
                          positions: np.ndarray,
                          ) -> None:
        """Make the data for the heat map"""
        return positions[:, :3]

    def stack_dict_arrays(self) -> np.ndarray:
        """Stack all the arrays in the dictionary"""
        return np.vstack(list(self.order_parameter_frames.values()))[:, :3]

    def stack_list_arrays(self) -> np.ndarray:
        """Stack all the arrays in the list"""
        return np.vstack(self.tail_with_angle)

    def plot_heat_map_axis(self,
                           ax_i: plt.axes,
                           data: np.ndarray
                           ) -> None:
        """Plot the heat map of order parameters for a given axis"""
        print(ax_i, data)
        ax_i.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='hot')

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{OrderParameterHeatMap.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
