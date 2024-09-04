"""
Plot the potential of the average of the layers of the box.
The potential in each grid of the system that computed by APBS.
"""

import typing

import numpy as np

import matplotlib.pyplot as plt

from common import logger
from common import elsevier_plot_tools
from common.colors_text import TextColor as bcolors


class PlotPotentialLayerConfig:
    """set the name of the input files"""
    # pylint: disable=invalid-name

    oda_concentration: float = 0.003
    ADD_TEXT: bool = True

    @property
    def SINGLE_PLOT(self) -> dict[str, typing.Any]:
        """set the name of the input files"""
        return {
            'color': 'black',
            'linestyle': '-',
            'label': r'$\psi(r^*)$',
            'xlabel': r'r$^*$ [nm]',
            'ylabel': r'Potential ($\psi(r^*)$) [mV]',
        }


class PlotPotentialLayer:
    """
    Plot the potential of the average of the layers of the box.
    The potential in each grid of the system that computed by APBS.
    """
    info_msg: str = 'Message from PlotPotentialLayer:\n'
    configs: PlotPotentialLayerConfig

    def __init__(self,
                 cut_radii: list[np.ndarray],
                 cut_radial_average: list[np.ndarray],
                 radii_list: list[np.ndarray],
                 radial_average_list: list[np.ndarray],
                 sphere_grid_range: np.ndarray,
                 log: logger.logging.Logger,
                 configs: PlotPotentialLayerConfig = PlotPotentialLayerConfig()
                 ) -> None:
        """write and log messages"""
        # pylint: disable=too-many-arguments
        self.configs = configs
        self.plot_potential_layers(cut_radii,
                                   cut_radial_average,
                                   radii_list,
                                   radial_average_list,
                                   sphere_grid_range,
                                   log)
        self.write_msg(log)

    def plot_potential_layers(self,
                              cut_radii: list[np.ndarray],
                              cut_radial_average: list[np.ndarray],
                              radii_list: list[np.ndarray],
                              radial_average_list: list[np.ndarray],
                              sphere_grid_range: np.ndarray,
                              log: logger.logging.Logger
                              ) -> None:
        """plot the potential of the layers"""
        # pylint: disable=unused-variable
        # pylint: disable=unused-argument
        # pylint: disable=too-many-arguments
        fig_i: plt.Figure
        ax_i: plt.Axes
        _config: dict[str, typing.Any] = self.configs.SINGLE_PLOT
        for i, radial_average in enumerate(cut_radial_average):
            fig_i, ax_i = \
                elsevier_plot_tools.mk_canvas('single_column')
            layer: int = sphere_grid_range[i]
            ax_i.plot(radii_list[i] / 10.0,  # Convert to nm
                      radial_average_list[i],
                      color=_config['color'],
                      ls=_config['linestyle'],
                      label=_config['label']
                      )

            ax_i.set_xlabel(_config['xlabel'])
            ax_i.set_ylabel(_config['ylabel'])
            self.add_text(ax_i)
            self.add_text(ax_i,
                          text=f'z index = {layer}',
                          loc=(0.86, 0.9))
            elsevier_plot_tools.remove_mirror_axes(ax_i)
            elsevier_plot_tools.save_close_fig(fig_i,
                                               f'potential_layer_{layer}',
                                               loc='lower left',
                                               )

    def add_text(self,
                 ax_i: plt.Axes,
                 text: str | None = None,
                 loc: tuple[float, float] = (0.84, 1.0)
                 ) -> None:
        """add text to the plot"""
        if text is not None:
            pattern = text
        else:
            pattern = f'{self.configs.oda_concentration} ODA/nm$^2$'
        if self.configs.ADD_TEXT:
            ax_i.text(loc[0],
                      loc[1],
                      pattern,
                      transform=ax_i.transAxes,
                      fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT - 3,
                      verticalalignment='top',
                      horizontalalignment='center',
                      )

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{PlotPotentialLayer.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print(f'{bcolors.WARNING}\n\tThis script is not intended to be run '
          f'independently.{bcolors.ENDC}\n')
