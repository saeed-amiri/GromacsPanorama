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

    @property
    def MULTI_LAYERS(self) -> dict[str, typing.Any]:
        """set the name of the input files"""
        return {
            'indices': [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
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
        self.plot_potentials(cut_radii,
                             cut_radial_average,
                             radii_list,
                             radial_average_list,
                             sphere_grid_range,
                             )
        self.write_msg(log)

    def plot_potentials(self,
                        cut_radii: list[np.ndarray],
                        cut_radial_average: list[np.ndarray],
                        radii_list: list[np.ndarray],
                        radial_average_list: list[np.ndarray],
                        sphere_grid_range: np.ndarray,
                        ) -> None:
        """plot the potentials"""
        # pylint: disable=too-many-arguments
        self.plot_potential_layers(radii_list,
                                   radial_average_list,
                                   sphere_grid_range)
        self.plot_multi_layers(cut_radii,
                               cut_radial_average,
                               radii_list,
                               radial_average_list,
                               sphere_grid_range)

    def plot_potential_layers(self,
                              radii_list: list[np.ndarray],
                              radial_average_list: list[np.ndarray],
                              sphere_grid_range: np.ndarray,
                              ) -> None:
        """plot the potential of the layers"""
        fig_i: plt.Figure
        ax_i: plt.Axes
        _config: dict[str, typing.Any] = self.configs.SINGLE_PLOT
        for i, radial_average in enumerate(radial_average_list):
            fig_i, ax_i = \
                elsevier_plot_tools.mk_canvas('single_column')
            layer: int = sphere_grid_range[i]
            ax_i.plot(radii_list[i] / 10.0,  # Convert to nm
                      radial_average,
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

    def plot_multi_layers(self,
                          cut_radii: list[np.ndarray],
                          cut_radial_average: list[np.ndarray],
                          radii_list: list[np.ndarray],
                          radial_average_list: list[np.ndarray],
                          sphere_grid_range: np.ndarray,
                          ) -> None:
        """plot the potentials of the layers"""
        # pylint: disable=too-many-arguments
        self.plot_multi_layers_whole(
            radii_list, radial_average_list, sphere_grid_range)
        self.plot_multi_layers_cut(
            cut_radii, cut_radial_average, sphere_grid_range)

    def plot_multi_layers_whole(self,
                                radii_list: list[np.ndarray],
                                radial_average_list: list[np.ndarray],
                                sphere_grid_range: np.ndarray,
                                ) -> None:
        """plot the potentials of the layers"""
        fig_i: plt.Figure
        ax_i: plt.Axes
        _config: dict[str, typing.Any] = self.configs.MULTI_LAYERS
        fig_i, ax_i = elsevier_plot_tools.mk_canvas('double_column')
        for i, layer in enumerate(_config['indices']):
            ind: int = sphere_grid_range.tolist().index(layer)
            ax_i.plot(radii_list[ind] / 10.0,  # Convert to nm
                      radial_average_list[ind],
                      label=f'z index = {layer}',
                      ls=elsevier_plot_tools.LINESTYLE_TUPLE[i][1],
                      color=elsevier_plot_tools.DARK_RGB_COLOR_GRADIENT[i],
                      )
        ax_i.set_xlabel(_config['xlabel'])
        ax_i.set_ylabel(_config['ylabel'])
        self.add_text(ax_i, loc=(0.75, 1.0))
        elsevier_plot_tools.remove_mirror_axes(ax_i)
        elsevier_plot_tools.save_close_fig(fig_i,
                                           'potential_layers_whole',
                                           loc='upper right',
                                           )

    def plot_multi_layers_cut(self,
                              cut_radii: list[np.ndarray],
                              cut_radial_average: list[np.ndarray],
                              sphere_grid_range: np.ndarray,
                              ) -> None:
        """plot the potentials of the layers"""
        plt.rcParams.update(
            {'font.size': elsevier_plot_tools.LABEL_FONT_SIZE_PT})
        fig_i: plt.Figure
        ax_i: plt.Axes
        _config: dict[str, typing.Any] = self.configs.MULTI_LAYERS
        fig_i, ax_i = elsevier_plot_tools.mk_canvas('double_column')
        for i, layer in enumerate(_config['indices']):
            ind: int = sphere_grid_range.tolist().index(layer)
            ax_i.plot(cut_radii[ind] / 10.0,  # Convert to nm
                      cut_radial_average[ind],
                      label=f'z index = {layer}',
                      ls=elsevier_plot_tools.LINESTYLE_TUPLE[i][1],
                      color=elsevier_plot_tools.DARK_RGB_COLOR_GRADIENT[i],
                      )
        ax_i.set_xlabel(_config['xlabel'])
        ax_i.set_ylabel(_config['ylabel'])
        self.add_text(ax_i, loc=(0.75, 1.0))
        elsevier_plot_tools.remove_mirror_axes(ax_i)
        elsevier_plot_tools.save_close_fig(fig_i,
                                           'potential_layers_cut',
                                           loc='upper right',
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
