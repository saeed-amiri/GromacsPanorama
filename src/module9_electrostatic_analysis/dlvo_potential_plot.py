"""
Plot the electrostatic potential
"""

from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec

from common import logger, elsevier_plot_tools, xvg_to_dataframe
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_configs import \
    AllConfig, PlotConfig


class PlotPotential:
    """Plot the electrostatic potential"""

    info_msg: str = 'Message from PlotPotential:\n'

    def __init__(self,
                 radii: np.ndarray,
                 phi_r: np.ndarray,
                 debye_l: float,
                 configs: AllConfig,
                 log: logger.logging.Logger
                 ) -> None:
        # pylint: disable=too-many-arguments
        self.configs = configs
        self.plot_potential(radii, phi_r, debye_l, log)
        self.write_msg(log)

    def plot_potential(self,
                       radii: np.ndarray,
                       phi_r: np.ndarray,
                       debye_l: float,
                       log: logger.logging.Logger
                       ) -> None:
        """plot and save the electostatic potential"""
        configs: PlotConfig = self.configs.plot_config

        axs: plt.axes
        fig_i: plt.figure
        fig_i, axs = elsevier_plot_tools.mk_canvas_multi('double_column',
                                                         n_rows=1,
                                                         n_cols=7)
        for ax_i in axs:
            ax_i.axis('off')
        grid_panel = gridspec.GridSpec(1, 7, figure=fig_i)

        axs[0] = fig_i.add_subplot(grid_panel[0, :2])
        axs[1] = fig_i.add_subplot(grid_panel[0, 2:5])
        axs[2] = fig_i.add_subplot(grid_panel[0, 5:])
        self.plot_panel_a(axs[0], configs)
        self.plot_panel_b(axs[1], radii, phi_r, debye_l, configs, log)
        self.plot_panel_c(axs[2], configs)
        self._save_fig(fig_i, configs)

    def plot_panel_a(self,
                     ax_i: plt.axes,
                     configs: PlotConfig
                     ) -> None:
        """add shcem of the dlvo model for a sphere"""
        ax_i.axis('off')
        ax_i.imshow(plt.imread(configs.scheme_fig_path))
        ax_i.text(0.09,
                  1,
                  'a)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
        self.info_msg += f'\tScheme image: {configs.scheme_fig_path}\n'

    def plot_panel_b(self,
                     ax_i: plt.axes,
                     radii: np.ndarray,
                     phi_r: np.ndarray,
                     debye_l: float,
                     configs: PlotConfig,
                     log: logger.logging.Logger
                     ) -> None:
        """plot the data"""
        # pylint: disable=too-many-arguments
        phi_mv: np.ndarray = phi_r * configs.voltage_to_mV
        # kappa * radius of the np
        kappa_r: float = \
            self.configs.np_radius / (debye_l * configs.angstrom_to_nm)
        self._plot_data(ax_i, radii, phi_mv, configs)
        self._plot_radial_avg(ax_i, configs, log)

        self._set_grids(ax_i)
        self._set_axis_labels(ax_i, configs)
        self._set_title(ax_i, kappa_r, configs)
        self._plot_vertical_lines(ax_i, configs, phi_mv, radii, debye_l)
        self._set_axis_lims(ax_i, configs)
        self._set_axis_ticks(ax_i, debye_l, configs)
        self._set_fig_labels(ax_i)
        self._set_mirror_axes(ax_i, configs)
        ax_i.legend(loc=configs.legend_loc,
                    fontsize=elsevier_plot_tools.FONT_SIZE_PT)

    def plot_panel_c(self,
                     ax_i: plt.axes,
                     configs: PlotConfig
                     ) -> None:
        """add scheme of the dlvo model for a sphere"""
        ax_i.axis('off')
        img = plt.imread(configs.isosurface_fig)
        ax_i.imshow(img)
        ax_i.text(0.09,
                  1,
                  'c)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
        ax_i.text(0.15,
                  0.5,
                  r'$\lambda_D$',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT-2)
        self.info_msg += f'\tIsosurface image: {configs.isosurface_fig}\n'
        # Add a circle on top of the image
        self._add_circle(ax_i, img, 890, '--', 'black')  # debye length
        self._add_circle(ax_i, img, 670, '-', 'black')  # Stern layer

    def _add_circle(self,
                    ax_i: plt.axes,
                    img: np.ndarray,
                    radius: float,
                    line_style: str,
                    color: str
                    ) -> None:
        """add a circle to the image"""
        # pylint: disable=too-many-arguments
        x_np_com = img.shape[0] / 2 - 15
        y_np_com = img.shape[1] / 2 + 40
        circle = plt.Circle((x_np_com, y_np_com),
                            radius,
                            fill=False,
                            color=color,
                            linestyle=line_style,
                            linewidth=0.7)
        ax_i.add_patch(circle)
        circle.set_zorder(10)

    def _plot_data(self,
                   ax_i: plt.axes,
                   radii: np.ndarray,
                   phi_mv: np.ndarray,
                   configs: PlotConfig
                   ) -> None:
        """plot the data"""
        ax_i.plot(radii, phi_mv, **configs.graph_styles)

    def _plot_radial_avg(self,
                         ax_i: plt.axes,
                         configs: PlotConfig,
                         log: logger.logging.Logger
                         ) -> None:
        """plot the radial average"""
        if configs.plot_radial_avg:
            try:
                df_i: pd.DataFrame = xvg_to_dataframe.XvgParser(
                    fname=configs.radial_avg_file,
                    log=log,
                    x_type=float).xvg_df
            except FileNotFoundError:
                self.info_msg += ('\tRadial average file not found: '
                                  f'{configs.radial_avg_file}\n')
                return
            ax_i.plot(df_i.iloc[:, 0],
                      df_i.iloc[:, 1],
                      c='k',
                      linestyle=':',
                      linewidth=elsevier_plot_tools.LINE_WIDTH,
                      label='numerical average')

    def _set_grids(self,
                   ax_i: plt.axes
                   ) -> None:
        """set the grid"""
        if self.configs.plot_config.if_grid:
            ax_i.grid(True, 'both', ls='--', color='gray', alpha=0.5, zorder=2)

    def _set_axis_labels(self,
                         ax_i: plt.axes,
                         configs: PlotConfig
                         ) -> None:
        """set the axis labels"""
        ax_i.set_xlabel(configs.labels.get('xlabel'),
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_ylabel(configs.labels.get('ylabel'),
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)

    def _set_title(self,
                   ax_i: plt.axes,
                   kappa_r: float,
                   configs: PlotConfig
                   ) -> None:
        """set the title"""
        if configs.if_title:
            ax_i.set_title(
                rf' $\kappa a$ := $\lambda_D^{{-1}} r_{{NP}}$ = {kappa_r:.2f}')

    def _plot_vertical_lines(self,
                             ax_i: plt.axes,
                             configs: PlotConfig,
                             phi_mv: np.ndarray,
                             radii: np.ndarray,
                             debye_l: float
                             ) -> None:
        """plot vertical lines"""
        # pylint: disable=too-many-arguments
        if configs.if_stern_line:
            self._plot_stern_layer_lines(ax_i, phi_mv, configs)
        if configs.if_debye_line:
            idx_closest = np.abs(radii - debye_l).argmin()
            phi_value = phi_mv[idx_closest+1]
            self._plot_debye_lines(ax_i, phi_value, debye_l, configs)
        if configs.if_2nd_debye:
            idx_closest = np.abs(radii - debye_l*2).argmin()
            phi_value = phi_mv[idx_closest+1]
            self._plot_debye_lines(ax_i, phi_value, debye_l*2, configs)

        self.info_msg += f'\tPotential at Debye: {phi_value:.2f} [mV]\n'

    def _set_axis_lims(self,
                       ax_i: plt.axes,
                       configs: PlotConfig
                       ) -> None:
        """set the axis limits"""
        x_lims: tuple[float, float] = ax_i.get_xlim()
        ax_i.set_xlim(configs.x_lims[0], x_lims[1])
        ax_i.set_ylim(configs.y_lims)

    def _set_axis_ticks(self,
                        ax_i: plt.axes,
                        debye_l: float,
                        configs: PlotConfig
                        ) -> None:
        """set the axis ticks"""
        x_tick_labels: list[str] = [str(f'{i:.1f}') for i in configs.x_ticks]
        debye_l_str: str = f'{debye_l:.2f}'
        configs.x_ticks.extend([debye_l])
        ax_i.set_xticks(configs.x_ticks)
        ax_i.set_xticklabels(x_tick_labels + [debye_l_str],
                             fontsize=elsevier_plot_tools.FONT_SIZE_PT)

        ax_i.set_yticks(configs.y_ticks)

    def _set_fig_labels(self,
                        ax_i: plt.axes
                        ) -> None:
        """set the figure labels"""
        ax_i.text(-0.013,
                  1,
                  'b)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)

    def _save_fig(self,
                  fig_i: plt.Figure,
                  configs: PlotConfig
                  ) -> None:
        """save the figure"""
        elsevier_plot_tools.save_close_fig(
            fig_i, fname=(fout := configs.graph_suffix))
        self.info_msg += f'\tFigure saved as `{fout}`\n'

    def _plot_debye_lines(self,
                          ax_i: plt.axes,
                          phi_value: float,
                          debye_l: float,
                          configs: PlotConfig
                          ) -> None:
        """plot lines for the debye length"""
        # pylint: disable=too-many-arguments

        h_label: str = r'$_{\lambda_D}$'
        ax_i.vlines(x=debye_l,
                    ymin=configs.y_lims[0],
                    ymax=phi_value+20,
                    color=configs.colors[4],
                    linestyle=configs.line_styles[2],
                    linewidth=elsevier_plot_tools.LINE_WIDTH)
        ax_i.text(debye_l-0.05,
                  phi_value+26,
                  h_label,
                  fontsize=elsevier_plot_tools.FONT_SIZE_PT+2)
        # Plot horizontal line from phi_value to the graph
        h_line_label = rf'$\psi${h_label} = {phi_value:.2f}'
        ax_i.hlines(y=phi_value,
                    xmin=0,
                    xmax=debye_l+0.5,
                    color=configs.colors[4],
                    linestyle=configs.line_styles[2],
                    linewidth=elsevier_plot_tools.LINE_WIDTH)
        ax_i.text(debye_l+0.6,
                  phi_value-0.2,
                  h_line_label,
                  fontsize=elsevier_plot_tools.FONT_SIZE_PT)

    def _plot_stern_layer_lines(self,
                                ax_i: plt.axes,
                                phi_mv: np.ndarray,
                                configs: PlotConfig,
                                ) -> None:
        """plot the stern layer lines"""
        y_lims: tuple[float, float] = ax_i.get_ylim()
        ax_i.vlines(x=(x_temp := self.configs.stern_layer/10),
                    ymin=configs.y_lims[0],
                    ymax=y_lims[1]+10,
                    color=configs.colors[4],
                    linestyle=configs.line_styles[0],
                    linewidth=elsevier_plot_tools.LINE_WIDTH)
        ax_i.text(x_temp-0.5,
                  phi_mv.max()+25,
                  'Stern layer',
                  fontsize=elsevier_plot_tools.FONT_SIZE_PT)

        ax_i.hlines(y=(phi_0 := phi_mv.max()),
                    xmin=0,
                    xmax=x_temp+0.5,
                    color=configs.colors[4],
                    linestyle=configs.line_styles[2],
                    linewidth=elsevier_plot_tools.LINE_WIDTH)
        ax_i.text(x_temp+0.6,
                  phi_0-0.2,
                  fr'$\psi_0$ = {phi_0:.2f}',
                  fontsize=elsevier_plot_tools.FONT_SIZE_PT)

    def _set_mirror_axes(self,
                         ax_i: plt.axes,
                         configs: PlotConfig
                         ) -> None:
        """set the mirror axes"""
        if not configs.if_mirror_axes:
            elsevier_plot_tools.remove_mirror_axes(ax_i)

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{PlotPotential.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print(f'{bcolors.CAUTION}This module is not meant to be run '
          f'independently.{bcolors.ENDC}\n')
