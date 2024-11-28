"""
Plot the electrostatic potential
"""

from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches

from common import logger
from common import xvg_to_dataframe
from common import elsevier_plot_tools
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_configs import \
    AllConfig, PlotConfig


class PlotPotential:
    """Plot the electrostatic potential"""

    info_msg: str = 'Message from PlotPotential:\n'

    def __init__(self,
                 radii: np.ndarray,
                 phi_r: np.ndarray,
                 interface_radii: np.ndarray,
                 interface_phi_r: np.ndarray,
                 debye_l: float,
                 configs: AllConfig,
                 log: logger.logging.Logger
                 ) -> None:
        # pylint: disable=too-many-arguments
        self.configs = configs
        debye_distance: float = debye_l + self.configs.np_radius / 10.0
        self.plot_potential(radii,
                            phi_r,
                            interface_radii,
                            interface_phi_r,
                            debye_distance,
                            log)
        self.write_msg(log)

    def plot_potential(self,
                       radii: np.ndarray,
                       phi_r: np.ndarray,
                       interface_radii: np.ndarray,
                       interface_phi_r: np.ndarray,
                       debye_d: float,
                       log: logger.logging.Logger
                       ) -> None:
        """plot and save the electostatic potential"""
        configs: PlotConfig = self.configs.plot_config

        axs: plt.axes
        fig_i: plt.figure
        fig_i, axs = elsevier_plot_tools.mk_canvas_multi('double_column',
                                                         n_rows=2,
                                                         n_cols=8,
                                                         aspect_ratio=1)
        # Flatten axs if it's a 2D array
        axs = axs.flatten() if isinstance(axs, np.ndarray) else axs

        for ax_i in axs:
            ax_i.axis('off')
        grid_panel = gridspec.GridSpec(2, 8, figure=fig_i)

        axs[0] = fig_i.add_subplot(grid_panel[0, :2])
        axs[1] = fig_i.add_subplot(grid_panel[0, 2:5])
        axs[2] = fig_i.add_subplot(grid_panel[0, 5:])
        axs[3] = fig_i.add_subplot(grid_panel[1, 0:4])
        axs[4] = fig_i.add_subplot(grid_panel[1, 4:])
        # Manually adjust the position of axses
        pos1 = axs[1].get_position()
        pos2 = axs[2].get_position()
        pos3 = axs[3].get_position()
        pos4 = axs[4].get_position()
        axs[1].set_position(
            [pos1.x0, pos1.y0, pos1.width + 0.06, pos1.height])
        axs[2].set_position(
            [pos2.x0, pos2.y0, pos2.width + 0.05, pos2.height])
        axs[3].set_position(
            [pos3.x0, pos3.y0, pos3.width - 0.015, pos3.height])
        axs[4].set_position(
            [pos4.x0 + 0.03, pos4.y0, pos4.width - 0.015, pos4.height])

        self.plot_panel_a(axs[0], configs)
        self.plot_panel_b(axs[1], configs)
        self.plot_panel_c(axs[2], configs)
        self.plot_panel_d(axs[3], radii, phi_r, debye_d, configs, log)
        self.plot_panel_e(
            axs[4], interface_radii, interface_phi_r, debye_d, configs, log)
        # Add a box around the subplots
        box = patches.Rectangle(
            (0.095, 0.01), 0.83, 0.88, transform=fig_i.transFigure,
            linewidth=1.5, edgecolor='black', facecolor='none')
        fig_i.patches.append(box)

        self._save_fig(fig_i, configs)

    def plot_panel_a(self,
                     ax_i: plt.axes,
                     configs: PlotConfig
                     ) -> None:
        """add shcem of the dlvo model for a sphere"""
        ax_i.axis('off')
        ax_i.imshow(plt.imread(configs.scheme_fig_path))
        ax_i.text(-0.02,
                  1,
                  'a)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
        self.info_msg += f'\tScheme image: {configs.scheme_fig_path}\n'

    def plot_panel_d(self,
                     ax_i: plt.axes,
                     radii: np.ndarray,
                     phi_r: np.ndarray,
                     debye_d: float,
                     configs: PlotConfig,
                     log: logger.logging.Logger
                     ) -> None:
        """plot the data"""
        # pylint: disable=too-many-arguments
        phi_mv: np.ndarray = phi_r * configs.voltage_to_mV
        # kappa * radius of the np
        kappa_r: float = \
            self.configs.computation_radius / \
            (debye_d * configs.angstrom_to_nm)
        self._plot_data(ax_i, radii, phi_mv, configs)
        phi_vlaue_calced: float = \
            self._plot_vertical_lines(ax_i, configs, phi_mv, radii, debye_d)
        self._plot_radial_avg(ax_i, configs, debye_d, phi_vlaue_calced, log)

        self._plot_experiment_lines(ax_i, phi_mv, radii, configs)
        ax_i.hlines(y=0,
                    xmin=0,
                    xmax=7.0,
                    color=configs.colors[4],
                    linestyle=configs.line_styles[2],
                    linewidth=elsevier_plot_tools.LINE_WIDTH)
        ax_i.text(4.8,
                  12,
                  rf'$\psi\,=\,0.0$',
                  ha='right',
                  va='top',
                  fontsize=elsevier_plot_tools.FONT_SIZE_PT)

        self._set_grids(ax_i)
        self._set_axis_labels(ax_i, configs)
        self._set_title(ax_i, kappa_r, configs)
        self._set_axis_lims(ax_i, configs)
        self._set_axis_ticks(ax_i, debye_d, configs)
        self._set_fig_labels(ax_i)
        self._set_mirror_axes(ax_i, configs)
        ax_i.legend(loc=configs.legend_loc,
                    fontsize=elsevier_plot_tools.FONT_SIZE_PT)

    def plot_panel_e(self,
                     ax_i: plt.axes,
                     interface_radii: np.ndarray,
                     interface_phi_r: np.ndarray,
                     debye_d: float,
                     configs: PlotConfig,
                     log: logger.logging.Logger
                     ) -> None:
        """plot surface potential"""
        # pylint: disable=too-many-arguments
        _configs = configs
        _configs.x_ticks = [1.75] + [2.5] + configs.x_ticks[1:]
        self._set_axis_ticks(ax_i, debye_d, _configs)
        xticks_labels: list[str] = \
            [r'$r^\star_c$'] + [str(f'{i:.1f}') for i in configs.x_ticks[1:]]
        ax_i.set_xticks(_configs.x_ticks)
        ax_i.set_xticklabels(xticks_labels,
                             fontsize=elsevier_plot_tools.FONT_SIZE_PT)

        ax_i.plot(interface_radii,
                  interface_phi_r,
                  color=_configs.colors[0],
                  linestyle=_configs.line_styles[0],
                  linewidth=elsevier_plot_tools.LINE_WIDTH,
                  label='interface')
        ax_i.set_xlim(1.4, 8.0)
        ax_i.set_xlabel(r'$r^\star$ [nm]',
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_ylabel(r'interface EP, $\psi^\star$ [mV]',
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        self._set_mirror_axes(ax_i, _configs)
        xlims: tuple[float, float] = ax_i.get_xlim()
        ylims: tuple[float, float] = ax_i.get_ylim()
        ymax: float = interface_phi_r.max()
        ax_i.vlines(x=1.75,
                    ymin=ylims[0],
                    ymax=ylims[1],
                    color=_configs.colors[4],
                    linestyle=_configs.line_styles[0],
                    linewidth=elsevier_plot_tools.LINE_WIDTH)
        ax_i.hlines(y=ymax,
                    xmin=xlims[0],
                    xmax=2.8,
                    color=_configs.colors[4],
                    linestyle=_configs.line_styles[2],
                    linewidth=elsevier_plot_tools.LINE_WIDTH)
        ax_i.hlines(y=0,
                    xmin=xlims[0],
                    xmax=6.8,
                    color=_configs.colors[4],
                    linestyle=_configs.line_styles[2],
                    linewidth=elsevier_plot_tools.LINE_WIDTH)
        ax_i.set_xlim(xlims)
        ax_i.set_ylim(ylims)
        ax_i.text(-0.078,
                  1,
                  'e)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
        ax_i.text(4.2,
                  ymax+5.5,
                  rf'$\psi^\star_0=\,${ymax:.1f}',
                  ha='right',
                  va='top',
                  fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.text(7.9,
                  7,
                  rf'$\psi^\star=\,${0.0:.1f}',
                  ha='right',
                  va='top',
                  fontsize=elsevier_plot_tools.FONT_SIZE_PT)

    def plot_panel_c(self,
                     ax_i: plt.axes,
                     configs: PlotConfig
                     ) -> None:
        """potential from the apbs simulation with vmd"""
        ax_i.axis('off')
        ax_i.imshow(plt.imread(configs.apbs_fig))
        ax_i.text(-0.02,
                  1,
                  'c)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
        self.info_msg += f'\tScheme image: {configs.apbs_fig}\n'

    def plot_panel_b(self,
                     ax_i: plt.axes,
                     configs: PlotConfig
                     ) -> None:
        """add scheme of the dlvo model for a sphere"""
        ax_i.axis('off')
        img = plt.imread(configs.isosurface_fig)
        ax_i.imshow(img)
        ax_i.text(-0.02,
                  1,
                  'b)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)

        self.info_msg += f'\tIsosurface image: {configs.isosurface_fig}\n'

    def _add_circle(self,
                    ax_i: plt.axes,
                    img: np.ndarray,
                    radius: float,
                    line_style: str,
                    color: str
                    ) -> None:
        """add a circle to the image"""
        # pylint: disable=too-many-arguments
        x_np_com = img.shape[0] / 2
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
                         debye_d: float,
                         phi_value_from_computation: float,
                         log: logger.logging.Logger
                         ) -> float:
        """plot the radial average"""
        # pylint: disable=too-many-arguments
        apbs_files: dict[str, str] = {}
        if configs.plot_radial_avg:
            try:
                for item, d_file in configs.radial_avg_files.items():
                    df_i: pd.DataFrame = xvg_to_dataframe.XvgParser(
                        fname=d_file,
                        log=log,
                        x_type=float).xvg_df
                    apbs_files[item] = df_i
            except FileNotFoundError:
                self.info_msg += (msg := '\tRadial average file not found: '
                                  f'{configs.radial_avg_files}\n')
                print(f'{bcolors.WARNING}{msg}{bcolors.ENDC}')
                return
            for j, (item, df_i) in enumerate(apbs_files.items()):
                ax_i.plot(df_i.iloc[:, 0],
                          df_i.iloc[:, 1],
                          color=configs.colors[j+1],
                          linestyle=configs.line_styles[j*2],
                          linewidth=elsevier_plot_tools.LINE_WIDTH,
                          label=item)
                self._get_debye_potential(df_i, item, debye_d)
                idx_closest = np.abs(df_i.iloc[:, 0] - debye_d).argmin()
                try:
                    phi_value = (df_i.iloc[:, 1][idx_closest] +
                                 df_i.iloc[:, 1][idx_closest+1])/2
                except IndexError:
                    phi_value = df_i.iloc[:, 1][idx_closest]
                except KeyError:
                    phi_value = df_i.iloc[:, 0][idx_closest]
                if configs.if_debye_line and \
                   np.abs(phi_value - phi_value_from_computation) > 10:
                    self._plot_debye_lines(
                        ax_i, phi_value, debye_d, configs, order_of_plot=2)
            return phi_value

    def _get_debye_potential(self,
                             df_i: pd.DataFrame,
                             item: str,
                             debye_d: float
                             ) -> None:
        """get the potential at the Debye length"""
        idx_closest = np.abs(df_i.iloc[:, 0] - 3).argmin()
        try:
            phi_value = (
                df_i.iloc[idx_closest, 1] + df_i.iloc[idx_closest+1, 1])/2
        except IndexError:
            phi_value = df_i.iloc[idx_closest, 1]
        except KeyError:
            phi_value = df_i.iloc[idx_closest, 0]

        self.info_msg += (
            f'\tPotential at stern ({item}): {phi_value:.2f} [mV] = '
            f'{phi_value/25.7:.2f} [kT/e]\n')
        idx_closest = np.abs(df_i.iloc[:, 0] - debye_d).argmin()
        try:
            phi_value = (
                df_i.iloc[idx_closest, 1] + df_i.iloc[idx_closest+1, 1])/2
        except IndexError:
            phi_value = df_i.iloc[idx_closest, 1]
        except KeyError:
            phi_value = df_i.iloc[idx_closest, 0]
        self.info_msg += (
            f'\tPotential at Debye ({item}): {phi_value:.2f} [mV] = '
            f'{phi_value/25.7:.2f} [kT/e]\n')

    def _plot_experiment_lines(self,
                               ax_i: plt.axes,
                               phi_mv: np.ndarray,
                               radii: np.ndarray,
                               configs: PlotConfig
                               ) -> None:
        """plot the experimental lines"""
        if configs.if_experiment:
            # finding the location of the experimental data in x axis
            r_closest = np.abs(phi_mv - configs.phi_r_exprimental_avg).argmin()
            ax_i.hlines(y=configs.phi_r_exprimental_avg,
                        xmin=0,
                        xmax=radii[r_closest] + 0.5,
                        color=configs.colors[3],
                        linestyle=configs.line_styles[2],
                        linewidth=elsevier_plot_tools.LINE_WIDTH,
                        label='Experimental')
            ax_i.text(radii[r_closest] + 0.5,
                      configs.phi_r_exprimental_avg,
                      fr'$\psi_{{exp}}$ = {configs.phi_r_exprimental_avg:.2f}',
                      fontsize=elsevier_plot_tools.FONT_SIZE_PT)

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
        # Adjust the position of the x-axis label
        ax_i.set_xlabel(ax_i.get_xlabel(), labelpad=6)
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
                             debye_d: float
                             ) -> float:
        """plot vertical lines"""
        # pylint: disable=too-many-arguments
        phi_value: float = 0.0
        if configs.if_np_radius_line:
            ax_i.vlines(x=self.configs.np_radius/10,
                        ymin=configs.y_lims[0],
                        ymax=configs.y_lims[1],
                        color=configs.colors[4],
                        linestyle=configs.line_styles[0],
                        linewidth=elsevier_plot_tools.LINE_WIDTH)
        if configs.if_stern_line:
            self._plot_stern_layer_lines(ax_i, phi_mv, configs)
        else:
            self._plot_stern_vline_only(ax_i, configs)
        if configs.if_debye_line:
            idx_closest = np.abs(radii - debye_d).argmin()
            phi_value = phi_mv[idx_closest+1]
            self._plot_debye_lines(ax_i, phi_value, debye_d, configs)
        if configs.if_2nd_debye:
            idx_closest = np.abs(radii - debye_d*2).argmin()
            phi_value = phi_mv[idx_closest+1]
            self._plot_debye_lines(ax_i, phi_value, debye_d*2, configs)

        self.info_msg += f'\tPotential at Debye: {phi_value:.2f} [mV]\n'
        return phi_value

    def _set_axis_lims(self,
                       ax_i: plt.axes,
                       configs: PlotConfig
                       ) -> None:
        """set the axis limits"""
        ax_i.set_xlim(configs.x_lims)
        ax_i.set_ylim(configs.y_lims)

    def _set_axis_ticks(self,
                        ax_i: plt.axes,
                        debye_d: float,
                        configs: PlotConfig
                        ) -> None:
        """set the axis ticks"""
        x_tick_labels: list[str] = [str(f'{i:.1f}') for i in configs.x_ticks]
        if configs.if_debye_line:
            debye_d_str: str = f'{debye_d:.1f}'
            configs.x_ticks.extend([debye_d])
            x_tick_labels.append(debye_d_str)
        if configs.if_stern_line:
            stern_layer_str: str = f'{self.configs.stern_layer/10:.1f}'
            configs.x_ticks.extend([self.configs.stern_layer/10])
            x_tick_labels.append(stern_layer_str)
        ax_i.set_xticks(configs.x_ticks)
        ax_i.set_xticklabels(x_tick_labels,
                             fontsize=elsevier_plot_tools.FONT_SIZE_PT)

        ax_i.set_yticks(configs.y_ticks)

    def _set_fig_labels(self,
                        ax_i: plt.axes
                        ) -> None:
        """set the figure labels"""
        ax_i.text(-0.013,
                  1,
                  'd)',
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
                          debye_d: float,
                          configs: PlotConfig,
                          order_of_plot: int = 1
                          ) -> None:
        """plot lines for the debye length"""
        # pylint: disable=too-many-arguments
        h_label: str = r'$_{\lambda_D}$'
        h_line_label = rf'$\psi${h_label} = {phi_value:.2f}'
        if order_of_plot == 1:
            ax_i.vlines(x=debye_d,
                        ymin=configs.y_lims[0],
                        ymax=phi_value+60,
                        color=configs.colors[4],
                        linestyle=configs.line_styles[2],
                        linewidth=elsevier_plot_tools.LINE_WIDTH)
            ax_i.text(debye_d+0.05,
                      phi_value+66,
                      h_label,
                      fontsize=elsevier_plot_tools.FONT_SIZE_PT+2)
            # Plot horizontal line from phi_value to the graph
        ax_i.hlines(y=phi_value,
                    xmin=0,
                    xmax=debye_d+0.5,
                    color=configs.colors[4],
                    linestyle=configs.line_styles[2],
                    linewidth=elsevier_plot_tools.LINE_WIDTH)
        ax_i.text(debye_d-3.5,
                  phi_value-15.0,
                  h_line_label,
                  fontsize=elsevier_plot_tools.FONT_SIZE_PT-1)

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

        ax_i.hlines(y=(phi_0 := phi_mv.max()),
                    xmin=0,
                    xmax=x_temp+0.5,
                    color=configs.colors[4],
                    linestyle=configs.line_styles[2],
                    linewidth=elsevier_plot_tools.LINE_WIDTH)
        ax_i.text(x_temp+0.6,
                  phi_0,
                  fr'$\psi_0$ = {phi_0:.2f}',
                  fontsize=elsevier_plot_tools.FONT_SIZE_PT)

    def _plot_stern_vline_only(self,
                               ax_i: plt.axes,
                               configs: PlotConfig
                               ) -> None:
        """plot the stern layer vertical line only"""
        ax_i.vlines(self.configs.stern_layer/10,
                    ymin=configs.y_lims[0],
                    ymax=configs.y_lims[1],
                    color=configs.colors[4],
                    linestyle=configs.line_styles[0],
                    linewidth=elsevier_plot_tools.LINE_WIDTH)

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
