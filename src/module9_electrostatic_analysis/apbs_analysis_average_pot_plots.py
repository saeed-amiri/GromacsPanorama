"""
plots the ouputs of the APBS analysis (average potential plots)

"""
from typing import Callable, Tuple, List, Dict, Union, Any


import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from common import elsevier_plot_tools


class PlotParameterFittedPotential:
    """parameter for plottinge the Debye length and surface potential"""
    # pylint: disable=missing-function-docstring
    # pylint: disable=invalid-name
    # pylint: disable=too-many-arguments

    @property
    def LAMBDA_D(self) -> dict[str, str | tuple[float, float] | list[float]]:
        return {'label': r'$\lambda_d$ (from Sphere`s surface)',
                'ylable': 'Debye length [nm]',
                'output_file': 'debye_length.jpg',
                'legend_loc': 'upper left',
                'y_lim': (1.35, 2.45),
                'y_ticks': [1.5, 1.9, 2.3],
                'x_ticks': [9, 10, 11, 12, 13]}

    @property
    def PSI_0(self) -> dict[str, str | tuple[float, float] | list[float]]:
        return {'label': r'$\psi_0$',
                'ylable': 'potential [mV]',
                'output_file': 'surface_potential.jpg',
                'legend_loc': 'lower left',
                'y_lim': (-15, 130),
                'y_ticks': [0, 60, 120],
                'x_ticks': [9, 10, 11, 12, 13]}

    @property
    def SIGMA(self) -> dict[str, str | tuple[float, float] | list[float]]:
        return {'label': r'$\sigma$',
                'ylable': r'charge density [e/m$^2$]',
                'output_file': 'sigma.jpg',
                'legend_loc': 'lower left',
                'y_lim': (-0.001, 0.013),
                'y_ticks': [0, 0.006, 0.012],
                'x_ticks': [9, 10, 11, 12, 13]}

    @property
    def X_LABEL(self) -> str:
        return 'z [nm] (of Box)'

    @property
    def MARKSIZE(self) -> float:
        return 2.0

    @property
    def LINEWIDTH(self) -> float:
        return 0.75

    @property
    def ODA_BOUND(self) -> tuple[int, int]:
        return (90, 96)


def interactive_plot(plots_data: list[tuple[np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            tuple[float, float, float],
                                            int]],
                     z_grid_spacing: float,
                     ) -> None:
    """Interactive plot for the fitted potential"""
    mpl.rcParams['font.size'] = 20
    fig, ax_i = plt.subplots(figsize=(20, 16))

    def plot_index(idx):
        ax_i.cla()  # Clear the current figure
        radii, radial_average, fitted_pot, popt, fit_metrics, grid = \
            plots_data[idx]
        ax_i.plot(radii, radial_average, 'k-')
        ax_i.plot(radii, fitted_pot, 'r--')
        ax_i.text(0.5,
                  0.5,
                  s=(rf'$\lambda_d$={popt[0]:.2f} A',
                      rf'$\psi_0$={popt[1]:.2f}$ mV',
                      rf'$\psi_{{inf}}$={popt[2]:.2f} mV'),
                  transform=ax_i.transAxes,
                  )
        ax_i.text(0.5,
                  0.6,
                  s=(f'$R^2$={fit_metrics[0]:.2f}',
                      f'MSE={fit_metrics[1]:.2f}',
                      f'MAE={fit_metrics[2]:.2f}'),
                  transform=ax_i.transAxes,
                  )

        ax_i.set_title((f'z_index={grid}, '
                        f'z={grid*z_grid_spacing:.3f}'))
        ax_i.set_xlabel('r (Å)')
        ax_i.set_ylabel('Potential')

        fig.canvas.draw_idle()  # Use fig's canvas to redraw

    current_index = [0]  # Use list for mutable integer

    def on_key(event):
        if event.key == 'right':
            current_index[0] = min(len(plots_data) - 1, current_index[0] + 1)
            plot_index(current_index[0])
        elif event.key == 'left':
            current_index[0] = max(0, current_index[0] - 1)
            plot_index(current_index[0])
        elif event.key == 'up':
            current_index[0] = min(len(plots_data) - 5, current_index[0] + 5)
            plot_index(current_index[0])
        elif event.key == 'down':
            current_index[0] = max(0, current_index[0] - 5)
            plot_index(current_index[0])

    fig.canvas.mpl_connect('key_press_event', on_key)

    plot_index(0)
    plt.show()


def plot_debug(cut_radii: list[np.ndarray],
               cut_radial_average: list[np.ndarray],
               radii_list: list[np.ndarray],
               radial_average_list: list[np.ndarray],
               sphere_grid_range: np.ndarray,
               ) -> None:
    """Plot for debugging"""
    # pylint: disable=too-many-arguments

    mpl.rcParams['font.size'] = 20

    for i, radial_average in enumerate(cut_radial_average):
        _, ax = plt.subplots(figsize=(30, 16))
        ax.plot(radii_list[i], radial_average_list[i], 'r:')
        ax.plot(cut_radii[i],
                radial_average,
                'k-',
                label=f'z={sphere_grid_range[i]}')
        plt.legend()
        plt.show()


def plot_debye_surface_potential(data: dict[np.int64, float],
                                 type_data: str,
                                 z_grid_spacing: tuple[float, float, float],
                                 np_z_offset: float,
                                 close_fig: bool = True,
                                 return_data: bool = False,
                                 ) -> None | \
                                    Tuple[plt.Figure, plt.Axes] | \
                                    dict[str, Any]:
    """Plot the Debye length and surface potential"""
    # pylint: disable=too-many-arguments
    plot_config: PlotParameterFittedPotential = PlotParameterFittedPotential()
    figure: tuple[plt.Figure, plt.Axes] = elsevier_plot_tools.mk_canvas(
        'single_column')
    fig_i, ax_i = figure

    xdata: np.ndarray = np.asanyarray(
        [float(i)*z_grid_spacing[2] for i in data.keys()])
    ydata: np.ndarray = np.asanyarray(list(data.values()))

    if type_data == 'lambda_d':
        plot_parameters: dict[str, str | tuple[float, float] | list[float]] = \
            plot_config.LAMBDA_D
        ydata /= 10.0  # Convert to nm
    elif type_data == 'psi_0':
        plot_parameters = plot_config.PSI_0
    elif type_data == 'sigma':
        plot_parameters = plot_config.SIGMA
    else:
        raise ValueError(f'Unknown type_data: {type_data}')

    ax_i.plot(xdata / 10.0 - np_z_offset,  # Convert to nm
              ydata,
              ls=elsevier_plot_tools.LINE_STYLES[3],
              color=elsevier_plot_tools.DARK_RGB_COLOR_GRADIENT[0],
              marker=elsevier_plot_tools.MARKER_SHAPES[0],
              lw=plot_config.LINEWIDTH,
              markersize=plot_config.MARKSIZE,
              label=plot_parameters['label'])

    ax_i.set_xlabel(plot_config.X_LABEL)
    ax_i.set_xticks(plot_parameters['x_ticks'])
    ax_i.set_ylabel(plot_parameters['ylable'])
    ax_i.set_ylim(plot_parameters['y_lim'])
    ax_i.set_yticks(plot_parameters['y_ticks'])

    ax_i.grid(True, ls='--', lw=0.5, alpha=0.5, color='grey')

    ax_i.legend()

    oda_bound: tuple[float, float] = (
        plot_config.ODA_BOUND[0] * z_grid_spacing[2] / 10.0 - np_z_offset,
        plot_config.ODA_BOUND[1] * z_grid_spacing[2] / 10.0 - np_z_offset
        )
    # Shade the area between ODA_BOUND
    ax_i.fill_betweenx(ax_i.get_ylim(),
                       oda_bound[0],
                       oda_bound[1],
                       color='gray',
                       edgecolor=None,
                       alpha=0.5,
                       label='ODA`s N locations',
                       )
    if return_data:
        return {'xdata': xdata / 10.0 - np_z_offset,
                'ydata': ydata,
                'oda_bound': oda_bound,
                'z_indicies': np.asanyarray(list(data.keys()))
                }
    if not close_fig:
        return fig_i, ax_i
    elsevier_plot_tools.save_close_fig(fig_i,
                                       plot_parameters['output_file'],
                                       loc=plot_parameters['legend_loc'],
                                       )
    return None


# Boltzman distribution plots
class PlotBoltzmanDistribution:
    """parameter for plottinge the Debye length and surface potential"""
    # pylint: disable=missing-function-docstring
    # pylint: disable=invalid-name
    # pylint: disable=too-many-arguments

    @property
    def BOLTZMANN(self) -> Dict[
       str, Union[str, Tuple[float, float], List[float]]]:
        return {'label': 'Distribution',
                'ylable': r'$c^+/c_0$',
                'xlable': r'r$^*$ [nm]',
                'output_file': 'oda_distribution.jpg',
                'legend_loc': 'upper left',
                'y_lim': (-0.1, 4.1),
                'y_ticks': [0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
                'x_lim': (-0.5, 10.5),
                'x_ticks': [0, 2, 4, 6, 8, 10]}

    @property
    def COLOR_STR(self) -> List[str]:
        return elsevier_plot_tools.LINE_COLORS

    @property
    def COLOR_GENERATE(self) -> Callable[[int, str, int], List[str]]:
        return elsevier_plot_tools.generate_shades

    @property
    def LINESTYLE(self) -> List[
       Tuple[str, Tuple[None] | Tuple[int, Tuple[int, ...]]]]:
        return elsevier_plot_tools.LINESTYLE_TUPLE[::-1]


def plot_boltzman_distribution(dist_radii: Dict[int,  # z index
                                                Tuple[np.ndarray,  # phi
                                                      np.ndarray]  # radii
                                                ],
                               cut_ind: int  # cut of radius
                               ) -> None:
    """Plot the Boltzman distribution"""
    # pylint: disable=too-many-locals
    configs: PlotBoltzmanDistribution = PlotBoltzmanDistribution()
    figure_i: Tuple[plt.Figure, plt.Axes] = elsevier_plot_tools.mk_canvas(
            'single_column')
    fig_i, ax_i = figure_i
    figure_j: Tuple[plt.Figure, plt.Axes] = elsevier_plot_tools.mk_canvas(
            'single_column')
    fig_j, ax_j = figure_j
    plot_parameters: Dict[
       str, Union[str, Tuple[float, float], List[float]]] = \
        configs.BOLTZMANN
    colors: list[str] = configs.COLOR_STR
    lstyles: List[
       Tuple[str, Union[Tuple[None], Tuple[int, Tuple[int, ...]]]]] = \
        configs.LINESTYLE

    for i, (ind, phi_i_radii) in enumerate(dist_radii.items()):
        ax_i.plot(phi_i_radii[1][:cut_ind]/10.0,  # Convert to nm
                  phi_i_radii[0][:cut_ind],
                  label=f'z index = {ind}',
                  color=colors[i],
                  ls=lstyles[i][1],
                  )
        ax_j.plot(phi_i_radii[1][:cut_ind]/10.0,  # Convert to nm
                  phi_i_radii[0][:cut_ind]/np.max(phi_i_radii[0][:cut_ind]),
                  label=f'z index = {ind}',
                  color=colors[i],
                  ls=lstyles[i][1],
                  )
    for ax_ in (ax_i, ax_j):
        ax_.set_xlabel(plot_parameters['xlable'])
        ax_.set_xticks(plot_parameters['x_ticks'])
        ax_.set_yticks(plot_parameters['y_ticks'])
        ax_.grid(True, ls='--', lw=0.5, alpha=0.5, color='grey')
        ax_.legend()
    ax_i.set_ylim(plot_parameters['y_lim'])
    ax_j.set_ylim((-0.1, 1.1))
    ax_i.set_ylabel(plot_parameters['ylable'])
    ax_j.set_ylabel(plot_parameters['ylable'] + ' (normalized)')
    elsevier_plot_tools.save_close_fig(
        fig_i,
        plot_parameters['output_file'],
        loc=plot_parameters['legend_loc'])
    elsevier_plot_tools.save_close_fig(
        fig_j,
        'normalized_' + plot_parameters['output_file'],
        loc=plot_parameters['legend_loc'])


def plot_all_boltman_distribution(all_dist_radii: Dict[int, Tuple[np.ndarray,
                                                                  np.ndarray]],
                                  cut_ind: int  # cut of radius
                                  ) -> None:
    """Plot all the Boltzman distribution"""
    # pylint: disable=too-many-locals
    configs: PlotBoltzmanDistribution = PlotBoltzmanDistribution()
    figure_i: Tuple[plt.Figure, plt.Axes] = elsevier_plot_tools.mk_canvas(
            'double_column')
    fig_i, ax_i = figure_i
    figure_j: Tuple[plt.Figure, plt.Axes] = elsevier_plot_tools.mk_canvas(
            'double_column')
    fig_j, ax_j = figure_j
    plot_parameters: Dict[
       str, Union[str, Tuple[float, float], List[float]]] = \
        configs.BOLTZMANN
    colors: List[str] = configs.COLOR_GENERATE(len(all_dist_radii),)

    for i, (ind, phi_i_radii) in enumerate(all_dist_radii.items()):
        if i in (0, len(all_dist_radii) - 1):
            label: str | None = f'z index = {ind}'
        else:
            label = None
        ax_i.plot(phi_i_radii[1][:cut_ind]/10.0,  # Convert to nm
                  phi_i_radii[0][:cut_ind],
                  label=label,
                  color=colors[i],
                  ls='-',
                  )
        ax_j.plot(phi_i_radii[1][:cut_ind]/10.0,  # Convert to nm
                  phi_i_radii[0][:cut_ind]/np.max(phi_i_radii[0][:cut_ind]),
                  label=label,
                  color=colors[i],
                  ls='-',
                  )
    for ax_ in (ax_i, ax_j):
        ax_.set_xlabel(plot_parameters['xlable'])
        ax_.set_xticks(plot_parameters['x_ticks'])
        ax_.grid(True, ls='--', lw=0.5, alpha=0.5, color='grey')
        ax_.legend()
    ax_i.set_ylabel(plot_parameters['ylable'])
    ax_j.set_ylabel(plot_parameters['ylable'] + ' (normalized)')

    elsevier_plot_tools.save_close_fig(
        fig_i,
        fname='all_' + plot_parameters['output_file'],
        loc=plot_parameters['legend_loc'])
    elsevier_plot_tools.save_close_fig(
        fig_j,
        fname='all_normalized' + plot_parameters['output_file'],
        loc=plot_parameters['legend_loc'])
