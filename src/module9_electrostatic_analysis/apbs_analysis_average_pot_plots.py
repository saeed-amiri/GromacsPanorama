"""
plots the ouputs of the APBS analysis (average potential plots)

"""

import os
import typing

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from common import logger
from common import elsevier_plot_tools
from common.colors_text import TextColor as bcolors


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
    for average, ind in zip(cut_radial_average, sphere_grid_range):
        average -= average[0]
        average += ind
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
