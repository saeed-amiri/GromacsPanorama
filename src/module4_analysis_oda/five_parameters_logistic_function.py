"""
The function and plot the main function of:
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pylab as plt

from common import logger
from common import plot_tools
from common.colors_text import TextColor as bcolors


def logistic_5pl2s(x_data: np.ndarray,
                   c_init_guess: float,
                   b_init_guess: float,
                   g_init_guess: float,
                   ) -> np.ndarray:
    """Five-parameters logistic function with double slopes"""
    response_zero: float = 0.0
    response_infinite: float = 1.0
    g_modified: float = \
        2 * abs(b_init_guess) * g_init_guess / (1 + g_init_guess)
    return response_infinite + \
        (response_zero - response_infinite) /\
        (1+(x_data/c_init_guess)**b_init_guess) ** g_modified


@dataclass
class BasePlotConfig:
    """Basic configurations for all the plots"""
    fname: str = '5plf.png'


@dataclass
class MultiParamPlotConfig(BasePlotConfig):
    """parameters for the the plots with different 5pl2s parameters"""


class Plot5PLF2S:
    """create a data set and plot five parameters logistic function"""

    info_msg: str = 'Messege from Plot5PLF2S:\n'
    config: "MultiParamPlotConfig"

    def __init__(self,
                 log: logger.logging.Logger,
                 config: "MultiParamPlotConfig" = MultiParamPlotConfig()
                 ) -> None:
        self.config = config
        xdata, ydata,  first_derivative, second_derivative = \
            self.initiate_data()
        self.plot_graphs(xdata, ydata, first_derivative, second_derivative)
        self._write_msg(log)

    def initiate_data(self) -> tuple[np.ndarray, ...]:
        """make the data"""
        xdata: np.ndarray = np.linspace(0, 120)
        initial_guesses: tuple[float, ...] = self.set_initial_guesses()
        ydata = logistic_5pl2s(xdata, *initial_guesses)
        first_derivative, second_derivative = \
            self.initiate_derivative(xdata, ydata)
        return xdata, ydata, first_derivative, second_derivative

    def set_initial_guesses(self) -> tuple[float, ...]:
        """set the initial guesses"""
        c_init_guess: float = 100
        b_init_guess: float = 6
        g_init_guess: float = 1
        self.info_msg += ('\tFunction value set:\n'
                          f'\t\tc {c_init_guess:.3f}\n'
                          f'\t\tb {b_init_guess:.3f}\n'
                          f'\t\tg {g_init_guess:.3f}\n'
                          f'\t\ta {0:.3f}\n'
                          f'\t\tb {1:.3f}\n'
                          )
        return (c_init_guess, b_init_guess, g_init_guess)

    def plot_graphs(self,
                    xdata: np.ndarray,
                    ydata: np.ndarray,
                    first_derivative: np.ndarray,
                    second_derivative: np.ndarray
                    ) -> None:
        """plot the graph"""
        # pylint: disable=too-many-locals
        curvature: np.ndarray = \
            self.initiate_curvature(first_derivative, second_derivative)
        first_derv_norm_factor: float = max(first_derivative)
        curvature_norm_factor: float = \
            max(np.abs(min(curvature)), np.abs(max(curvature)))

        xrange: tuple[float, float] = (min(xdata), max(xdata))
        yrange: tuple[float, float] = \
            (min(curvature/curvature_norm_factor), max(ydata))

        c_vline: float = xdata[np.argmax(first_derivative)]
        first_turn_vline: float = xdata[np.argmax(curvature)]
        second_turn_vline: float = xdata[np.argmin(curvature)]

        fig_i, ax_i = \
            plot_tools.mk_canvas(x_range=xrange, height_ratio=(5**0.5-1)*1.75)

        ax_i.plot(xdata,
                  ydata,
                  c='k',
                  ls='-',
                  label='F(r)')

        ax_i.plot(xdata,
                  first_derivative/first_derv_norm_factor,
                  c='k',
                  ls='--',
                  label=r'F$^\prime$(r) [normalized]')

        ax_i.plot(xdata,
                  curvature/curvature_norm_factor,
                  c='k',
                  ls=':',
                  label=r'$\kappa$ [normalized]')

        ylims: tuple[float, float] = ax_i.get_ylim()

        fit_points: list[float] = \
            [c_vline, first_turn_vline, second_turn_vline]

        for fit_point in fit_points:
            ax_i.vlines(fit_point,
                        ymin=ylims[0],
                        ymax=ylims[1],
                        color='grey',
                        ls='-')

        plt_xlims: tuple[float, float] = ax_i.get_xlim()
        for h_line in [-1, 1]:
            ax_i.hlines(h_line,
                        xmin=plt_xlims[0],
                        xmax=plt_xlims[1],
                        color='grey',
                        alpha=0.75,
                        ls=':')

        ax_i.set_ylim(ylims)
        ax_i.set_xlim(plt_xlims)

        ax_i = self._set_x_axis(ax_i, fit_points)
        ax_i = self._set_y_axis(ax_i, yrange)

        # ax_i.grid(True, linestyle='--', color='gray', alpha=0.5)
        self.save_close_fig(fig_i, fname=self.config.fname)
        self.info_msg += f'\tThe figure saved as `{self.config.fname}`\n'

        plt.show()

    @staticmethod
    def _set_x_axis(ax_i: plt.axes,
                    fit_points: list[float]
                    ) -> plt.axes:
        """modify and set the x axis"""
        ax_i.set_xticks([])
        turns_labels = ['b', 'a', 'c']
        for loc_i, loc_l in zip(fit_points, turns_labels):
            ax_i.text(loc_i-2, -1.25, f'{loc_l}')
        ax_i.set_xlabel('r', labelpad=20)
        return ax_i

    @staticmethod
    def _set_y_axis(ax_i: plt.axes,
                    yrange: tuple[float, float]
                    ) -> plt.axes:
        """modify and set the x axis"""
        y_values = np.linspace(yrange[0], yrange[1], 5)
        ax_i.set_yticks(y_values)

        # Set all labels to empty, except the first and last
        labels = [''] * len(y_values)
        labels[0] = r'$-l$'
        labels[2] = r'$u$'
        labels[-1] = r'$l$'

        plt.gca().set_yticklabels(labels)
        plt.tick_params(axis='y', direction='in', length=3, width=1)
        ax_i.set_ylabel('F(r)')
        return ax_i

    @staticmethod
    def initiate_derivative(xdata,
                            ydate
                            ) -> tuple[np.ndarray, ...]:
        """return the dervetive of the data"""
        first_derivative = np.gradient(ydate, xdata)
        second_derivative = np.gradient(first_derivative, xdata)
        return first_derivative, second_derivative

    @staticmethod
    def initiate_curvature(first_derivative,
                           second_derivative
                           ) -> np.ndarray:
        """calculate the cuvature"""
        return second_derivative / (1 + first_derivative**2)**(1.5)

    @staticmethod
    def save_close_fig(fig: plt.figure,  # The figure to save,
                       fname: str,  # Name of the output for the fig
                       transparent=False
                       ) -> None:
        """
        Save the figure and close it.

        This method saves the given figure and closes it after saving.
        """
        plt.legend(ncol=3,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1.16),
                   fontsize=11)
        fig.savefig(fname,
                    dpi=300,
                    pad_inches=0.1,
                    edgecolor='auto',
                    bbox_inches='tight',
                    transparent=transparent
                    )
        plt.close(fig)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    Plot5PLF2S(log=logger.setup_logger('plot_five_param_logistic.log'))
