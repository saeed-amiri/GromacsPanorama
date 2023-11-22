"""must be updated
for PRE
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

import common.logger as logger
import common.static_info as stinfo
import common.xvg_to_dataframe as xvg
import common.plot_tools as plot_tools
from common.colors_text import TextColor as bcolors


class PlotXvg(xvg.XvgParser):
    """plot xvg here"""
    def __init__(self,
                 fname: str,
                 log: logger.logging.Logger
                 ) -> None:
        super().__init__(fname, log)
        # self.make_graph()
        self.plot_contact_info()

    def plot_contact_info(self) -> None:
        column_i: int = 2
        label_y: str = "$\theta_{{c}}$ [deg]"
        fig_i, ax_i = plot_tools.mk_canvas(
            x_range:=(0, self.xvg_df.iloc[:, 0].max()/1000), height_ratio=2.5)
        print(np.mean(self.xvg_df.iloc[:, column_i][:200]))
        ax_i.set_xlim(x_range[0]-2, x_range[1]+2)
        ax_i.hlines(y:=np.mean(self.xvg_df.iloc[:, column_i]),
                    xmin=x_range[0]-2, xmax=x_range[1]+2, label=rf'$\theta_{{c,ave}}$={y:.2f}')
        ax_i.plot(self.xvg_df.iloc[:, 0]/1000,
                  self.xvg_df.iloc[:, column_i], 'k', lw=0.85,
                  label=rf'$\theta_{{c}}$', zorder=0)
        ax_i.set_xlabel('Time [ns]')
        ax_i.set_ylabel(rf'$\theta_{{c}}$ [deg]')
        plot_tools.save_close_fig(fig_i, ax_i, 'angle_contact_5.png')
        plt.show()

    def make_graph(self) -> None:
        """make the canvas here"""
        fig_i, ax_i = plot_tools.mk_canvas(
            x_range:=(0, self.xvg_df.iloc[:, 0].max()/1000), height_ratio=2.5)
        ax_i.set_xlim(x_range[0]-2, x_range[1]+2)
        ax_i.hlines(y:=np.mean(self.xvg_df.iloc[:, 1])/20,
                    xmin=x_range[0]-2, xmax=x_range[1]+2, label=rf'$\gamma_{{ave}}$={y:.2f}')
        ax_i.plot(self.xvg_df.iloc[:, 0]/1000,
                  self.xvg_df.iloc[:, 1]/20, 'k', lw=0.85,
                  label=r'$\gamma$', zorder=0)
        ax_i.set_xlabel('Time [ns]')
        ax_i.set_ylabel(r'$\gamma$ [mN/m]')
        plot_tools.save_close_fig(fig_i, ax_i, 'tension_15.png')
        plt.show()



if __name__ == "__main__":
    PlotXvg(sys.argv[1], log=logger.setup_logger('plot_xvg.log'))
