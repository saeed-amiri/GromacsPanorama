"""tools for ploting"""

import numpy as np
import matplotlib
import matplotlib.pylab as plt
import common.static_info as stinfo


def set_sizes(width: float,  # Width of the plot in points
              fraction: float = 1,
              height_ratio: float = 3
              ) -> tuple[float, float]:
    """
    Calculate figure dimensions based on width and fraction.

    This function calculates the dimensions of a figure based on the
    desired width and an optional fraction. It uses the golden ratio
    to determine the height.

    Args:
        width (float): The width of the plot in points.
        fraction (float, optional): A fraction to adjust the width.
        Default is 1.

    Returns:
        tuple[float, float]: A tuple containing the calculated width
        and height in inches for the figure dimensions.
    """
    fig_width_pt = width*fraction
    inches_per_pt = 1/72.27
    golden_ratio = (5**0.5 - 1)
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio / height_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


def mk_circle(radius: float,
              center: tuple[float, float] = (0, 0)
              ) -> matplotlib.patches.Circle:
    """
    Create a dashed circle.

    This function creates a dashed circle with the specified radius and
    center coordinates.

    Args:
        radius (float): The radius of the circle.
        center (tuple[float, float], optional): The center coordinates
        of the circle. Default is (0, 0).

    Returns:
        matplotlib.patches.Circle: A `Circle` object representing the
        dashed circle.
    """
    circle = plt.Circle(center,
                        radius,
                        color='red',
                        linestyle='dashed',
                        fill=False, alpha=1)
    return circle

def mk_canvas(x_range: tuple[float, float],
              num_xticks: int = 5,
              width = stinfo.plot['width'],
              nrows: int = 1,  # Numbers of rows
              ncols: int = 1,  # Numbers of rows
              fsize: float = 0,  # Font size
              add_xtwin: bool = True,
              width_ratio: float = 1,
              height_ratio: float = 3
              ) -> tuple[plt.figure, plt.axes]:
    """
    Create a canvas for the plot.

    This function generates a canvas (figure and axes) for plotting.

    Args:
        x_range (tuple[float, ...]): Range of x-axis values.
        num_xticks (int, optional): Number of x-axis ticks.
        Default is 5.

    Returns:
        tuple[plt.figure, plt.axes]: A tuple containing the figure
        and axes objects.
    """
    fig_main, ax_main = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=set_sizes(width*width_ratio, height_ratio=height_ratio))
    # Set font for all elements in the plot)
    xticks = _set_xrange_for_xticks(x_range, num_xticks)
    for ax in np.ravel(ax_main):
        ax.set_xticks(xticks)
    # ax_main = set_x2ticks(ax_main, add_xtwin)
    ax_main = set_ax_font_label(ax_main, fsize=fsize)
    return fig_main, ax_main

def _set_xrange_for_xticks(xrange: tuple[float, float],
                           num_ticks: int
                           ) -> list[float]:
    """set the place of the ticks"""
    x_i: float = num_ticks * (int(xrange[0]/num_ticks) - 1)
    x_f: float = num_ticks * (int(xrange[1]/num_ticks) + 1)
    x_r: list[float] = np.linspace(x_i, x_f, num_ticks).tolist()
    xticks: list[float] = [0]
    xticks.extend(int(item) for item in x_r[1:])
    return xticks

def save_close_fig(fig: plt.figure,  # The figure to save,
                   axs: plt.axes,  # Axes to plot
                   fname: str,  # Name of the output for the fig
                   loc: str = 'upper right',  # Location of the legend
                   transparent: bool = False,
                   legend: bool = True,
                   if_close: bool = True,
                   legend_font_size: int = 7,
                   bbox_to_anchor=None
                   ) -> None:
    """
    Save the figure and close it.

    This method saves the given figure and closes it after saving.

    Args:
        fig (plt.figure): The figure to save.
        axs (plt.axes): The axes to plot.
        fname (str): Name of the output file for the figure.
        loc (str, optional): Location of the legend. Default is
        'upper right'.
    """
    try:
        handles, labels = axs.get_legend_handles_labels()
    except AttributeError:
        handles = None

    if handles:
        axs.legend(
            handles, labels, loc=loc, fontsize=legend_font_size)
        if not legend:
            for ax_i in np.ravel(axs):
                legend = ax_i.legend(loc=loc, fontsize=legend_font_size)
            legend.set_bbox_to_anchor(bbox_to_anchor)
        else:
            for ax_j in np.ravel(axs):
                legend = ax_j.legend(loc=loc, fontsize=legend_font_size)
            if bbox_to_anchor is not None:
                legend.set_bbox_to_anchor(bbox_to_anchor)

    fig.savefig(fname,
                dpi=300,
                pad_inches=0.1,
                edgecolor='auto',
                bbox_inches='tight',
                transparent=transparent
                )
    if if_close:
        plt.close(fig)

def set_x2ticks(ax_main: plt.axes,  # The axes to wrok with
                add_xtwin: bool = True
                    ) -> plt.axes:
    """
    Set secondary x-axis ticks.

    This method sets secondary x-axis ticks for the given main axes.

    Args:
        ax_main (plt.axes): The main axes to work with.

    Returns:
        plt.axes: The modified main axes.
    """
    if add_xtwin:
        for ax_j in np.ravel(ax_main):
        # Set twiny
            ax2 = ax_j.twiny()
            ax2.set_xlim(ax_j.get_xlim())
            # Synchronize x-axis limits and tick positions
            ax2.xaxis.set_major_locator(ax_j.xaxis.get_major_locator())
            ax2.xaxis.set_minor_locator(ax_j.xaxis.get_minor_locator())
            ax2.set_xticklabels([])  # Remove the tick labels on the top x-axis
            ax2.tick_params(axis='x', direction='in')
    return ax_main

def set_y2ticks(ax_main: plt.axes  # The axes to wrok with
                    ) -> plt.axes:
    """
    Set secondary y-axis ticks.

    This method sets secondary y-axis ticks for the given main axes.

    Args:
        ax_main (plt.axes): The main axes to work with.

    Returns:
        plt.axes: The modified main axes.
    """
    # Reset the y-axis ticks and locators
    ax3 = ax_main.twinx()
    ax3.set_ylim(ax_main.get_ylim())
    # Synchronize y-axis limits and tick positions
    ax3.yaxis.set_major_locator(ax_main.yaxis.get_major_locator())
    ax3.yaxis.set_minor_locator(ax_main.yaxis.get_minor_locator())
    ax3.set_yticklabels([])  # Remove the tick labels on the right y-axis
    ax3.tick_params(axis='y', direction='in')
    for ax_i in [ax_main, ax3]:
        ax_i.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        ax_i.yaxis.set_minor_locator(
            matplotlib.ticker.AutoMinorLocator(n=5))
        ax_i.tick_params(which='minor', direction='in')
    return ax_main

def set_ax_font_label(ax_main: plt.axes,  # Main axis to set parameters
                      fsize: int = 0,  # font size if called with font
                      x_label = 'frame index',
                      y_label = 'z [A]'
                      ) -> plt.axes:
    """
    Set font and labels for the plot axes.

    This method sets font size and labels for the plot axes.

    Args:
        ax_main (plt.axes): The main axis to set parameters for.
        fsize (int, optional): Font size if called with font.
        Default is 0.

    Returns:
        plt.axes: The modified main axis.
    """
    if fsize == 0:
        fontsize = 14
    else:
        fontsize = fsize
    for ax_i in np.ravel(ax_main):
        ax_i.set_xlabel(x_label, fontsize=fontsize)
        ax_i.set_ylabel(y_label, fontsize=fontsize)

        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.size'] = fontsize
        ax_i.tick_params(axis='x', labelsize=fontsize)
        ax_i.tick_params(axis='y', labelsize=fontsize)
    return ax_main
