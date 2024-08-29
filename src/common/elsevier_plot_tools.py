"""
This module contains constants and functions to create plots that
follow Elsevier's guidelines.
Elsevier's aim is to have a uniform look for all artwork contained in
a single article. It is important to be aware of the journal style, as
some of our publications have special instructions beyond the common
guidelines given here. Please check the journal-specific guide for
authors (available from the homepage of the journal in question).

As a general rule, the lettering on the artwork should have a finished,
printed size of 7 pt for normal text and no smaller than 6 pt for
subscript and superscript characters. Smaller lettering will yield text
that is hardly legible. This is a rule-of-thumb rather than a strict
rule. There are instances where other factors in the artwork (e.g.,
tints and shadings) dictate a finished size of perhaps 10 pt.

When Elsevier decides on the size of a line art graphic, in addition
to the lettering, there are several other factors to assess. These all
have a bearing on the reproducibility/readability of the final artwork.
Tints and shadings have to be printable at finished size. All relevant
detail in the illustration, the graph symbols (squares, triangles,
circles, etc.) and a key to the diagram (explaining the symbols used)
must be discernible.

Sizing of halftones (photographs, micrographs, etc.) can normally cause
more problems than line art. It is sometimes difficult to know what an
author is trying to emphasize on a photograph, so you can help us by
identifying the important parts of the image, perhaps by highlighting
the relevant areas on a photocopy.

The best advice that we give to our graphics suppliers is to not
over-reduce halftones, and pay attention to magnification factors or
scale bars on the artwork, and compare them with the details given in
the artwork itself. If a collection of artwork contains more than one
halftone, again make sure that there is consistency in size between
similar diagrams. Halftone/line art combinations are difficult to size,
as factors for one may be detrimental for the other part. In these
cases, the author can help by suggesting an appropriate final size for
the combination (single, 1.5, two column). Number of pixels versus
resolution and print size, for bitmap images

Image resolution, number of pixels and print size are related
mathematically:

Pixels = Resolution (DPI) * Print size (in inches)

300 DPI for halftone images; 500 DPI for combination art; 1000 DPI for
line art. 72 points in one inch. Number of pixels versus resolution
and print size, for bitmap images Image resolution, number of pixels
and print size are related mathematically:
Pixels = Resolution (DPI) * Print size (in inches) 300 DPI
for halftone images; 500 DPI
for combination art; 1000 DPI for line art.
72 points in one inch.
* Minimal size 30 mm 85 pt
* Single column 90 mm 255 pt
* 1.5 column 140 mm 397 pt
* Double column (full width) 190 mm 539 pt

"""

import typing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Constants for Elsevier's guidelines
# Fig sizes in points
MINIMAL_SIZE_PT: int = 85
SINGLE_COLUMN_PT: int = 255
ONE_AND_HALF_COLUMN_PT: int = 397
DOUBLE_COLUMN_PT: int = 539
POINT_PER_INCH: int = 72
# Font sizes in points
FONT_SIZE_PT: int = 7
SUB_SUPER_FONT_SIZE_PT: int = 6
LABEL_FONT_SIZE_PT: int = 10
# DPI for different types of images
DPI_HALFTONE: int = 300
DPI_COMBINATION: int = 500
DPI_LINE_ART: int = 1000
# Golden ratio
GOLDEN_RATIO: float = (1 + 5 ** 0.5) / 2.0
# Lines and markers
LINE_WIDTH: float = 1.0
LINE_STYLES: list[str] = ['--', '-', '-.', ':']
LINE_COLORS: list[str] = \
    ['black', 'darkred', 'royalblue', 'green', 'orange', 'purple']
MARKER_SIZE: float = 2.0
MARKER_COLORS: list[str] = \
    ['black', 'darkred', 'royalblue', 'green', 'orange', 'purple']
MARKER_SHAPES: list[str] = ['o', 's', 'D', '^', 'v', 'x']
# Output file format
IMG_FORMAT: str = 'jpg'
# Python list containing a palette of black shades
BLACK_SHADES = [
    "#000000",  # Pure Black
    "#1C1C1C",  # Very Dark Gray
    "#383838",  # Dark Gray
    "#545454",  # Slightly Darker Gray
    "#707070",  # Mid Gray
    "#8C8C8C",  # Medium Light Gray
    "#A8A8A8",  # Light Gray
    "#C4C4C4",  # Very Light Gray
    "#E0E0E0",  # Almost White Gray
    "#FFFFFF"   # White (for the sake of gradient completion)
]

# Python list containing a gradient palette from dark red to dark blue
DARK_RGB_COLOR_GRADIENT = [
    "#8B0000",  # Dark Red
    "#9B111E",  # Firebrick
    "#A52A2A",  # Brown
    "#B22222",  # Firebrick4
    "#CD5C5C",  # IndianRed3
    "#006400",  # Dark Green
    "#008000",  # Green
    "#228B22",  # ForestGreen
    "#2E8B57",  # SeaGreen
    "#00688B",  # DeepSkyBlue4
    "#00008B"   # Dark Blue
]

CLEAR_COLOR_GRADIENT = [
    "#8B0000",  # Dark Red
    "#9932CC",  # Dark Orchid
    "#483D8B",  # Dark Slate Blue
    "#00008B",  # Dark Blue
    "#008B8B",  # Dark Cyan
    "#2E8B57",  # Sea Green
    "#556B2F",  # Dark Olive Green
    "#FF8C00",  # Dark Orange
    "#800000",  # Maroon
    "#B22222",  # Firebrick
]

LINESTYLE_TUPLE: \
   list[tuple[str, tuple[None] | tuple[int, tuple[int, ...]]]] = [
    ('long dash with offset', (5, (5, 3))),
    ('loosely dashdotted',    (0, (3, 3, 1, 2))),
    ('dashdotted',            (0, (3, 5, 1, 5))),
    ('densely dashdotted',    (0, (3, 1, 1, 1))),
    ('dashdotdotted',         (0, (3, 3, 1, 3, 1, 3))),
    ('loosely dashdotdotted', (0, (3, 4, 3, 4, 3, 4))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
    ('loosely dotted',        (0, (1, 3))),
    ('dashed',                (0, (5, 5))),
    ('densely dashed',        (0, (5, 1))),
    ('densely dotted',        (0, (1, 1))),
    ('solid',                 (0, ())),
    ]


def set_figure_height(width: float,
                      aspect_ratio: float = 1.0
                      ) -> int:
    """Set the figure height based on the width and aspect ratio"""
    return int(width / (aspect_ratio * GOLDEN_RATIO))


def set_figure_size(size_type: str,
                    aspect_ratio: float = 1.0
                    ) -> tuple[float, float]:
    """Set the size of the figure based on the size type"""
    sizes: dict[str, tuple[float, float]] = {
        "minimal": (
            MINIMAL_SIZE_PT / POINT_PER_INCH,
            set_figure_height(MINIMAL_SIZE_PT / POINT_PER_INCH,
                              aspect_ratio)),
        "single_column": (
            SINGLE_COLUMN_PT / POINT_PER_INCH,
            set_figure_height(SINGLE_COLUMN_PT / POINT_PER_INCH,
                              aspect_ratio)),
        "one_half_column": (
            ONE_AND_HALF_COLUMN_PT / POINT_PER_INCH,
            set_figure_height(ONE_AND_HALF_COLUMN_PT / POINT_PER_INCH,
                              aspect_ratio)),
        "double_column": (
            DOUBLE_COLUMN_PT / POINT_PER_INCH,
            set_figure_height(DOUBLE_COLUMN_PT / POINT_PER_INCH,
                              aspect_ratio)),
        "double_height": (
            SINGLE_COLUMN_PT / POINT_PER_INCH,
            set_figure_height(SINGLE_COLUMN_PT / POINT_PER_INCH,
                              aspect_ratio/1.66)),
    }
    return sizes.get(size_type, (SINGLE_COLUMN_PT,
                     set_figure_height(SINGLE_COLUMN_PT, aspect_ratio)))


def set_font_size(ax_i: plt.Axes,
                  font_size: int = FONT_SIZE_PT,
                  sub_super_font_size: int = SUB_SUPER_FONT_SIZE_PT
                  ) -> plt.Axes:
    """Set the font size for the axes"""
    for item in ([ax_i.title, ax_i.xaxis.label, ax_i.yaxis.label] +
                 ax_i.get_xticklabels() + ax_i.get_yticklabels()):
        item.set_fontsize(font_size)
    for item in ax_i.get_xticklabels() + ax_i.get_yticklabels():
        item.set_fontsize(sub_super_font_size)
    return ax_i


def set_dpi(dpi: int) -> None:
    """Set the DPI for the figure"""
    plt.rcParams['figure.dpi'] = dpi


def mk_canvas(size_type: str,
              dpi: int = DPI_HALFTONE,
              aspect_ratio: float = 1.0
              ) -> tuple[plt.Figure, plt.Axes]:
    """Create a canvas for the figure"""
    fig_i, ax_i = \
        plt.subplots(figsize=set_figure_size(size_type, aspect_ratio))
    set_dpi(dpi)
    ax_i = set_font_size(ax_i)
    return fig_i, ax_i


def mk_canvas_multi(size_type: str,
                    n_rows: int = 1,
                    n_cols: int = 1,
                    aspect_ratio: float = 1.0,
                    dpi: int = DPI_HALFTONE
                    ) -> tuple[plt.Figure, plt.Axes |
                               np.ndarray[typing.Any, typing.Any]]:
    """Create a canvas for a multi-panel figure"""
    fig_i, axs_i = \
        plt.subplots(n_rows,
                     n_cols,
                     figsize=set_figure_size(size_type, aspect_ratio))
    set_dpi(dpi)
    for ax_i in axs_i:
        ax_i = set_font_size(ax_i)
    return fig_i, axs_i


def remove_mirror_axes(ax: plt.Axes
                       ) -> None:
    """Remove the top and right spines and ticks from a matplotlib Axes"""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def save_close_fig(fig: plt.Figure,
                   fname: str,
                   dpi: int = DPI_HALFTONE,
                   loc: str = 'upper right',
                   horizontal_legend: bool = False,
                   show_legend: bool = True,
                   close_fig: bool = True
                   ) -> None:
    """Save and close the figure"""
    # pylint: disable=too-many-arguments
    if show_legend:
        if horizontal_legend:
            ncol = len(fig.axes[0].get_legend_handles_labels()[0])
            y_loc = 0.23
            fig.legend(bbox_to_anchor=(0.9, y_loc),
                       fontsize=FONT_SIZE_PT,
                       ncol=ncol)
        else:
            ncol = 1
            y_loc = 1
            fig.axes[0].legend(loc=loc,
                               fontsize=FONT_SIZE_PT,
                               ncol=ncol)
    fig.savefig(fname,
                dpi=dpi,
                pad_inches=0.1,
                bbox_inches='tight'
                )
    if close_fig:
        plt.close(fig)


def set_custom_line_prop(
        ax: plt.Axes,
        color: str = 'black',
        marker: str = 'o',
        linestyle: str = ':',
        markersize: float = 0,
        linewidth: float = LINE_WIDTH
) -> None:
    """Set a custom line style for a matplotlib Axes"""
    # pylint: disable=too-many-arguments
    ax.set_prop_cycle(color=[color],
                      marker=[marker],
                      linestyle=[linestyle],
                      markersize=[markersize],
                      linewidth=[linewidth])


def generate_shades(nr_shade: int,
                    color_hex: str = '#000000',
                    min_shade: int = 0,
                    ) -> list[str]:
    """
    Generate shades of a color in hex
    black: #000000
    white: #FFFFFF
    red: #FF0000
    green: #00FF00
    blue: #0000FF
    yellow: #FFFF00
    cyan: #00FFFF
    magenta: #FF00FF
    Dark red: #8B0000
    Dark green: #006400
    Dark blue: #00008B
    Dark orange: #FF8C00
    Dark cyan: #008B8B
    Royal blue: #4169E1
    """
    # For black, generate shades of gray up to white
    if color_hex == '#000000':
        shades = []
        step = 255 // (nr_shade - 1)  # Ensure the last shade is white
        for i in range(nr_shade):
            gray_value = min(i * step, 255)
            shades.append(mcolors.to_hex([gray_value / 255] * 3))
    else:
        # Convert hex color to RGB for other colors
        rgb = mcolors.hex2color(color_hex)
        # Convert RGB to 0-255 scale
        rgb = tuple(int(255 * x) for x in rgb)

        shades = []
        for i in range(nr_shade):
            # Calculate shade
            shade = [
                max(min(c - (c // nr_shade) * i, 255), min_shade) for c in rgb]
            # Convert shade back to hex and append to list
            shades.append(mcolors.to_hex([x / 255 for x in shade]))

    return shades[::-1]
