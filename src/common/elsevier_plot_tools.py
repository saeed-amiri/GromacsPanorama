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

import matplotlib.pyplot as plt

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
GOLDEN_RATIO: float = (1 + 5 ** 0.5) / 2
# Lines and markers
LINE_WIDTH: float = 1.0
MARKER_SIZE: float = 2.0
LINE_STYLES: list[str] = ['-', '--', '-.', ':']
LINE_COLORS: list[str] = ['black', 'red', 'blue', 'green', 'orange', 'purple']


def set_figure_height(width: int,
                      aspect_ratio: float = GOLDEN_RATIO
                      ) -> int:
    """Set the figure height based on the width and aspect ratio"""
    return int(width / aspect_ratio)


def set_figure_size(size_type: str
                    ) -> tuple[int, int]:
    """Set the size of the figure based on the size type"""
    sizes: dict[str, tuple[int, int]] = {
        "minimal": (
            MINIMAL_SIZE_PT / POINT_PER_INCH,
            set_figure_height(MINIMAL_SIZE_PT / POINT_PER_INCH)
            ),
        "single_column": (
            SINGLE_COLUMN_PT / POINT_PER_INCH,
            set_figure_height(SINGLE_COLUMN_PT / POINT_PER_INCH)
            ),
        "one_half_column": (
            ONE_AND_HALF_COLUMN_PT / POINT_PER_INCH,
            set_figure_height(ONE_AND_HALF_COLUMN_PT / POINT_PER_INCH)
            ),
        "double_column": (
            DOUBLE_COLUMN_PT / POINT_PER_INCH,
            set_figure_height(DOUBLE_COLUMN_PT / POINT_PER_INCH)
            )
    }
    return sizes.get(size_type, (SINGLE_COLUMN_PT,
                     set_figure_height(SINGLE_COLUMN_PT)))


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
              dpi: int = DPI_HALFTONE
              ) -> tuple[plt.Figure, plt.Axes]:
    """Create a canvas for the figure"""
    fig_i, ax_i = plt.subplots(figsize=set_figure_size(size_type))
    set_dpi(dpi)
    ax_i = set_font_size(ax_i)
    return fig_i, ax_i


def save_close_fig(fig: plt.Figure,
                   fname: str,
                   dpi: int = DPI_HALFTONE,
                   loc: str = 'upper right'
                   ) -> None:
    """Save and close the figure"""
    fig.axes[0].legend(loc=loc, fontsize=FONT_SIZE_PT)
    fig.savefig(fname,
                dpi=dpi,
                pad_inches=0.1,
                bbox_inches='tight'
                )
    plt.close(fig)
