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


# Constants for Elsevier's guidelines
MINIMAL_SIZE_PT = 85
SINGLE_COLUMN_PT = 255
ONE_AND_HALF_COLUMN_PT = 397
DOUBLE_COLUMN_PT = 539
FONT_SIZE_PT = 7
SUB_SUPER_FONT_SIZE_PT = 6
DPI_HALFTONE = 300
DPI_COMBINATION = 500
DPI_LINE_ART = 1000
GOLDEN_RATIO = (1 + 5 ** 0.5) / 2


def set_figure_height(width: int,
                      aspect_ratio: float = GOLDEN_RATIO
                      ) -> int:
    """Set the figure height based on the width and aspect ratio"""
    return int(width / aspect_ratio)


def set_figure_size(size_type: str
                    ) -> tuple[int, int]:
    """Set the size of the figure based on the size type"""
    sizes: dict[str, tuple[int, int]] = {
        "minimal": (MINIMAL_SIZE_PT,
                    set_figure_height(MINIMAL_SIZE_PT)),
        "single_column": (SINGLE_COLUMN_PT,
                          set_figure_height(SINGLE_COLUMN_PT)),
        "one_half_column": (ONE_AND_HALF_COLUMN_PT,
                            set_figure_height(ONE_AND_HALF_COLUMN_PT)),
        "double_column": (DOUBLE_COLUMN_PT,
                          set_figure_height(DOUBLE_COLUMN_PT))
    }
    return sizes.get(size_type, (SINGLE_COLUMN_PT,
                     set_figure_height(SINGLE_COLUMN_PT)))
