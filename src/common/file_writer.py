"""helper functions to write files with different formats:
xvg, gro, pdb, ...
"""

import os
import warnings
import datetime
import pandas as pd

from common import logger
from common.colors_text import TextColor as bcolors


def write_xvg(df_i: pd.DataFrame,
              log: logger.logging.Logger,
              fname: str = 'df.xvg',
              extra_comments: str = '',
              xaxis_label: str = 'Frame index',
              yaxis_label: str = 'Varies',
              title: str = 'Contact information'
              ) -> None:
    """
    Write the data into xvg format
    Raises:
        ValueError: If the DataFrame has no columns.
    """
    if df_i.columns.empty:
        log.error(msg := "\tThe DataFrame has no columns.\n")
        raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}\n')
    if df_i.empty:
        log.warning(
            msg := f"The df is empty. `{fname}` will not contain data.")
        warnings.warn(msg, UserWarning)

    columns: list[str] = df_i.columns.to_list()

    header_lines: list[str] = [
        f"# {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f'# Written by {write_xvg.__module__}',
        f"# Current directory: {os.getcwd()}",
        f"# {extra_comments}\n"
        f'@   title "{title}"',
        f'@   xaxis label "{xaxis_label}"',
        f'@   yaxis label "{yaxis_label}"',
        '@TYPE xy',
        '@ view 0.15, 0.15, 0.75, 0.85',
        '@legend on',
        '@ legend box on',
        '@ legend loctype view',
        '@ legend 0.78, 0.8',
        '@ legend length 2'
    ]
    legend_lines: list[str] = \
        [f'@ s{i} legend "{col}"' for i, col in enumerate(df_i.columns)]

    with open(fname, 'w', encoding='utf8') as f_w:
        for line in header_lines + legend_lines:
            f_w.write(line + '\n')
        df_i.to_csv(f_w,
                    sep=' ',
                    index=True,
                    header=None,
                    na_rep='NaN',
                    float_format='%.3f',
                    quoting=3,
                    escapechar=" ",
                    )