"""tools used in multiple scripts"""
import os
import re
import sys
import typing
import warnings

import pandas as pd

from common import logger
from common.colors_text import TextColor as bcolors


class InvalidFileExtensionError(Exception):
    """file extension error"""


def check_file_exist(fname: str,  # Name of the file to check
                     log: logger.logging.Logger,  # log the error,
                     if_exit: bool = True
                     ) -> typing.Union[None, bool]:
    """check if the file exist, other wise exit"""
    if not os.path.exists(fname):
        log.error(f'Error! `{fname}` dose not exist.')
        if if_exit:
            sys.exit(f'{bcolors.FAIL}{__name__}: '
                     f'(Error! `{fname}` dose not '
                     f'exist \n{bcolors.ENDC}')
        else:
            log.warning('\tThere was a problem in reading the file; return\n')
            return False
    else:
        log.info(msg := f'`{fname}` exist.')
        print(f'{bcolors.OKBLUE}my_tools:\n\t{msg}{bcolors.ENDC}\n')


def check_file_extension(fname: str,  # Name of the file to check
                         extension: str,  # Extension of expected file
                         log: logger.logging.Logger
                         ) -> None:
    """check if the file name is a correct one"""
    if (fname_exten := fname.split('.')[1]) == extension:
        pass
    else:
        msg = (f'\tThe provided file has the extension: `{fname_exten}` '
               f'which is not `{extension}`\n'
               f'\tProvid a file with correct extension\n')
        log.error(msg)
        raise InvalidFileExtensionError(
            f'{bcolors.FAIL}{msg}{bcolors.ENDC}')


def check_file_reanme(fname: str,  # Name of the file to check
                      log: logger.logging.Logger
                      ) -> str:
    """checking if the file fname is exist and if, rename the old one"""
    # Check if the file already exists
    if os.path.isfile(fname):
        # Generate a new file name by appending a counter
        counter = 1
        while os.path.isfile(f"{fname}_{counter}"):
            counter += 1
        new_fname = f"{fname}_{counter}"

        # Rename the existing file
        os.rename(fname, new_fname)
        print(f'{bcolors.CAUTION}{__name__}:\n\tRenaming an old `{fname}` '
              f' file to: `{new_fname}`{bcolors.ENDC}')
        log.info(f'renmaing an old `{fname}` to `{new_fname}`')
    return fname


def drop_string(input_string: str,
                string_to_drop: str
                ) -> str:
    """drop strings"""
    output_string = input_string.replace(string_to_drop, "")
    return output_string


def extract_string(input_string: str) -> list[typing.Any]:
    """return matches str"""
    pattern = r'"(.*?)"'
    matches = re.findall(pattern, input_string)
    return matches


def clean_string(input_string: str) -> str:
    """Remove special characters at the beginning and end of the string"""
    cleaned_string: str = \
        re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', input_string)

    # Replace special characters in the middle with underscores
    cleaned_string = re.sub(r'[^a-zA-Z0-9]+', '_', cleaned_string)

    return cleaned_string


def write_xvg(df_i: pd.DataFrame,
              log: logger.logging.Logger,
              extra_msg: list[str],
              fname: str = 'df.xvg',
              write_index: bool = True,
              x_axis_label: str = 'Frame index',
              y_axis_label: str = 'Varies',
              title: str = 'Contact information'
              ) -> None:
    """
    Write the data into xvg format
    Raises:
        ValueError: If the DataFrame has no columns.
    """
    if df_i is None:
        log.error(msg := f'{__name__}\tThe DataFarme is `None`!\n')
        sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

    if df_i.columns.empty:
        log.error(msg := "\tThe DataFrame has no columns.\n")
        raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}\n')
    if df_i.empty:
        log.warning(
            msg := f"The df is empty. `{fname}` will not contain data.")
        warnings.warn(msg, UserWarning)

    columns: list[str] = df_i.columns.to_list()

    comment_lines: list[str] = [
        '# Written by my_tools.write_xvg\n',
        f'# Current directory: {os.getcwd()}',
    ] + extra_msg

    header_lines: list[str] = [
        f'@   title "{title}"',
        f'@   xaxis label "{x_axis_label}"',
        f'@   yaxis label "{y_axis_label}"',
        '@TYPE xy',
        '@ view 0.15, 0.15, 0.75, 0.85',
        '@legend on',
        '@ legend box on',
        '@ legend loctype view',
        '@ legend 0.78, 0.8',
        '@ legend length 2'
    ]
    legend_lines: list[str] = \
        [f'@ s{i} legend "{col}"' for i, col in enumerate(columns)]
    
    log.info(f'\t`{fname}` is written succsssfuly\n')

    with open(fname, 'w', encoding='utf8') as f_w:
        f_w.writelines([line + '\n' for line in
                        comment_lines + header_lines + legend_lines])
        df_i.to_csv(f_w, sep=' ', index=write_index, header=None, na_rep='NaN')
