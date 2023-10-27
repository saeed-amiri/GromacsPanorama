"""tools used in multiple scripts"""
import os
import re
import sys
import typing
import logger
from colors_text import TextColor as bcolors


def check_file_exist(fname: str,  # Name of the file to check
                     log: logger.logging.Logger  # log the error
                     ) -> None:
    """check if the file exist, other wise exit"""
    if not os.path.exists(fname):
        log.error(f'Error! `{fname}` dose not exist.')
        sys.exit(f'{bcolors.FAIL}{__name__}: '
                 f'(Error! `{fname}` dose not '
                 f'exist \n{bcolors.ENDC}')
    else:
        log.info(f'Checking: `{fname}`')


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
