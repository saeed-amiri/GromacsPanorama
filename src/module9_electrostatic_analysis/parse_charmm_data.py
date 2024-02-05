"""
Read the data of the radius and charges from charmm data of the
APBS server
"""

import sys
import typing
import pandas as pd

from common import logger
from common.colors_text import TextColor as bcolors


class ParseData:
    """get the data"""
    info_msg = 'Message from ParseData:\n'

    def __init__(self,
                 fname: str,
                 log: logger.logging.Logger
                 ) -> None:

        self.radius_df: pd.DataFrame = self.read_file(fname)
        self.write_msg(log)

    def read_file(self,
                  fname: str
                  ) -> pd.DataFrame:
        """parse the file and return the data"""
        lines: list[dict[str, typing.Any]] = []
        self.info_msg += f'\tReading {fname}\n'
        with open(fname, 'r', encoding='utf8') as f_r:
            while True:
                line: str = f_r.readline().strip()
                if line.startswith("#"):
                    pass
                elif len(line) > 1:
                    lines.append(self._process_line(line))
                if not line:
                    break
        return pd.DataFrame(lines)

    def _process_line(self,
                      line: str
                      ) -> dict[str, typing.Any]:
        """parse a line and return them"""
        tmp: list[str] = line.split("\t")
        return {
            'resname': tmp[0],
            'atom_name': tmp[1],
            'charge': float(tmp[2]),
            'radius': float(tmp[3]),
            'atom_type': tmp[4]
        }

    def write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ParseData.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)

if __name__ == '__main__':
    ParseData(sys.argv[1], log=logger.setup_logger('parse_charmm.log'))
