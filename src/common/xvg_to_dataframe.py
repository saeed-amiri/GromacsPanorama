"""reading xvg files and return it as ddaframe"""

import re
import sys
import typing
import pandas as pd

from common import logger
from common import my_tools
from common.colors_text import TextColor as bcolors


class XvgParser:
    """reading and getting info from xvg file"""

    info_msg: str = 'Message from XvgParser:\n'  # Meesage in methods to log

    fname: str  # Name of the input file
    title: str = ''  # Title of the data
    xaxis: str = ''  # Label of the x axis
    yaxis: str = ''  # Label of the y axis

    def __init__(self,
                 fname: str,  # Name of the xvg file
                 log: logger.logging.Logger
                 ) -> None:
        self.fname = fname

        self.get_xvg()
        print(self.__dict__)
        self.write_log_msg(log)

    def get_xvg(self) -> pd.DataFrame:
        """parse xvg file"""
        columns_names: list[str] = []
        with open(self.fname, 'r', encoding='utf8') as f_r:
            while True:
                line: str = f_r.readline().strip()
                if line.startswith('#'):
                    pass
                elif line.startswith('@'):
                    if (column := self.parse_header(line)):
                        columns_names.append(column)
                if not line:
                    break
        print(columns_names)

    def parse_header(self,
                     line: str  # Read line from the file
                     ) -> typing.Union[str, None]:
        """get info from the header with @"""
        line = my_tools.drop_string(line, '@').strip()
        if line.startswith('title'):
            self.title = self.clean_line(line, ['title'])
        elif line.startswith('xaxis'):
            self.xaxis = self.clean_line(line, ['xaxis', 'label'])
        elif line.startswith('yaxis'):
            self.yaxis = self.clean_line(line, ['yaxis', 'label'])
        else:
            pattern = r'^s\d+'
            if (match := re.match(pattern, line)):
                column_i = match.group(0)
                column_iname = self.clean_line(line, [column_i, 'legend'])
                return column_iname
        return None

    @staticmethod
    def clean_line(line: str,  # line to clean
                   droplets: list[str]  # Strings to drop from the line
                   ) -> str:
        """clean up the line"""
        tmp_str: str = line
        for droplet in droplets:
            tmp_str = my_tools.drop_string(tmp_str, droplet)
        return my_tools.extract_string(tmp_str)[0]

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == "__main__":
    XvgParser(sys.argv[1], log=logger.setup_logger(log_name='xvg.log'))
