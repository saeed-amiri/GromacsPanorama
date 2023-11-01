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
    columns_names: list[str] = []
    xvg_df: pd.DataFrame  # The final dataframe

    def __init__(self,
                 fname: str,  # Name of the xvg file
                 log: logger.logging.Logger
                 ) -> None:
        self.fname = fname
        self.xvg_df = self.get_xvg(log)
        print(self.xvg_df)
        self.write_log_msg(log)

    def get_xvg(self,
                log: logger.logging.Logger
                ) -> pd.DataFrame:
        """parse xvg file"""
        data_list: list[list[str]] = []
        data_block: bool = False
        with open(self.fname, 'r', encoding='utf8') as f_r:
            for line in f_r:
                line = line.strip()
                if line:
                    if re.match(pattern=r'^\d', string=line):
                        data_list.append(self.parse_data_block(line))
                        data_block = True
                    elif not data_block:
                        if line.startswith('#'):
                            pass
                        elif line.startswith('@'):
                            if (column := self.parse_header(line)):
                                self.columns_names.append(column)
                    else:
                        msg = ('\tError! Something is wrong! String '
                               'after data\n')
                        log.error(msg)
                        sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
        return self.make_df(data_list)

    def make_df(self,
                data_list: list[list[str]],
                ) -> pd.DataFrame:
        """make the dataframe from datalist"""
        columns_names: list[str] = []
        columns_names.append(self.xaxis)
        columns_names.extend(self.columns_names)
        columns_names = \
            [re.sub(r'\s+', ' ', item) for item in columns_names]
        self.columns_names = [item.replace(' ', '_') for item in columns_names]
        xvg_df = pd.DataFrame(data=data_list, columns=self.columns_names)
        xvg_df.iloc[:, 0] = xvg_df.iloc[:, 0].astype(int)
        xvg_df.iloc[:, 1:] = xvg_df.iloc[:, 1:].astype(float)
        return xvg_df

    @staticmethod
    def parse_data_block(line: str
                         ) -> list[str]:
        """parse data block by replacing multispaces with one space"""
        tmp: str = re.sub(r'\s+', ' ', line)
        return tmp.split(' ')

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
        self.info_msg += (f'\tThe input file: `{self.fname}`\n'
                          f'\tThe title is: `{self.title}`\n'
                          f'\tThe xaxis is: `{self.xaxis}`\n'
                          f'\tThe yaxis is: `{self.yaxis}`\n'
                          f'\tThe columns are: `{self.columns_names}`\n')
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == "__main__":
    XvgParser(sys.argv[1], log=logger.setup_logger(log_name='xvg.log'))
