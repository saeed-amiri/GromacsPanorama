"""reading xvg files and return it as a dataframe"""

import os
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
    title: str = ''  # Title of the data
    xaxis: str = ''  # Label of the x axis
    yaxis: str = ''  # Label of the y axis
    xvg_df: pd.DataFrame  # The final dataframe

    def __init__(self,
                 fname: str,  # Name of the xvg file
                 log: logger.logging.Logger,
                 x_type: type = int,
                 if_exit: bool = True
                 ) -> None:
        self.nr_frames: int  # Number of the frames
        self.columns_names: list[str] = []
        my_tools.check_file_exist(fname, log, if_exit)
        my_tools.check_file_extension(fname, 'xvg', log)
        self.fname: str = fname  # Name of the input file
        self.xvg_df = self.get_xvg(log, x_type)
        self.nr_frames = len(self.xvg_df.index)
        self.info_msg += (f'\tThe input file: `{self.fname}`\n'
                          f'\tThe file is: `{os.path.abspath(self.fname)}`\n'
                          f'\tThe title is: `{self.title}`\n'
                          f'\tThe xaxis is: `{self.xaxis}`\n'
                          f'\tThe yaxis is: `{self.yaxis}`\n'
                          f'\tThe columns are: `{self.columns_names}`\n'
                          f'\tNumbers of the frames are: `{self.nr_frames}`\n')
        self.write_log_msg(log)

    def get_xvg(self,
                log: logger.logging.Logger,
                x_type: type
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
        return self.make_df(data_list, log, x_type)

    def make_df(self,
                data_list: list[list[str]],
                log: logger.logging.Logger,
                x_type: type
                ) -> pd.DataFrame:
        """make the dataframe from datalist"""
        columns_names: list[str] = []
        columns_names.append(self.xaxis)
        columns_names.extend(self.columns_names)
        if (l_1 := len(columns_names)) != (l_2 := len(data_list[0])):
            msg = (f'\n\tThe number of columns` names: `{l_1}` not the same'
                   f' as the number of data columns: `{l_2}`\n')
            log.error(msg)
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
        columns_names = \
            [re.sub(r'\s+', ' ', item) for item in columns_names]
        columns_names = [item.replace(' ', '_') for item in columns_names]
        self.columns_names = \
            [my_tools.clean_string(item) for item in columns_names]
        xvg_df = pd.DataFrame(data=data_list, columns=self.columns_names)
        xvg_df = xvg_df.astype(float)
        xvg_df.iloc[:, 0] = xvg_df.iloc[:, 0].astype(x_type)
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
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == "__main__":
    XvgParser(sys.argv[1], log=logger.setup_logger(log_name='xvg.log'))
