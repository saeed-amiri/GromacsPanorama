"""this script reads the "topol.top" file to get the paths of the itp
files and determine the number of each residue in the system; it is
necessary to refer to the topology file. Remember that since I moved
one directory, the "itp" files will be located one back."""

import os
import json
import typing
import common.logger as logger
import common.my_tools as my_tools
import common.static_info as stinfo
from common.colors_text import TextColor as bcolors


class ReadTop:
    """read the topology file;
    It should named: topol.top"""

    info_msg: str = 'Message:\n'  # To log the info

    itp_paths: dict[str, str]  # To save the paths of itp files
    mols_num: dict[str, int]  # To get all residues number

    def __init__(self,
                 log: logger.logging.Logger
                 ) -> None:
        self.fanme: str = stinfo.topo['fname']
        self.itp_paths, self.mols_num = self.get_top(log)
        self.__write_msg(log)

    def get_top(self,
                log: logger.logging.Logger
                ) -> tuple[dict[str, str], dict[str, int]]:
        """read the top file and return numbers of each residue and
        paths"""
        # Check the input file
        my_tools.check_file_exist(self.fanme, log)
        self.info_msg += f'\tReading topology file: `{self.fanme}`\n'
        line: typing.Any  # Line from the file
        restraints: bool = False  # To pass the restrains parts
        molecule: bool = False  # To get the number of each residues
        itp_paths: dict[str, str]  # To save the paths of itp files
        paths: list[str] = []  # To save the paths
        mols_num: dict[str, int] = {}  # To get all residues number
        with open(self.fanme, 'r', encoding='utf8') as f_r:
            while True:
                line = f_r.readline()
                if line.strip():
                    line = line.strip()
                    if line.startswith(';'):
                        pass
                    elif line.startswith('#ifdef'):
                        restraints = True
                    elif line.startswith('#include'):
                        if not restraints:
                            paths.append(self.__get_path(line))
                    elif line.startswith('#endif'):
                        restraints = False
                    elif line.startswith('[ molecules ]'):
                        molecule = True
                    elif molecule:
                        key, value = self.__get_nums(line)
                        mols_num[key] = value
                    else:
                        pass
                elif not line:
                    break
        itp_paths = self.__get_itp(paths, list(mols_num.keys()))
        self.info_msg += \
            f'\tITP pathes:\n{json.dumps(itp_paths, indent=8)}\n'
        self.info_msg += \
            f'\tNumber of residues:\n{json.dumps(mols_num, indent=8)}\n'
        return itp_paths, mols_num

    @staticmethod
    def __get_itp(paths: list[str],  # All the paths saved from topol.top
                  residues: list[str]  # Name of the residues
                  ) -> dict[str, str]:
        """make a dict from name of the mol and thier path"""
        path_dict: dict[str, str] = {}
        for item in residues:
            for path in paths:
                if item in path:
                    path_dict[item] = path
                elif item == 'ODN' and 'ODA' in path:
                    path_dict[item] = path
                elif item == 'SOL' and 'TIP' in path:
                    path_dict[item] = path
        return path_dict

    @staticmethod
    def __get_nums(line: str  # The line contains the number of resds
                   ) -> tuple[str, int]:
        """get the number of residues from molecule section"""
        res: list[str] = [item for item in line.split(' ') if item]
        return res[0], int(res[1])

    def __get_path(self,
                   line: str  # Raw line of the file
                   ) -> str:
        """breacking the line get the itp path"""
        line = my_tools.drop_string(line, "#include")
        line = my_tools.extract_string(line)[0]
        path: str = self.__mk_path(line)
        return path

    @staticmethod
    def __mk_path(line: str  # Make real path
                  ) -> str:
        """making paths for the itp files"""
        if line.startswith("./"):
            line = line[2:]  # Remove the first two characters (./)
        path: str = os.path.join('../', line)
        return path

    def __write_msg(self,
                    log: logger.logging.Logger,  # To log info in it
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ReadTop.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    top = ReadTop(log=logger.setup_logger(stinfo.log['test_log']))
