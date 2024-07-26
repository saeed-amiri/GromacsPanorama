"""
Read PQR and GRO files and based on the grid spacing, report the index
of lowest and highest grid points in each dimension for each resides.
Should be able to handle multiple files.
Should be able to run standalone.
"""

import os
import sys
import typing
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import multiprocessing as mp

import pandas as pd

from common import logger
from common import my_tools
from common import pqr_to_df
from common import gro_to_df
from common import cpuconfig
from common import file_writer
from common.colors_text import TextColor as bcolors


@dataclass
class GridConfig:
    """
    Class to hold configuration parameters
    """
    grid_number: tuple[int, int, int] = field(default_factory=lambda: (
        161, 161, 161))
    # The size of the box in angstroms
    box_size: tuple[float, float, float] = field(default_factory=lambda: (
        230.0, 230.0, 227.1))


class ResidueName(Enum):
    """
    Enum class to hold residue names
    """
    SOL = "SOL"
    ODA = "ODN"
    OIL = "D10"
    COR = "COR"
    APT = "APT"
    NA = "POT"
    CL = "CLA"


class ColumnName(Enum):
    """
    Enum class to hold column names
    """
    ATOM_ID = 'atom_id'
    ATOM_NAME = 'atom_name'
    RESIDUE_NAME = 'residue_name'
    CHAIN_ID = 'chain_id'
    RESIDUE_NUMBER = 'residue_number'
    X = 'x'
    Y = 'y'
    Z = 'z'
    CHARGE = 'charge'
    RADIUS = 'radius'


@dataclass
class AllConfig:
    """
    Class to hold all configuration parameters
    """
    accebtable_file_type: str = 'pqr'
    grid_config: "GridConfig" = field(default_factory=GridConfig)


class AnalysisStructure:
    """
    Read PQR or GRO files and based on the grid spacing, report the index
    """
    info_msg: str = "Message from AnalysisStructure:\n"
    config: AllConfig
    structure_data: dict[str, pd.DataFrame]
    grid_spacing: tuple[float, float, float]

    def __init__(self,
                 files: list[str],
                 log: logger.logging.Logger,
                 config: AllConfig = AllConfig(),
                 ) -> None:
        self.config = config
        self.structure_data = self.validate_and_process_files(files, log)
        self.get_indices(log)
        self._write_msg(log)

    def get_indices(self,
                    log: logger.logging.Logger
                    ) -> None:
        """Get the indices of the lowest and highest grid points in each
        dimension for each residue
        Each box has a diminsion which is defined in the APBS input file
        That box should be used because the grids are based on the output
        of the APBS calculations.
        Each box is has specific number of grid points in each dimension
        The min and max values of each residue should be found and then
        the indices of the lowest and highest grid points in each dimension
        should be reported.
        """
        # find the min and max of each residue
        self.grid_spacing = self.calc_grid_spacing()
        residue_boundary_grid: pd.DataFrame = \
            self.calculate_residue_boundary(log)
        global_min_max: pd.DataFrame = \
            self.calculate_columnwise_min_max(residue_boundary_grid)
        self.write_xvg_file(residue_boundary_grid, global_min_max)

    def calculate_residue_boundary(self,
                                   log: logger.logging.Logger
                                   ) -> pd.DataFrame:
        """
        Get the min and max of each residue
        """
        boundary_data_list: list[pd.DataFrame] = []
        nr_cores = cpuconfig.ConfigCpuNr(log).cores_nr
        if (nr_files := len(self.structure_data)) < nr_cores:
            nr_cores = nr_files
        with mp.Pool(nr_cores) as pool:
            results = pool.map(self.worker_get_min_max_residue,
                               self.structure_data.values())
        boundary_data_list.extend(results)
        residue_boundary_grid: pd.DataFrame = pd.concat(boundary_data_list,
                                                        ignore_index=True)
        return residue_boundary_grid

    def worker_get_min_max_residue(self,
                                   pqr_df: pd.DataFrame,
                                   ) -> pd.DataFrame:
        """
        Get the min and max of each residue in a file
        """
        return self.get_min_max_residue_df(pqr_df)

    def calc_grid_spacing(self) -> tuple[float, float, float]:
        """
        Calculate the grid spacing
        """
        grid_number = self.config.grid_config.grid_number
        box_size = self.config.grid_config.box_size
        grid_spacing = (box_size[0] / (grid_number[0] - 1),
                        box_size[1] / (grid_number[1] - 1),
                        box_size[2] / (grid_number[2] - 1))
        self.info_msg += f"\tGrid spacing: {grid_spacing}\n"
        return grid_spacing

    def get_min_max_residue_df(self,
                               data: pd.DataFrame,
                               ) -> pd.DataFrame:
        """
        Get the min and max of each residue in a file
        """
        residue_boundary_df: pd.DataFrame = pd.DataFrame(
            columns=[member.value for member in ResidueName])
        for residue_name in ResidueName:
            residue_df = data[
                data[ColumnName.RESIDUE_NAME.value] == residue_name.value]
            # finding the resiude number of the residue with min x
            min_x_residue = residue_df[ColumnName.Z.value].min()
            # finding the resiude number of the residue with max x
            max_x_residue = residue_df[ColumnName.Z.value].max()
            if not pd.isna(min_x_residue) and not pd.isna(max_x_residue):
                grid_min, grid_max = \
                    self._get_grid_indices(min_x_residue, max_x_residue)
                residue_boundary_df[residue_name.value] = \
                    [(grid_min, grid_max)]
        return residue_boundary_df

    def _get_grid_indices(self,
                          min_x_residue: float,
                          max_x_residue: float,
                          ) -> typing.Union[tuple[int, int],
                                            tuple[None, None]]:
        """
        Get the grid indices
        """
        grid_min: int = int(min_x_residue / self.grid_spacing[0])
        grid_max: int = int(max_x_residue / self.grid_spacing[0])
        return grid_min, grid_max

    def validate_and_process_files(self,
                                   files: list[str],
                                   log: logger.logging.Logger
                                   ) -> dict[str, pd.DataFrame]:
        """
        Check if files are valid
        """
        if not files:
            log.error(msg := "\tNo structure files are provided!\n")
            sys.exit(f"{bcolors.FAIL}{msg}{bcolors.ENDC}")
        return ReadInputStructureFile(log, files, self.config).structure_dict

    def calculate_columnwise_min_max(self,
                                     residue_boundary_grid: pd.DataFrame
                                     ) -> pd.DataFrame:
        """
        Calculate the column-wise min and max from tuples, handling
        empty iterables.
        """
        # Assuming a sensible default value can be defined.
        default_min_value = float('inf')
        default_max_value = float('-inf')

        column_min = residue_boundary_grid.apply(
            lambda col: min((tup[0] for tup in col if isinstance(tup, tuple)),
                            default=default_min_value),
            axis=0
        )

        column_max = residue_boundary_grid.apply(
            lambda col: max((tup[1] for tup in col if isinstance(tup, tuple)),
                            default=default_max_value),
            axis=0
        )

        min_max_residue = pd.concat([column_min, column_max], axis=1)
        self.info_msg += f"\tColumn-wise min and max: {min_max_residue}\n"
        return min_max_residue

    def write_xvg_file(self,
                       residue_boundary_grid: pd.DataFrame,
                       global_min_max: pd.DataFrame,
                       log: logger.logging.Logger
                       ) -> None:
        """
        Write the data into xvg format
        """
        extra_comments = "Residue boundary grid" + \
            self.df_to_string(global_min_max)
        file_writer.write_xvg(df_i=residue_boundary_grid,
                              log=log,
                              fname='residue_boundary_grid.xvg',
                              extra_comments=extra_comments,
                              xaxis_label='Residue',
                              yaxis_label='Grid index',
                              title='Residue boundary grid')

    @staticmethod
    def df_to_string(df_i
                     ) -> str:
        """Convert the DataFrame to a string"""
        lines = ["Column-wise min and max:"]
        for idx, row in df_i.iterrows():
            line = f"#{idx}  {row[0]}  {row[1]}"
            lines.append(line)
        return "\n".join(lines)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ReadInputStructureFile.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class ReadInputStructureFile:
    """reading all the input structure files and return them in dict
    with the name of the file as key"""
    # pylint: disable=too-few-public-methods

    info_msg: str = '\nMessage from ReadInputStructureFile:\n'
    file_type: str  # Type of extension of the files
    _configs: AllConfig
    structure_dict: dict[str, pd.DataFrame]

    def __init__(self,
                 log: logger.logging.Logger,
                 strucure_files: list[str],
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self._configs = configs
        strucure_files = [
            file for file in strucure_files
            if my_tools.check_int_in_filename(os.path.basename(file))]
        if not strucure_files:
            log.error(msg := '\tNo structure files are provided!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        self.structure_dict = self._proccess_files(strucure_files, log)
        self._write_msg(log)

    def _proccess_files(self,
                        strucure_files: list[str],
                        log) -> dict[str, pd.DataFrame]:
        """read and return data about each structure file"""
        strucure_files = list((dict.fromkeys(strucure_files, '')).keys())
        self.file_type = self._check_file_extension(strucure_files, log)
        return self._read_files(strucure_files, log)

    def _read_files(self,
                    strucure_files: list[str],
                    log: logger.logging.Logger
                    ) -> dict[str, pd.DataFrame]:
        """read the files"""
        structure_dict: dict[str, pd.DataFrame] = {}
        if self.file_type == 'gro':
            for struct_i in strucure_files:
                fname_i: str = os.path.splitext(os.path.basename(struct_i))[0]
                structure_dict[fname_i] = \
                    gro_to_df.ReadGro(struct_i, log).gro_data
        else:
            for struct_i in strucure_files:
                fname_i = os.path.splitext(os.path.basename(struct_i))[0]
                structure_dict[fname_i] = \
                    pqr_to_df.read_pqr_to_dataframe(struct_i)
        return structure_dict

    def _check_file_extension(self,
                              strucure_files: list[str],
                              log: logger.logging.Logger
                              ) -> str:
        """check the files' extension, they all should be same gro or
        pdb"""
        file_extension: list[str] = \
            [os.path.splitext(item)[1][1:] for item in strucure_files]
        if (l_list := len(set_ext := set(file_extension))) > 1:
            log.error(
                msg := (f'\tThere are `{l_list}` file types: '
                        f'`{set_ext}`! There should be one of: '
                        f'{self._configs.accebtable_file_type}\n'))
            sys.exit(f'\n\t{bcolors.FAIL}{msg}{bcolors.ENDC}\n')
        if (exten_type := list(set(file_extension))[0]) not in \
           self._configs.accebtable_file_type:
            log.error(
                msg := (f'\tThe file type: `{exten_type}` is not exceptable!'
                        '\tshould be one of the\n'
                        f'{self._configs.accebtable_file_type}\n'))
            sys.exit(f'\n\t{bcolors.FAIL}{msg}{bcolors.ENDC}\n')

        self.info_msg += \
            f'\tReading `{len(strucure_files)}` files, type: `{exten_type}`\n'
        return exten_type

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ReadInputStructureFile.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    AnalysisStructure(sys.argv[1:],
                      logger.setup_logger("apbs_analysis_pqr.log"))
