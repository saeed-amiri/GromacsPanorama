"""
Radial averaging of the DLVO potential calculation
The input calculation is done with ABPS simulations and the output is in the
format of .dx files:

    object 1 class gridpositions counts nx ny nz
    origin xmin ymin zmin
    delta hx 0.0 0.0
    delta 0.0 hy 0.0
    delta 0.0 0.0 hz
    object 2 class gridconnections counts nx ny nz
    object 3 class array type double rank 0 items n data follows
    u(0,0,0) u(0,0,1) u(0,0,2)
    ...
    u(0,0,nz-3) u(0,0,nz-2) u(0,0,nz-1)
    u(0,1,0) u(0,1,1) u(0,1,2)
    ...
    u(0,1,nz-3) u(0,1,nz-2) u(0,1,nz-1)
    ...
    u(0,ny-1,nz-3) u(0,ny-1,nz-2) u(0,ny-1,nz-1)
    u(1,0,0) u(1,0,1) u(1,0,2)
    ...
    attribute "dep" string "positions"
    object "regular positions regular connections" class field
    component "positions" value 1
    component "connections" value 2
    component "data" value 3`

The variables in this format include:
    nx ny nz
        The number of grid points in the x-, y-, and z-directions
    xmin ymin zmin
        The coordinates of the grid lower corner
    hx hy hz
        The grid spacings in the x-, y-, and z-directions.
    n
        The total number of grid points; n=nx*ny*nz
    u(*,*,*)
        The data values, ordered with the z-index increasing most
        quickly, followed by the y-index, and then the x-index.


First the header should be read, than based on number of the grids,
and size of the grids, the data should be read and the radial averaging
should be done.
input files are in the format of .dx
13 Jun 2024
Saeed
"""

import os
import sys
import typing
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import logger, file_writer
from common.colors_text import TextColor as bcolors


@dataclass
class InputConfig:
    """set the name of the input files"""
    header_file: str = 'apt_cor_0.dx'
    number_of_header_lines: int = 11
    number_of_tail_lines: int = 5
    output_file: str = 'radial_average_potential.xvg'


class RadialAveragePotential:
    """
    Compute the radial average potential from the output files of ABPS
    simulations
    """

    info_msg: str = 'Message from RadialAveragePotential:\n'

    def __init__(self,
                 file_name: str,
                 log: logger.logging.Logger,
                 configs: InputConfig = InputConfig()
                 ) -> None:
        """write and log messages"""
        self.configs = configs
        self.process_file(file_name, log)
        self.write_msg(log)

    def process_file(self,
                     file_name: str,
                     log: logger.logging.Logger
                     ) -> None:
        """process the file"""
        lines: list[str] = self.read_file(file_name)

        grid_points: list[int]
        grid_spacing: list[float]
        origin: list[float]
        grid_points, grid_spacing, origin = self._get_header(
            lines[:self.configs.number_of_header_lines], log)
        tail: list[str] = self._get_tail(
            lines[-self.configs.number_of_tail_lines:])

    def _get_header(self,
                    head_lines: list[str],
                    log: logger.logging.Logger
                    ) -> tuple[list[int], list[float], list[float]]:
        """get the header"""
        grid_points: list[int]
        grid_spacing: list[float] = []
        origin: list[float]
        self.check_header(head_lines, log)
        for line in head_lines:
            if 'counts' in line:
                grid_points = [int(i) for i in line.split()[-3:]]
            if 'origin' in line:
                origin = [float(i) for i in line.split()[-3:]]
            if 'delta' in line:
                grid_spacing.append([
                    float(i) for i in line.split()[-3:] if float(i) != 0.0][0])
        return grid_points, grid_spacing, origin

    def _get_tail(self,
                  tail_lines: list[str]
                  ) -> list[str]:
        """get the tail"""
        return tail_lines

    @staticmethod
    def check_header(head_lines: list[str],
                     log: logger.logging.Logger
                     ) -> None:
        """check the header"""
        try:
            assert 'counts' in head_lines[4]
            assert 'origin' in head_lines[5]
            assert 'delta' in head_lines[6]
        except AssertionError:
            msg: str = 'The header is not in the correct format!'
            print(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
            log.error(msg)
            sys.exit(1)

    def read_file(self,
                  file_name: str
                  ) -> None:
        """read the file"""
        with open(file_name, 'r', encoding='utf-8') as f_dx:
            lines = f_dx.readlines()
        lines = [line.strip() for line in lines]
        return lines

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{RadialAveragePotential.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    try:
        RadialAveragePotential(
            sys.argv[1],
            log=logger.setup_logger('radial_average_potential.log'))
    except IndexError:
        print(f'{bcolors.FAIL}No file is provided!{bcolors.ENDC}')
        sys.exit(1)
