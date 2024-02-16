"""
Computing Rdf by MDAnalysis module
"""

import sys
import typing
from dataclasses import dataclass, field

import MDAnalysis as mda
from MDAnalysis.analysis import rdf

from common import logger, my_tools
from common.colors_text import TextColor as bcolors


@dataclass
class GroupConfig:
    """set the configurations for the rdf
    userguide.mdanalysis.org/1.1.1/selections.html?highlight=select_atoms

    sel_type -> str: type of the selection, is it residue or atom
    sel_names -> list[str]: names of the selection groups
    sel_pos -> str: If use the ceter of mass (COM) of the group or their
        poistions
    """
    ref_group: dict[str, typing.Any] = field(default_factory=lambda: ({
        'sel_type': 'resname',
        'sel_names': ['COR'],
        'sel_pos': 'COM'
    }))

    traget_group: dict[str, typing.Any] = field(default_factory=lambda: ({
        'sel_type': 'resname',
        'sel_names': ['CLA'],
        'sel_pos': 'position'
    }))


@dataclass
class ParamConfig:
    """set the parameters for the rdf computations with MDA
    MDA:
        "The RDF is limited to a spherical shell around each atom by
        range. Note that the range is defined around each atom, rather
        than the center-of-mass of the entire group.
        If density=True, the final RDF is over the average density of
        the selected atoms in the trajectory box, making it comparable
        to the output of rdf.InterRDF. If density=False, the density
        is not taken into account. This can make it difficult to
        compare RDFs between AtomGroups that contain different numbers
        of atoms."
    """
    n_bins: int = 75  # Default value in MDA
    dist_range: tuple[float, float] = field(default_factory=lambda: ((
        0, 10
    )))
    density: bool = True


@dataclass
class AllConfig(GroupConfig, ParamConfig):
    """set all the parameters for the computations"""


class RdfByMDAnalysis:
    """compute the rdf"""

    info_msg: str = 'Message from RdfByMDAnalysis:\n'
    configs: AllConfig
    u_traj: mda.Universe  # Trajectory read by MDAnalysis

    def __init__(self,
                 fname: str,  # Trr or Xtc file,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.write_log_msg(log)
        self.get_rdf(fname, log)

    def get_rdf(self,
                fname: str,
                log: logger.logging.Logger
                ) -> None:
        """set the parameters and get the rdf"""
        self._read_trajectory(fname, log)
        self._get_ref_group()

    def _get_ref_group(self) -> mda.core.groups.AtomGroup:
        """get the reference group"""
        ref_group: str = f'{self.configs.ref_group["sel_type"]}' + " "
        ref_group += ' '.join(self.configs.ref_group["sel_names"])
        selected_group = self.u_traj.select_atoms(ref_group)
        nr_sel_group = selected_group.n_atoms
        self.info_msg += \
            f'\tSelected group: `{ref_group}` has `{nr_sel_group}` atoms \n'
        return selected_group

    def _read_trajectory(self,
                         fname: str,
                         log: logger.logging.Logger
                         ) -> None:
        """read the input file"""
        tpr_file: str = fname.split('.', -1)[0] + '.tpr'
        my_tools.check_file_exist(tpr_file, log, if_exit=True)
        try:
            self.u_traj = mda.Universe(tpr_file, fname)
        except ValueError as err:
            log.error(msg := '\tThe input file is not correct!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}\n\t{err}\n')

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == '__main__':
    RdfByMDAnalysis(sys.argv[1], logger.setup_logger('rdf_by_mda.log'))
