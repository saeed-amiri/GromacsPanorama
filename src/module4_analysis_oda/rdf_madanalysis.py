"""
Computing Rdf by MDAnalysis module
"""

import sys
import typing
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pylab as plt

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

    target_group: dict[str, typing.Any] = field(default_factory=lambda: ({
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
    dist_range: tuple[float, float] = field(init=False)
    density: bool = True


@dataclass
class AllConfig(GroupConfig, ParamConfig):
    """set all the parameters for the computations"""
    show_plot: bool = True


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
        self.get_rdf(fname, log)
        self.write_log_msg(log)

    def get_rdf(self,
                fname: str,
                log: logger.logging.Logger
                ) -> None:
        """set the parameters and get the rdf"""
        self._read_trajectory(fname, log)
        self.configs.dist_range: tuple[float, float] = self._set_rdf_range()
        ref_group: "mda.core.groups.AtomGroup" = self._get_ref_group()
        target_group: "mda.core.groups.AtomGroup" = self._get_target_group()
        self._compute_rdf(ref_group, target_group)

    def _set_rdf_range(self) -> tuple[float, float]:
        """find thelimitation of the box to set the range of the
        calculations
        set the range based on the maximum size of the box
        """
        frame_index: int = 0
        box_dimensions = self.u_traj.trajectory[frame_index].dimensions
        dist_range: tuple[float, float] = (0.0, max(box_dimensions[0:3]) / 2)
        self.info_msg += (f'\tBox dims at frame `{frame_index}` is:\n'
                          f'\t\t{box_dimensions[0:3]}\n'
                          f'\t\tdist range is set to `{dist_range[1]:.3f}`\n')
        dist_range = (0, 100)
        return dist_range

    def _get_ref_group(self) -> "mda.core.groups.AtomGroup":
        """get the reference group"""
        ref_group: str = f'{self.configs.ref_group["sel_type"]}' + " "
        ref_group += ' '.join(self.configs.ref_group["sel_names"])
        selected_group = self.u_traj.select_atoms(ref_group)
        nr_sel_group = selected_group.n_atoms
        self.info_msg += \
            f'\tReference group: `{ref_group}` has `{nr_sel_group}` atoms \n'
        return selected_group

    def _get_target_group(self) -> "mda.core.groups.AtomGroup":
        """get the reference group"""
        target_group: str = f'{self.configs.target_group["sel_type"]}' + " "
        target_group += ' '.join(self.configs.target_group["sel_names"])
        selected_group = self.u_traj.select_atoms(target_group)
        nr_sel_group = selected_group.n_atoms
        self.info_msg += \
            f'\tTarget group: `{target_group}` has `{nr_sel_group}` atoms \n'
        return selected_group

    def _compute_rdf(self,
                     ref_group: "mda.core.groups.AtomGroup",
                     target_group: "mda.core.groups.AtomGroup"
                     ) -> np.ndarray:
        """compute rdf for the selected groups"""
        # Initialize the InterRDF object with your groups
        rdf_analyzer = rdf.InterRDF(ref_group, target_group,
                                    nbins=self.configs.n_bins,
                                    range=self.configs.dist_range)

        # Run the analysis
        rdf_analyzer.run()

        # Access the results
        rdf_values = rdf_analyzer.results.rdf
        rdf_distances = rdf_analyzer.results.bins
        rdf_arr: np.ndarray = np.zeros((len(rdf_values), 2))
        rdf_arr[:, 0] = rdf_distances
        rdf_arr[:, 1] = rdf_values

        # Optionally, log or print the results for verification
        self.info_msg += "\tComputed RDF successfully.\n"
        if self.configs.show_plot:
            plt.plot(rdf_distances, rdf_values, '-0')
            plt.show()
        return rdf_arr

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
