"""
Computing Rdf by MDAnalysis module
"""

import sys
import typing
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
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
        'sel_pos': 'com'
    }))

    target_group: dict[str, typing.Any] = field(default_factory=lambda: ({
        'sel_type': 'name',
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
    n_bins: int = 1080  # Default value in MDA
    dist_range: tuple[float, float] = field(init=False)
    density: bool = True


@dataclass
class OutFileConfig:
    """set the parameters for the output file
    The final file will be written in xvg format, similar to GROMACS
    RDF output.
    The columns' names of the target group, which will be set
    inside the script
    """
    fout_prefix: str = 'rdf_mda'
    columns: list[str] = field(default_factory=lambda: (['distance']))


@dataclass
class AllConfig(GroupConfig, ParamConfig, OutFileConfig):
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
        self.configs.dist_range = self._set_rdf_range()
        ref_group: "mda.core.groups.AtomGroup" = self._get_ref_group()
        target_group: "mda.core.groups.AtomGroup" = self._get_target_group()
        if self.configs.ref_group['sel_pos'] == 'position':
            rdf_arr: np.ndarray = \
                self._compute_rdf_from_all(ref_group, target_group)
        elif self.configs.ref_group['sel_pos'] == 'com':
            rdf_arr = self._compute_rdf_from_com(ref_group, target_group)

        rdf_df: pd.DataFrame = self._arr_to_df(rdf_arr)
        self._write_xvg(rdf_df, log)

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

    def _compute_rdf_from_com(self,
                              ref_group: "mda.core.groups.AtomGroup",
                              target_group: "mda.core.groups.AtomGroup"
                              ) -> np.ndarray:
        """
        Calculate RDF from the center of mass of ref_group to atoms in
        target_group.

        Parameters:
        - u: MDAnalysis Universe object.
        - ref_group: MDAnalysis AtomGroup for the reference group.
        - target_group: MDAnalysis AtomGroup for the target group.
        - nbins: Number of bins for the RDF histogram.
        - range: The minimum and maximum distances for the RDF calculation.
        """
        self.info_msg += '\tComputing RDF from COM of the reference group\n'

        distances = np.linspace(self.configs.dist_range[0],
                                self.configs.dist_range[1],
                                self.configs.n_bins+1)
        rdf_histogram = np.zeros(self.configs.n_bins)

        for tstep in self.u_traj.trajectory:
            box: np.ndarray = tstep.dimensions
            com = ref_group.center_of_mass()
            distances_to_com = \
                np.linalg.norm(target_group.positions - com, axis=1)

            # Note: This simple example does not consider periodic boundary
            # conditions for distances.
            hist, _ = np.histogram(distances_to_com, bins=distances)
            rdf_histogram += hist

        # Normalize the RDF
        # This part requires careful consideration of volume elements
        # and density to correctly normalize the RDF.
        # The normalization process can vary based on your system and
        # specific requirements.
        # As an example, a simple normalization by the number of
        # frames and target atoms might look like this:
        rdf_i = \
            rdf_histogram / (len(self.u_traj.trajectory) * len(target_group))
        # Calculate bin centers from distances for plotting
        bin_centers = (distances[:-1] + distances[1:]) / 2

        rdf_arr: np.ndarray = np.zeros((self.configs.n_bins, 2))
        rdf_arr[:, 0] = bin_centers
        rdf_arr[:, 1] = rdf_i
        self.info_msg += \
            "\tComputed RDF from all the COM of ref successfully\n"

        if self.configs.show_plot:
            plt.plot(bin_centers, rdf_i, '-0')
            plt.show()
        return rdf_arr

    def _compute_rdf_from_all(self,
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
        self.info_msg += "\tComputed RDF from all the ref atoms successfully\n"
        if self.configs.show_plot:
            plt.plot(rdf_distances, rdf_values, '-0')
            plt.show()
        return rdf_arr

    def _read_trajectory(self,
                         fname: str,
                         log: logger.logging.Logger
                         ) -> None:
        """read the input file"""
        my_tools.check_file_exist(fname, log, if_exit=True)
        tpr_file: str = fname.split('.', -1)[0] + '.tpr'
        my_tools.check_file_exist(tpr_file, log, if_exit=True)
        try:
            self.u_traj = mda.Universe(tpr_file, fname)
        except ValueError as err:
            log.error(msg := '\tThe input file is not correct!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}\n\t{err}\n')

    def _arr_to_df(self,
                   rdf_arr: np.ndarray
                   ) -> pd.DataFrame:
        """convert the arr to dataframe to write to xvg file"""
        self.configs.columns.extend(
            self.configs.target_group["sel_names"])
        rdf_df: pd.DataFrame = pd.DataFrame(columns=self.configs.columns)
        rdf_df['distance'] = rdf_arr[:, 0] / 10
        rdf_df[self.configs.columns[1]] = rdf_arr[:, 1]
        rdf_df.set_index('distance', inplace=True)
        self.info_msg += f'\tThe distance is converted to [nm]\n'
        return rdf_df

    def _write_xvg(self,
                   rdf_df: pd.DataFrame,
                   log: logger.logging.Logger
                   ) -> None:
        """write the rdf to a xvg file"""
        fout: str = \
            f'{self.configs.fout_prefix}_{self.configs.ref_group["sel_pos"]}'
        fout += f'_{self.configs.target_group["sel_names"][0]}'
        fout += ".xvg"
        extra_msg: list[str] = [
            '# This RDF is calculate by using MDAnalysis',
            f'# Ran by {self.__module__}']
        my_tools.write_xvg(rdf_df,
                           log,
                           extra_msg,
                           fout,
                           write_index=True,
                           x_axis_label='r (nm)',
                           y_axis_label='g (r)',
                           title='Radial distribution')
        self.info_msg += f'\tThe rdf is saved as `{fout}`\n'

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == '__main__':
    LOG: logger.logging.Logger = logger.setup_logger('rdf_by_mda.log')
    try:
        FNAME: str = sys.argv[1]
    except IndexError as err_0:
        LOG.warning(MSG := 'The input was empty! The traj set to `npt.trr`\n')
        print(f'{bcolors.CAUTION}{MSG}{bcolors.ENDC}')
        FNAME = 'npt.trr'
    RdfByMDAnalysis(FNAME, LOG)
