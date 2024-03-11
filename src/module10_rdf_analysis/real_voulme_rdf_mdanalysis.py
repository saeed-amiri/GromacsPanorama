"""
To correct the RDF computation, one must use the system with NP moved
to the center of the box.
The RDF in this system is not uniform in any condition. When
calculating the RDF, the probability of the existence of particles in
some portion of the system is zero. The standard RDF calculation
method may need to consider the system's heterogeneity, mainly when
the reference point (such as the center of mass of the nanoparticle)
is located at the interface between two phases with different
densities and compositions.
The RDF is typically used to determine the probability of finding a
particle at a distance r from another particle, compared to the
likelihood expected for a completely random distribution at the same
density. However, the standard RDF calculation method can produce
misleading results in a system where the density and composition vary
greatly, such as when half of the box is empty or when there are
distinct water and oil phases. The method assumes that particles are
uniformly and isotropically distributed throughout the volume.
Also, for water and all water-soluble residues, the volume computation
depends very much on the radius of the volume, and we compute the RDF
based on this radius.
This computation should be done by considering whether the radius is
completely inside water, half in water, or even partially contains
water and oil.
For oil, since the NP is put in the center of the box, some water will
be below and some above. This condition must be fixed by bringing all
the oil residues from the bottom of the box to the top of the box.


For this computation to be done, the main steps are:
    A. Count the number of residues in the nominal volume
    B. Compute the volume of the system
    C. Compute the RDF
    For every frame:
      A:
        1. Read the coordinated of the residues with MDAnalysis
        2. Calculate the distances between the residues and the NP
        3. Count the number of residues in the nominal volume
      B:
        4. Get the COM of the NP (coord.xvg)
        5. Get the box size (box.xvg)
        6. Get the intrface location (z) (contact.xvg)
        7. compute the real volume of the system
     C:
        8. Compute the RDF
        9. Save the RDF in a file
"""

import sys
import typing
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.analysis import rdf

from common import logger, xvg_to_dataframe, my_tools
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
        'sel_type': 'resname',
        'sel_names': ['D10'],
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
    bin_size: float = 0.01
    dist_range: tuple[float, float] = field(init=False)
    density: bool = True


@dataclass
class FileConfig:
    """Configuration for the RDF analysis"""
    # Input files
    interface_info: str = "contact.xvg"
    box_size_fname: str = "box.xvg"
    np_com_fname: str = "coord.xvg"
    top_fname: str = 'topol.top'
    trr_fname: str = field(init=False)


@dataclass
class DataConfig:
    """Configuration for the RDF analysis"""
    # Input files
    interface: np.ndarray = field(init=False)
    box_size: np.ndarray = field(init=False)
    np_com: np.ndarray = field(init=False)
    top: str = field(init=False)
    u_traj: "mda.Universe" = field(init=False)


@dataclass
class AllConfig(GroupConfig,
                ParamConfig,
                FileConfig,
                DataConfig
                ):
    """All the configurations for the RDF analysis
    """


class RealValumeRdf:
    """compute RDF for the system based on the configuration"""

    info_msg: str = 'Message from RealValumeRdf:\n'
    config: AllConfig

    def __init__(self,
                 trr_fname: str,
                 log: logger.logging.Logger,
                 config: AllConfig = AllConfig()
                 ) -> None:
        self.config = config
        self.config.trr_fname = trr_fname
        self.write_msg(log)
        self.initiate(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> None:
        """initiate the RDF computation"""
        self.check_file_existence(log)
        self.parse_and_store_data(log)
        self._read_trajectory(log)
        ref_group: "mda.core.groups.AtomGroup" = self._get_ref_group()
        target_group: "mda.core.groups.AtomGroup" = self._get_target_group(log)
        dist_range: np.ndarray = self._get_radius_bins()
        self.compute_rdf(ref_group, target_group, dist_range, log)

    def compute_rdf(self,
                    ref_group: "mda.core.groups.AtomGroup",
                    target_group: "mda.core.groups.AtomGroup",
                    dist_range: np.ndarray,
                    log: logger.logging.Logger
                    ) -> None:
        """compute the RDF"""
        # Compute the number of in each bin; return it with the bin edges,
        # np_com, and the number of atoms in the target group
        # The new interface will be the difference between the interface
        # and the np_com from the z axis; based on this we should compute
        # real volume of the system
        rdf_counts: np.ndarray = \
            self._count_numbers_in_bins(ref_group, target_group, dist_range)
        print(np.mean(rdf_counts))

    def _count_numbers_in_bins(self,
                               ref_group: "mda.core.groups.AtomGroup",
                               target_group: "mda.core.groups.AtomGroup",
                               dist_range: np.ndarray
                               ) -> np.ndarray:
        """count the number of atoms in each bin"""
        rdf_counts = np.zeros(dist_range.shape[0] - 1, dtype=int)
        for _ in self.config.u_traj.trajectory:
            np_com: np.ndarray = ref_group.center_of_mass()
            distances_to_com = \
                np.linalg.norm(target_group.positions - np_com, axis=1)
            for i in range(len(dist_range) - 1):
                indices = \
                    np.where((distances_to_com > dist_range[i]) &
                             (distances_to_com <= dist_range[i + 1]))[0]
                rdf_counts[i] += len(indices)
        rdf_counts = rdf_counts / self.config.u_traj.trajectory.n_frames
        return rdf_counts

    def check_file_existence(self,
                             log: logger.logging.Logger
                             ) -> None:
        """check the existence of the files"""
        for fname in [self.config.interface_info,
                      self.config.box_size_fname,
                      self.config.np_com_fname,
                      self.config.trr_fname,
                      self.config.top_fname]:
            my_tools.check_file_exist(fname, log, if_exit=True)

    def parse_and_store_data(self,
                             log: logger.logging.Logger
                             ) -> None:
        """get the data from the files"""
        interface = xvg_to_dataframe.XvgParser(
            self.config.interface_info, log).xvg_df
        self.config.interface = \
            self._df_to_numpy(interface, ['interface_z'])

        box_size = xvg_to_dataframe.XvgParser(
            self.config.box_size_fname, log).xvg_df
        self.config.box_size = \
            self._df_to_numpy(box_size, ['XX', 'YY', 'ZZ'])

        np_com = xvg_to_dataframe.XvgParser(
            self.config.np_com_fname, log).xvg_df
        self.config.np_com = self._df_to_numpy(
            np_com, ['COR_APT_X', 'COR_APT_Y', 'COR_APT_Z'])

    def _read_trajectory(self,
                         log: logger.logging.Logger
                         ) -> None:
        """read the input file"""
        fname: str = self.config.trr_fname
        my_tools.check_file_exist(fname, log, if_exit=True)
        tpr_file: str = fname.split('.', -1)[0] + '.tpr'
        my_tools.check_file_exist(tpr_file, log, if_exit=True)
        try:
            self.config.u_traj = mda.Universe(tpr_file, fname)
        except ValueError as err:
            log.error(msg := '\tThe input file is not correct!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}\n\t{err}\n')

    def _df_to_numpy(self,
                     df_i: pd.DataFrame,
                     columns: list[str]
                     ) -> np.ndarray:
        """convert the dataframe to numpy array"""
        return df_i[columns].to_numpy()

    def _get_ref_group(self) -> "mda.core.groups.AtomGroup":
        """get the reference group"""
        ref_group: str = f'{self.config.ref_group["sel_type"]}' + " "
        ref_group += ' '.join(self.config.ref_group["sel_names"])
        selected_group = self.config.u_traj.select_atoms(ref_group)
        nr_sel_group = selected_group.n_atoms
        self.info_msg += \
            f'\tReference group: `{ref_group}` has `{nr_sel_group}` atoms \n'
        return selected_group

    def _get_target_group(self,
                          log: logger.logging.Logger
                          ) -> "mda.core.groups.AtomGroup":
        """get the reference group"""
        target_group: str = f'{self.config.target_group["sel_type"]}' + " "
        target_group += ' '.join(self.config.target_group["sel_names"])
        selected_group = self.config.u_traj.select_atoms(target_group)
        nr_sel_group = selected_group.n_atoms
        self.info_msg += \
            f'\tTarget group: `{target_group}` has `{nr_sel_group}` atoms \n'
        if nr_sel_group == 0:
            msg = (f'\tThe target group has 0 atoms!\n'
                   f'\tThe target group was set to {target_group}!\n')
            log.error(msg)
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
        return selected_group

    def _get_radius_bins(self) -> np.ndarray:
        """get the radius bins for the RDF computation
        Angstrom to nm: 1 Angstrom = 0.1 nm
        """
        max_length: float = np.max(self.config.box_size) * 10.0
        self.info_msg = f'\tmax_length: `{max_length} [A]`\n'
        number_of_bins: int = int(max_length / self.config.bin_size)
        dist_range = np.linspace(0.0, max_length, number_of_bins)
        return dist_range

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    RealValumeRdf(sys.argv[1], logger.setup_logger('real_volume_rdf.log'))
