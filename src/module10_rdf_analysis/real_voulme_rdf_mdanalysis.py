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

import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MDAnalysis as mda

from common import logger, xvg_to_dataframe, my_tools, cpuconfig
from common.colors_text import TextColor as bcolors


@dataclass
class GroupConfig:
    """set the configurations for the rdf
    userguide.mdanalysis.org/1.1.1/selections.html?highlight=select_atoms

    sel_type -> str: type of the selection, is it residue or atom
    sel_names -> list[str]: names of the selection groups
    sel_pos -> str: If use the ceter of mass (COM) of the group or their
        poistions
    Since here the number of the residues is more intersted, for each
    residue, one atom is seleced to compute the RDF:
    Water: OH2
    Oil: C5
    ODN: NH2
    APT: N
    CLA: CLA
    POT: POT
    """
    oda_group: list[str] = field(default_factory=lambda: (['ODN', 'NH2']))
    oil_group: list[str] = field(default_factory=lambda: (['D10', 'C5', 'c9']))

    ref_group: dict[str, typing.Any] = field(default_factory=lambda: ({
        'sel_type': 'name',
        'sel_names': ['NH2'],
        'sel_pos': 'position'
    }))

    target_group: dict[str, typing.Any] = field(default_factory=lambda: ({
        'sel_type': 'name',
        'sel_names': ['NH2'],
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
    bin_size: float = 0.05
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
    d_time: float = field(init=False)  # time step of the trajectory


@dataclass
class AllConfig(GroupConfig,
                ParamConfig,
                FileConfig,
                DataConfig
                ):
    """All the configurations for the RDF analysis
    """
    # pylint: disable=too-many-instance-attributes
    num_cores: int = field(init=False)   # number of cores to use
    n_frames: int = field(init=False)   # number of frames in the trajectory


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
        self.initiate(log)
        self.write_msg(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> None:
        """initiate the RDF computation"""
        self.check_file_existence(log)
        self.parse_and_store_data(log)
        self.load_trajectory(log)
        self.config.n_frames = self.config.u_traj.trajectory.n_frames
        self.config.d_time = self.config.u_traj.trajectory.dt
        self.info_msg += (f'\tNumber of frames: `{self.config.n_frames}`\n'
                          f'\tTime step: `{self.config.d_time}` [ps]\n')
        self.config.num_cores = self.set_number_of_cores(log)
        ref_group: "mda.core.groups.AtomGroup" = self.get_ref_group(log)
        target_group: "mda.core.groups.AtomGroup"
        nr_sel_group: int
        target_group, nr_sel_group = self.get_target_group(log)
        dist_range: np.ndarray = self.get_radius_bins()
        self.compute_rdf(
            ref_group, target_group, dist_range, nr_sel_group, log)

    def compute_rdf(self,
                    ref_group: "mda.core.groups.AtomGroup",
                    target_group: "mda.core.groups.AtomGroup",
                    dist_range: np.ndarray,
                    nr_sel_group: int,
                    log: logger.logging.Logger,
                    ) -> None:
        """compute the RDF"""
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        rdf_counts: np.ndarray  # number of atoms in each bin
        np_com_arr: np.ndarray  # center of mass of the NP
        np_com_arr, target_group_pos_list = \
            self._get_np_com_traget(ref_group, target_group)

        sol_volume_dict: dict[int, np.ndarray]  # volume of the bins
        oil_volume_dict: dict[int, np.ndarray]  # volume of the bins
        interface_below: np.ndarray  # interface below the NP
        interface_main: np.ndarray  # interface of the NP
        sol_volume_dict, oil_volume_dict, interface_below, interface_main = \
            self._get_volume_of_system(dist_range, np_com_arr, log)

        phase: str = self.config.target_group['sel_names'][0]
        if phase not in self.config.oil_group:
            volume_dict = sol_volume_dict
            self.info_msg += '\tThe phase is water.\n'
        elif phase in self.config.oil_group:
            volume_dict = oil_volume_dict
            self.info_msg += '\tThe phase is oil.\n'

        rdf_counts_dict: dict[int, np.ndarray]  # number of atoms in each bin
        rdf_counts, rdf_counts_dict = \
            self._count_numbers_in_bins(np_com_arr,
                                        target_group_pos_list,
                                        dist_range,
                                        interface_below,
                                        interface_main)

        rdf_dict: dict[int, np.ndarray] = {}  # the RDF for each frame
        for frame, rdf_counts in rdf_counts_dict.items():
            water_volume: np.float64 = np.sum(volume_dict[frame])
            nr_in_frame: int = np.sum(rdf_counts)
            number_density: np.float64 = nr_in_frame / water_volume
            bin_volumes = volume_dict[frame]
            rdf = rdf_counts / (number_density * bin_volumes)
            rdf_dict[frame] = rdf

        # average RDF over all frames
        avg_rdf = np.mean(np.array(list(rdf_dict.values())), axis=0)

        plt.plot(dist_range[:-1], avg_rdf)
        plt.show()

        self._write_rdf_xvg(dist_range,
                            bin_volumes,
                            np.mean(water_volume),
                            rdf_counts,
                            avg_rdf,
                            nr_sel_group,
                            log)
        self.wrrite_cdf_xvg(dist_range, rdf_counts, nr_sel_group, log)

    def _get_np_com_traget(self,
                           ref_group: "mda.core.groups.AtomGroup",
                           target_group: "mda.core.groups.AtomGroup",
                           ) -> tuple[np.ndarray, list[np.ndarray]]:
        """compute the center of mass of the NP and the position of the
        target group
        """
        np_com_list: list[np.ndarray] = []
        target_group_pos_list: list[np.ndarray] = []
        with mp.Pool(processes=self.config.num_cores) as pool:
            args = [(ref_group, target_group, frame) for frame in
                    range(self.config.n_frames)]
            results = pool.starmap(self._compute_frame_np_com, args)
            np_com_list, target_group_pos_list = zip(*results)
        np_com_arr: np.ndarray = np.vstack(np_com_list)
        return np_com_arr, target_group_pos_list

    def _get_volume_of_system(self,
                              dist_range: np.ndarray,
                              np_com: np.ndarray,
                              log: logger.logging.Logger
                              ) -> tuple[dict[int, np.ndarray],
                                         dict[int, np.ndarray],
                                         np.ndarray,
                                         np.ndarray]:
        """compute the volume of the system"""
        volume_prop = ComputeRealVolume(self.config, dist_range, np_com, log)
        return volume_prop.sol_volume, volume_prop.oil_volume, \
            volume_prop.interface_below, volume_prop.interface_main

    def _compute_frame_np_com(self,
                              ref_group: "mda.core.groups.AtomGroup",
                              target_group: "mda.core.groups.AtomGroup",
                              frame: int
                              ) -> tuple[list[np.ndarray], np.ndarray]:
        """compute the center of mass of the NP and the position of the
        target group
        the statement:
            self.config.u_traj.trajectory[frame]
        should be used to get the frame, otherwies it will give the
        count in a histogram form!
        """
        # pylint: disable=pointless-statement
        self.config.u_traj.trajectory[frame]
        np_com: np.ndarray = ref_group.center_of_mass()
        np_com_list: list[np.ndarray] = [np_com]
        target_group_pos: np.ndarray = target_group.positions
        return np_com_list, target_group_pos

    def _count_numbers_in_bins(self,
                               np_com: np.ndarray,
                               target_group_list: list[np.ndarray],
                               dist_range: np.ndarray,
                               interface_below: np.ndarray,
                               interface_main: np.ndarray,
                               ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """count the number of atoms in each bin"""
        # pylint: disable=too-many-arguments
        rdf_counts = np.zeros(dist_range.shape[0] - 1, dtype=int)
        rdf_counts_dict: dict[int, np.ndarray] = {}
        with mp.Pool(processes=self.config.num_cores) as pool:
            args = [(com_i,
                     target_group_list[frame],
                     dist_range,
                     self.config.box_size[frame],
                     interface_below[frame],
                     interface_main[frame]
                     ) for frame, com_i in enumerate(np_com)]
            results = pool.starmap(self._frame_count_in_bin, args)
            for i, res in enumerate(results):
                rdf_counts += res
                rdf_counts_dict[i] = res
        rdf_counts = rdf_counts / self.config.n_frames
        return rdf_counts, rdf_counts_dict

    def _frame_count_in_bin(self,
                            com_i: np.ndarray,
                            target_group: np.ndarray,
                            dist_range: np.ndarray,
                            box_size: np.ndarray,
                            interface_below: np.ndarray,
                            interface_main: np.ndarray
                            ) -> np.ndarray:
        """count the number of atoms in a single bin"""
        # pylint: disable=too-many-arguments

        rdf_counts = np.zeros(dist_range.shape[0] - 1, dtype=int)

        # Filter the particles by the phase
        target_group = self.filter_particles_by_phase(
            target_group, interface_below, interface_main)

        # Calculate the distances with PBC
        d_x: np.ndarray
        d_y: np.ndarray
        d_z: np.ndarray
        d_x, d_y, d_z = \
            self.calculate_pbc_distances(target_group, com_i, box_size)

        # Calculate the distance
        distances_to_com = np.sqrt(d_x**2 + d_y**2 + d_z**2)

        # count the number of atoms in each bin
        for i in range(len(dist_range) - 1):
            indices = \
                np.where((distances_to_com > dist_range[i]) &
                         (distances_to_com <= dist_range[i + 1]))[0]
            rdf_counts[i] += len(indices)
        return rdf_counts

    def filter_particles_by_phase(self,
                                  target_group: np.ndarray,
                                  interface_below: np.ndarray,
                                  interface_main: np.ndarray
                                  ) -> np.ndarray:
        """filter the particles by the phase"""
        if (self.config.target_group['sel_names'][0]
           not in self.config.oil_group):
            target_group = target_group[(target_group[:, 2] > interface_below)]
            if (self.config.target_group['sel_names'][0]
               not in self.config.oda_group):
                target_group = \
                    target_group[(target_group[:, 2] < interface_main[2])]
        else:
            target_group = \
                target_group[(target_group[:, 2] > interface_main[2])]
        return target_group

    def calculate_pbc_distances(self,
                                target_group: np.ndarray,
                                com_i: np.ndarray,
                                box_size: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """calculate the distances with PBC"""
        # Calculate distances in x, y, and z separately
        d_x = target_group[:, 0] - com_i[0]
        d_y = target_group[:, 1] - com_i[1]
        d_z = target_group[:, 2] - com_i[2]

        # Apply PBC
        d_x = d_x - box_size[0] * np.round(d_x / box_size[0])
        d_y = d_y - box_size[1] * np.round(d_y / box_size[1])
        d_z = d_z - box_size[2] * np.round(d_z / box_size[2])

        return d_x, d_y, d_z

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
            self._df_to_numpy(box_size, ['XX', 'YY', 'ZZ']) * 10.0

        np_com = xvg_to_dataframe.XvgParser(
            self.config.np_com_fname, log).xvg_df
        self.config.np_com = self._df_to_numpy(
            np_com, ['COR_APT_X', 'COR_APT_Y', 'COR_APT_Z']) * 10.0

    def load_trajectory(self,
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

    def get_ref_group(self,
                      log: logger.logging.Logger
                      ) -> "mda.core.groups.AtomGroup":
        """get the reference group"""
        ref_group: str = f'{self.config.ref_group["sel_type"]}' + " "
        ref_group += ' '.join(self.config.ref_group["sel_names"])
        selected_group = self.config.u_traj.select_atoms(ref_group)
        nr_sel_group = selected_group.n_atoms
        if nr_sel_group == 0:
            msg = (f'\tThe reference group has 0 atoms!\n'
                   f'\tThe reference group was set to {ref_group}!\n')
            log.error(msg)
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
        self.info_msg += \
            f'\tReference group: `{ref_group}` has `{nr_sel_group}` atoms \n'
        return selected_group

    def get_target_group(self,
                         log: logger.logging.Logger
                         ) -> tuple["mda.core.groups.AtomGroup", int]:
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
        return selected_group, nr_sel_group

    def get_radius_bins(self) -> np.ndarray:
        """get the radius bins for the RDF computation
        """
        max_length: float = np.max(self.config.box_size) / 2.0
        self.info_msg += f'\tmax_length: `{max_length:.3f} [A]`\n'
        number_of_bins: int = int(max_length / self.config.bin_size)
        dist_range = np.linspace(0.0, max_length, number_of_bins)
        return dist_range

    def set_number_of_cores(self,
                            log: logger.logging.Logger
                            ) -> int:
        """set the number of threads for the computation"""
        cores_nr: int = cpuconfig.ConfigCpuNr(log).cores_nr
        n_cores: int = min(cores_nr, self.config.n_frames)
        self.info_msg += f'\tThe number of cores to use: {n_cores}\n'
        return n_cores

    def _write_rdf_xvg(self,
                       dist_range: np.ndarray,
                       bin_volumes: np.ndarray,
                       water_volume: np.float64,
                       rdf_counts: np.ndarray,
                       rdf: np.ndarray,
                       nr_sel_group: int,
                       log: logger.logging.Logger
                       ) -> None:
        """make the xvg dataframe for rdf and write it into a file"""
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals

        target_group: str = self.config.target_group["sel_names"][0]
        view_point: str = self.config.ref_group["sel_pos"]
        rdf_df: pd.DataFrame = pd.DataFrame({
            'r [nm]': dist_range[:-1]/10,
            'bin_volumes [nm3]': bin_volumes/1e3,
            'rdf_counts': rdf_counts,
            target_group: rdf})
        rdf_df.set_index('r [nm]', inplace=True)
        title: str = f'RDF of {target_group}'
        x_axis_label: str = 'r [nm]'
        y_axis_label: str = 'g(r)'
        fname: str = f'rdf_{target_group}_{view_point}.xvg'
        extra_msg: list[str] = \
            ['# Rdf from the center of mass of the NP',
             f'# Number of frames: {self.config.n_frames}',
             f'# Water volume: {water_volume:.3f} [A^3]',
             f'# Number of atoms in the target group: {nr_sel_group}']
        my_tools.write_xvg(df_i=rdf_df,
                           log=log,
                           extra_msg=extra_msg,
                           fname=fname,
                           x_axis_label=x_axis_label,
                           y_axis_label=y_axis_label,
                           title=title)

        self.info_msg += f'\t`{fname}` is written succsssfuly\n'

    def wrrite_cdf_xvg(self,
                       dist_range: np.ndarray,
                       rdf_counts: np.ndarray,
                       nr_sel_group: int,
                       log: logger.logging.Logger
                       ) -> None:
        """write the CDF of the RDF"""
        # pylint: disable=too-many-arguments
        target_group: str = self.config.target_group["sel_names"][0]
        view_point: str = self.config.ref_group["sel_pos"]
        cdf = np.cumsum(rdf_counts)
        cdf_df: pd.DataFrame = pd.DataFrame({
            'r [nm]': dist_range[:-1]/10,
            f'{target_group}': cdf})
        cdf_df.set_index('r [nm]', inplace=True)
        title: str = f'CDF of {target_group}'
        x_axis_label: str = 'r [nm]'
        y_axis_label: str = 'CDF'
        fname: str = f'cdf_{target_group}_{view_point}.xvg'
        extra_msg: list[str] = \
            ['# CDF of the RDF from the center of mass of the NP',
             f'# Number of frames: {self.config.n_frames}',
             f'# Number of atoms in the target group: {nr_sel_group}']
        my_tools.write_xvg(df_i=cdf_df,
                           log=log,
                           extra_msg=extra_msg,
                           fname=fname,
                           x_axis_label=x_axis_label,
                           y_axis_label=y_axis_label,
                           title=title)
        self.info_msg += f'\t`{fname}` is written succsssfuly\n'

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class ComputeRealVolume:
    """compute the real volume of the system
    To compute the real volume of the system, we need to consider the
    interface's location and the NP's center of mass.
    The interface location is computed and read from the file:
    `contact.xvg` is the location of the interface between water and
    oil when the Np could be anywhere in the box. Since we must use the
    centralized NP, the interface should be updated in a new location.
    We have the real location of the NP and its location when it's
    centralized. The new interface location is the difference between
    these values that are added to the interface location.
    Here is a very important change:
        When we centerlized the NP, the second interface below the NP
        will be important! Because in many bins, there will be another
        phase in the bin volume.
        For this, we should find the higest z value of the oil below NP
        and consider the second interface (there is no need to compute
        the second interface like the first interface).
        This distance should be tracked for every frame.

    Now that we have the interface location and the box size, we should
    look at the rdf's dist_range to see what is inside every bin.

    If we look at water and water-soluble residues, we should consider
    the following:
        Case 1: if bin's radius (r) is smaller than the distance between
            the interface and the center of mass (COM), h, of the NP,
            the volume of the whole sphere is 4\\pi r^3/3, the volume
            of shell than is 4\\pi r^2 dr
        Case 2: If r is larger than the distance h but still smaller
            than the distance between the interface and the second
            interface, h', in this case only one cap exists and should
            be dropped from the volume of the shell.
        Case 3: If r is larger than h and h', the bin's volume has to
            be removed from two places, one at the top and one at the
            bottom.

    If we look at oil:
        There is also some oil below the water phase (under the second
        interface)
        When computing the volume, we should consider the volume of
        that cap for a big radius if it passes the second interface.
        One way would be to compute that volume and add it while
        removing the value of the water from the volume.

    The needed data:
        1. The interface location
        2. The actull np_com
        3. The box size
        4. The NP's center of mass (centerlized)
        5. The rdf's dist_range
    """

    info_msg: str = 'Message from ComputeRealVolume:\n'
    dist_range: np.ndarray
    box_size: np.ndarray
    np_com: np.ndarray
    # the volume of the bins, water, and oil
    sol_volume: dict[int, np.ndarray]
    oil_volume: dict[int, np.ndarray]
    interface_below: np.ndarray
    interface_main: np.ndarray

    def __init__(self,
                 config: AllConfig,
                 dist_range: np.ndarray,
                 np_com: np.ndarray,
                 log: logger.logging.Logger
                 ) -> None:
        self.dist_range = dist_range
        self.np_com = np_com
        self.sol_volume, self.oil_volume, self.interface_below, \
            self.interface_main = self.compute_volume(config, log)
        self.write_msg(log)

    def compute_volume(self,
                       config: AllConfig,
                       log: logger.logging.Logger
                       ) -> tuple[dict[int, np.ndarray],
                                  dict[int, np.ndarray],
                                  np.ndarray,
                                  np.ndarray]:
        """compute the volume of the system"""
        actual_interface: np.ndarray = config.interface
        actual_np_com: np.ndarray = config.np_com
        interface_main: np.ndarray = \
            self.compute_interface_location(actual_interface, actual_np_com)
        interface_below: np.ndarray = self.get_interface_below(config, log)
        sol_volumes: dict[int, np.ndarray]
        oil_volumes: dict[int, np.ndarray]
        sol_volumes, oil_volumes = self.compute_pahse_volumes(
            interface_main, interface_below, config)

        return sol_volumes, oil_volumes, interface_below, interface_main

    def compute_pahse_volumes(self,
                              interface_main: np.ndarray,
                              interface_below: np.ndarray,
                              config: AllConfig
                              ) -> tuple[dict[int, np.ndarray],
                                         dict[int, np.ndarray]]:
        """compute the water and oil volumes
        using mean values of the interfaces to compute the volume
        For each bin radius we find a volume.
        """
        sol_volumes: dict[int, np.ndarray] = {}  # Water phase volume
        oil_volumes: dict[int, np.ndarray] = {}  # Oil phase volume
        bin_sizes: np.ndarray = np.diff(self.dist_range)

        for frame in range(len(interface_main)):
            h_main: np.float64 = interface_main[frame, 2]
            h_prime: np.float64 = interface_below[frame]
            np_com: np.ndarray = self.np_com[frame]
            sol_bin_volumes: np.ndarray = \
                np.zeros(self.dist_range.shape[0] - 1)
            oil_bin_volumes: np.ndarray = \
                np.zeros(self.dist_range.shape[0] - 1)
            box_size: np.ndarray = config.box_size[frame]

            for i in range(len(self.dist_range) - 1):
                radius: np.float64 = self.dist_range[i]
                d_r: np.float64 = bin_sizes[i]
                shell_volume: np.float64 = \
                    4/3 * np.pi * ((radius + d_r)**3 - radius**3)
                if radius == 0:
                    sol_volume = 1.0
                    oil_volume = 1.0
                else:
                    radius_up_point = np_com[2] + radius  # Top of the shell
                    radius_bot_point = np_com[2] - radius  # Bottom of shell

                    if (radius_up_point <= h_main and
                       radius_bot_point >= h_prime):
                        sol_volume = float(shell_volume)
                        oil_volume = 1.0

                    elif (radius_up_point > h_main and
                          radius_bot_point > h_prime):
                        h_up = radius - (h_main - np_com[2])
                        cap = \
                            self._get_cap_volume(h_up + d_r, radius + d_r) - \
                            self._get_cap_volume(h_up, radius)
                        sol_volume = float(shell_volume - cap)
                        oil_volume = float(cap)

                    else:
                        h_up = radius - (h_main - np_com[2])
                        cap_up = \
                            self._get_cap_volume(h_up + d_r, radius + d_r) - \
                            self._get_cap_volume(h_up, radius)

                        h_bottom = radius - (np_com[2] - h_prime)
                        cap_bottom = self._get_cap_volume(
                            h_bottom + d_r, radius + d_r) - \
                            self._get_cap_volume(h_bottom, radius)

                        sol_volume = float(shell_volume - cap_up - cap_bottom)
                        oil_volume = float(cap_up)

                box_caps = self._calculate_cap_of_box(
                    np_com, radius, box_size, d_r)
                sol_bin_volumes[i] = sol_volume - box_caps
                oil_bin_volumes[i] = oil_volume

            sol_volumes[frame] = sol_bin_volumes
            oil_volumes[frame] = oil_bin_volumes

        return sol_volumes, oil_volumes

    def _calculate_cap_of_box(self,
                              np_com: np.ndarray,
                              radius: np.float64,
                              box_size: np.ndarray,
                              d_r: np.float64,
                              ) -> np.float64:
        """compute the volume of the cap, if the shell is outside the box
        volume of the cap with height in a sphere with radius r is:
        (1/3) * pi * h^2 * (3r - h) - (1/3) * pi * h^2 * (3r - h)
        """
        cap_volume: np.float64 = np.float64(0.0)
        for i in range(3):
            if np_com[i] + radius > box_size[i]:
                h_right = radius - (box_size[i] - np_com[i])
                cap_right = self._get_cap_volume(
                    h_right + d_r, radius + d_r) - \
                    self._get_cap_volume(h_right, radius)
                cap_volume += cap_right
            if np_com[i] - radius < 0:
                h_left = radius - np_com[i]
                cap_left = self._get_cap_volume(
                    h_left + d_r, radius + d_r) - \
                    self._get_cap_volume(h_left, radius)
                cap_volume += cap_left
        return cap_volume

    def _get_phase_volume(self,
                          h_main_mean: np.float64,
                          h_prime_mean: np.float64,
                          config: AllConfig
                          ) -> tuple[np.float64, np.float64]:
        """compute the volume of the phase"""
        mean_dimensions = np.mean(config.box_size, axis=0)
        xy_area: np.float64 = mean_dimensions[0] * mean_dimensions[1]
        box_volume: np.float64 = \
            mean_dimensions[0] * mean_dimensions[1] * mean_dimensions[2]
        water_volume: np.float64 = xy_area * (h_main_mean - h_prime_mean)
        oil_volume: np.float64 = box_volume - water_volume
        return water_volume, oil_volume

    def _get_cap_volume(self,
                        h_cap: np.float64,
                        radius: np.float64
                        ) -> np.float64:
        """compute the volume of the cap
        volume of the cap with height in a sphere with radius r is:
        (1/3) * pi * h^2 * (3r - h)
        """
        return (1/3) * np.pi * h_cap**2 * (3*radius - h_cap)

    def compute_interface_location(self,
                                   actual_interface: np.ndarray,
                                   actual_np_com: np.ndarray,
                                   ) -> np.ndarray:
        """compute the interface location"""
        shift: np.ndarray = self.np_com - actual_np_com
        actual_np_com.tofile('actual_np_com.dat', sep='\n')
        actual_interface.tofile('actual_interface.dat', sep='\n')
        shift.tofile('shift.dat', sep='\n')
        interface_main: np.ndarray = actual_interface + shift
        self.info_msg += (
            f'\tLocation shift avg z: `{np.mean(shift[2]):.3f}`\n'
            f'\tMain interface avg z: `{np.mean(interface_main[:, 2]):.3f}`\n')
        return interface_main

    def get_interface_below(self,
                            config: AllConfig,
                            log: logger.logging.Logger
                            ) -> np.ndarray:
        """get the interface below the NP"""
        interface_below: np.ndarray = np.zeros(self.np_com.shape[0])
        for frame in config.u_traj.trajectory:
            tstep = int(frame.time / config.d_time)
            oil = config.u_traj.select_atoms('resname D10')
            oil_pos = oil.positions
            oil_z = oil_pos[:, 2]
            oil_below_np = oil_z[oil_z < self.np_com[tstep, 2]]
            if config.target_group['sel_names'][0] != 'D10':
                # Sort in descending order and take the first 100 elements
                top_oil = np.sort(oil_below_np)[::-1][:1000]
            else:
                top_oil = np.sort(oil_below_np)[::-1][:1000]
            # Calculate the average of the top 100 elements
            interface_i = np.mean(top_oil)

            # interface_i = np.max(oil_below_np)
            interface_below[tstep] = interface_i
        self.info_msg += (
            f'\tInterface_below avg z: `{np.mean(interface_below):.3f}`\n')
        self._sanity_check_2nd_interface(interface_below, log)
        return interface_below

    def _sanity_check_2nd_interface(self,
                                    interface_below: np.ndarray,
                                    log: logger.logging.Logger
                                    ) -> None:
        """check the interface_below"""
        average = np.mean(interface_below)
        if average < 0:
            msg: str = \
                f'Error! The interface_below has negative mean: {average}!\n'
            log.error(msg)
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
        if np.any(interface_below - average >= 2 * average):
            msg = 'Error! There is a big shift in frame(s) shift!\n'
            log.error(msg)
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
        if np.any(interface_below == 0):
            msg = 'Error! The second interface at zero!\n'
            log.error(msg)
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    RealValumeRdf(sys.argv[1], logger.setup_logger('real_volume_rdf.log'))
