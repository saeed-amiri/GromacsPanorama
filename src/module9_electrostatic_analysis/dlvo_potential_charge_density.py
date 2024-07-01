"""
Compute the charge density of the NP in water.
"""

from datetime import datetime

import numpy as np
import pandas as pd


from common import logger, xvg_to_dataframe
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_configs import \
    AllConfig


class ChargeDensity:
    """compute the true density for the part inside water"""
    # pylint: disable=too-few-public-methods

    info_msg: str = 'Message from ChargeDensity:\n'
    density: np.ndarray
    charge: np.ndarray

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig
                 ) -> None:
        self.charge, self.density = self._get_density(configs, log)
        self._write_msg(log)

    def _get_density(self,
                     configs: AllConfig,
                     log: logger.logging.Logger
                     ) -> tuple[np.ndarray, np.ndarray]:
        """read the input files and compute the charge desnity"""
        charge: np.ndarray = \
            self._get_column(configs.charge_fname,
                             log, column=configs.total_charge_coloumn)
        try:
            contact_angle: np.ndarray = \
                self._get_column(configs.contact_fname,
                                 log,
                                 column='contact_angles',
                                 if_exit=False)
            self.info_msg += \
                f'\tContact angle is getting from `{configs.contact_fname}`\n'
        except FileNotFoundError as _:
            self.info_msg += \
                (f'\t`{configs.contact_fname}` not found!\n\tAaverage '
                 f'angle `{configs.avg_contact_angle}` is used.\n')
            contact_angle = np.zeros(charge.shape)
            contact_angle += configs.avg_contact_angle

        cap_surface: np.ndarray = \
            self._compute_under_water_area(configs.np_radius, contact_angle)

        np_cap_charge_density: float = \
            (configs.nr_aptes_charges - configs.np_core_charge) *\
            configs.phi_parameters['e_charge'] / cap_surface.mean()

        density: np.ndarray = \
            (charge * configs.phi_parameters['e_charge']) / cap_surface

        self.info_msg += (
            f'\tAve. `{charge.mean() = :.3f}` [e]\n'
            f'\t`{cap_surface.mean()*1e18 = :.3f}` [nm^2]\n'
            f'\tAve. `charge_{density.mean() = :.3f}` [C/m^2] or [As/m^2] \n'
            f'\tThe charge density of the NP in the water is:\n'
            f'\t{np_cap_charge_density = :.3f} [C/m^2]\n')
        return charge, density

    def _compute_under_water_area(self,
                                  np_radius: float,
                                  contact_angle: np.ndarray
                                  ) -> np.ndarray:
        """
        Compute the area under water for a NP at a given contact angle.

        The area is calculated based on the NP's radius and the contact
        angle, assuming the contact angle provides the extent of
        exposure to water.

        Parameters:
        - np_radius: Radius of the NP in Ångströms.
        - contact_angle: Contact angle(s) in degrees.

        Returns:
        - The area of the NP's surface exposed to water, in nm^2.
        """
        # Convert angles from degrees to radians for mathematical operations
        radians: np.ndarray = np.deg2rad(contact_angle)

        # Compute the surface area of the cap exposed to water
        # Formula: A = 2 * pi * r^2 * (1 + cos(θ))
        # Alco converted from Ångströms^2 to m^2
        in_water_cap_area: np.ndarray = \
            2 * np.pi * np_radius**2 * (1 + np.cos(radians))
        return in_water_cap_area * 1e-20

    def _get_column(self,
                    fname: str,
                    log: logger.logging.Logger,
                    column: str,
                    if_exit: bool = True
                    ) -> np.ndarray:
        """read the file and return the column in an array"""
        df_i: pd.DataFrame = \
            xvg_to_dataframe.XvgParser(fname, log, if_exit=if_exit).xvg_df
        return df_i[column].to_numpy(dtype=float)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{ChargeDensity.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print(f'{bcolors.CAUTION}This module is not meant to be run '
          f'independently.{bcolors.ENDC}\n')
