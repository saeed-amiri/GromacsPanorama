"""
compute the ionic strength in the system in case of non-multivalent ions
"""

import sys
from datetime import datetime

from common import logger, my_tools
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_configs import \
    AllConfig


class IonicStrengthCalculation:
    """computation of the ionic strength in the system by considering
    all charge groups in the system.
    The number of charge group are a lot, and effect of the salt on
    the results is very small. The results are almost the same without
    any salt in the box. If the concentration of the salt is zero, then
    the computing the debye length is impossible!
    """
    # pylint: disable=too-few-public-methods

    info_msg: str = 'Message from IonicStrengthCalculation:\n'
    ionic_strength: float

    def __init__(self,
                 topol_fname: str,  # Name of the topology file
                 log: logger.logging.Logger,
                 configs: AllConfig
                 ) -> None:
        res_nr: dict[str, int] = my_tools.read_topol_resnr(topol_fname, log)
        self.ionic_strength = self.compute_ionic_strength(res_nr, configs, log)
        self._write_msg(log)

    def compute_ionic_strength(self,
                               res_nr: dict[str, int],
                               configs: AllConfig,
                               log: logger.logging.Logger
                               ) -> float:
        """compute the ionic strength based on the number of charge
        groups"""
        ionic_strength: float = 0.0
        volume: float = self._get_water_volume(configs)
        concentration_dict = \
            self._compute_concentration(res_nr, volume, configs)
        self._check_electroneutrality(
            res_nr, concentration_dict, configs.charge_sings, log)
        for res, _ in res_nr.items():
            ionic_strength += \
                concentration_dict[res] * configs.charge_sings[res]**2
        self.info_msg += f'\t{ionic_strength = :.5f} [mol/l]\n'
        return ionic_strength

    def _compute_concentration(self,
                               res_nr: dict[str, int],
                               volume: float,
                               configs: AllConfig
                               ) -> dict[str, float]:
        """compute the concentration for each charge group"""
        # pylint: disable=invalid-name
        concentration: dict[str, float] = {}
        for res, nr in res_nr.items():
            if res not in ['D10', 'APT_COR']:
                pass
            elif res == 'APT_COR':
                nr = configs.nr_aptes_charges
            elif res == 'D10':
                concentration[res] = 0.0
            concentration[res] = nr / (
                    volume * configs.phi_parameters['n_avogadro'])
        return concentration

    def _check_electroneutrality(self,
                                 res_nr: dict[str, int],  # nr. of charges
                                 concentration: dict[str, float],
                                 charge_sings: dict[str, int],
                                 log: logger.logging.Logger
                                 ) -> None:
        """check the electroneutrality of the system:
        must: \\sum_i c_i Z_i = 0
        """
        # pylint: disable=invalid-name
        electroneutrality: float = 0
        for res, _ in res_nr.items():
            electroneutrality += concentration[res] * charge_sings[res]
        if round(electroneutrality, 3) != 0.0:
            log.error(
                msg := f'\tError! `{electroneutrality = }` must be zero!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

    def _get_water_volume(self,
                          configs: AllConfig
                          ) -> float:
        """return volume of the water section in the box
        dimensions are in nm, the return should be in liters, so:
        nm^3 -> liters: (1e-9)^3 * 1000 = 1e-24 liters
        """
        volume: float = \
            configs.phi_parameters['box_xlim'] * \
            configs.phi_parameters['box_ylim'] * \
            configs.phi_parameters['box_zlim'] * 1e-24
        self.info_msg += f'\tWater section `{volume = :.4e}` liters\n'
        return volume

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{IonicStrengthCalculation.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print(f'{bcolors.CAUTION}This module is not meant to be run '
          f'independently.{bcolors.ENDC}\n')
