"""
Reads a force field file and writes it to a LaTeX file.
The files are in itp format, which is a GROMACS file format.
The main data are in the data dircetory for the parents of this module.
Read itp files, clean up, if there are similar values for the similar
atoms, it is better to combine them.
If needed some extera comments can be added to the file.
"""


import hydra
from omegaconf import DictConfig

from common import logger

from module13_backing_up_data.forcefield_to_latex_read_itp import \
    ProccessForceField
from module13_backing_up_data.forcefield_to_latex_write_tex import WriteTex


@hydra.main(version_base=None,
            config_path="conf",
            config_name="config")
def main(cfg: DictConfig) -> None:
    """main function"""
    log: logger.logging.Logger = logger.setup_logger('ff_to_latex.log')
    # read and process the data
    latex_itp: ProccessForceField = ProccessForceField(cfg, log)
    # write the data to a LaTeX file
    WriteTex(latex_itp, cfg, log)

# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
