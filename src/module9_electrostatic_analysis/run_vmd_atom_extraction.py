"""
The `run_vmd_atom_extraction.py` script automates the process of
extracting atom data from molecular dynamics trajectories using
Visual Molecular Dynamics (VMD). It simplifies the workflow by
generating a custom TCL script based on user-defined parameters and
subsequently executing VMD with this script to perform the atom
extraction.

Key Features:
    - Update a TCL script tailored for atom extraction from a
    trajectory file.
    - Executes VMD with the generated TCL script.
    - Supports customization of extraction criteria, including atom
    selection and output format, through user inputs.



Note: This script requires a working installation of VMD and assumes
familiarity with VMD's TCL scripting for custom extraction logic.

Opt. ChatGPT
14 Feb 2024
Saeed
"""

import os
import sys
import typing
import subprocess
from dataclasses import dataclass, field

from common import logger
from common.colors_text import TextColor as bcolors


@dataclass
class TclBase:
    """template for the tcl run"""
    tcl_base: str = '''
    # Load the gro and trr and with optinal number of frames
    set mol [mol new @GRO  type @TYPE waitfor all]
    puts "Attempting to load file: $mol"
    set distance 5

    mol addfile @TRR type trr first @FIRST last @LAST step @STEP\
    waitfor all molid $mol

    set nf [molinfo top get numframes]
    puts " The numbers of frames are $nf"
    for {set i 0} {$i < $nf} {incr i} {
        animate goto $i
        set inRadiusResidues [atomselect top "@ATOMSELCT"]
        set resIDs [$inRadiusResidues get residue]
        if {[llength $resIDs] > 0} {
            set uniqueResIDs [lsort -unique $resIDs]
            set completeInRadiusResidues\
            [atomselect top "@FINALSELECT $uniqueResIDs"]
            $completeInRadiusResidues @WRITE "@OUTNAME_${i}.@TYPE"
            $completeInRadiusResidues delete
        } else {
            puts "No residues within $distance Å of APT or COR in frame $i"
        }
        $inRadiusResidues delete
    }

    exit

    '''


@dataclass
class ParameterConfig:
    """set the parameters for the modifying the tcl file"""
    key_values: dict[str, str] = field(default_factory=lambda: ({
        'GRO': 'center_mol_unwrap.gro',
        'TRR': 'center_mol_unwrap.trr',
        'FIRST': '0',
        'LAST': '1',
        'STEP': '1',
        'ATOMSELCT': 'resname APT COR',
        'FINALSELEC': 'resname APT COR',
        'WRITE': 'writegro',
        'OUTNAME': 'apt_cor',
        'TYPE': 'gro'
        }))


@dataclass
class AllConfig(TclBase, ParameterConfig):
    """set the all the parameters and configurations"""
    fout: str = 'atom_extract.tcl'


class ExecuteTclVmd:
    """
    update the tcl input for the vmd based on the inputs
    """

    info_msg: str = 'Message from ExecuteTclVmd:\n'
    configs: AllConfig
    struct_files: list[str]

    def __init__(self,
                 src: str,  # Path of the working dir (pwd)
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.prepare_tcl(src)
        stdout: typing.Union[str, None] = self.execute_vmd(log)
        self.struct_files = self.get_structs_names(stdout)
        self.write_log_msg(log)

    def prepare_tcl(self,
                    src: str
                    ) -> None:
        """prepare and write the tcl file for vmd"""
        self._set_path(src)
        updated_tcl: str = self._update_tcl()
        self._write_tcl(updated_tcl)

    def execute_vmd(self,
                    log: logger.logging.Logger
                    ) -> typing.Union[str, None]:
        """exacute the vmd file to get the files"""
        command = ["vmd", "-dispdev", "text", "-e", self.configs.fout]
        result = subprocess.run(
            command, capture_output=True, text=True, check=True)

        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True)
            self.info_msg += "\tCommand executed successfully! The log is:\n\n"
            self.info_msg += result.stdout
            return result.stdout
        except subprocess.CalledProcessError as err:
            msg: str = \
                f"\tError! in executing command: {result.stderr}\n{err}\n"
            log.error(msg)
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

    def get_structs_names(self,
                          stdout: typing.Union[str, None]
                          ) -> list[str]:
        """trim the name of the output files and return it as a list"""
        if stdout is not None:
            lines_with_name: list[str] = [
                item for item in stdout.split('\n') if
                (item.startswith('Info) Finished with coordinate file') &
                 (self.configs.key_values['OUTNAME'] in item))]
            files: list[str] = [
                item.split('coordinate file')[1].split('.', -1)[0] for
                item in lines_with_name]
            files = [
                f'{item}.{self.configs.key_values["TYPE"]}' for item in files]
            return files
        return ['None']

    def _update_tcl(self) -> str:
        """
        Replace placeholders in the template with values from
        ParameterConfig.
        """
        template: str = self.configs.tcl_base
        for key, value in self.configs.key_values.items():
            placeholder = f"@{key}"
            template = template.replace(placeholder, value)
            self.info_msg += f'\t`{placeholder}` is replaced by `{value}`\n'
        return template

    def _set_path(self,
                  src: str
                  ) -> None:
        """update the pathes for the gro and trr files"""
        self.configs.key_values['GRO'] = \
            os.path.join(src, self.configs.key_values['GRO'])
        self.configs.key_values['TRR'] = \
            os.path.join(src, self.configs.key_values['TRR'])

    def _write_tcl(self,
                   updated_tcl: str
                   ) -> None:
        """Write the updated tcl into a file"""
        with open(fout := self.configs.fout, 'w', encoding='utf8') as f_w:
            f_w.write(updated_tcl)
        self.info_msg += f'\tThe tcl file is saved as `{fout}`\n'

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == '__main__':
    ExecuteTclVmd(src=os.getcwd(), log=logger.setup_logger('run_tcl.log'))
