"""
ConfigCpuNr Module Documentation
-------------------------------

The `ConfigCpuNr` module provides a class, `ConfigCpuNr`, that
facilitates the configuration of the number of available CPU cores
based on the host environment where a script is running. This module
is particularly useful for optimizing multi-core computation by
adapting the core utilization to different computing environments.

Module Overview:
-----------------

The `ConfigCpuNr` class in this module allows for dynamic determination
of the number of CPU cores that should be utilized for a computational
task. By considering the hostname and the type of computing environment,
the class can intelligently adjust the core count to achieve efficient
resource utilization.

Class Details:
--------------

class ConfigCpuNr:
    ConfigCpuNr class is designed to determine the optimal number of
    CPU cores to use for a computation based on the host environment.

    Attributes:
        hostname (str): The hostname of the machine where the script
        is running.
        cores_nr (int): The configured number of CPU cores for the
        computation.

    Methods:
        __init__(self, log: logger.logging.Logger)
            Initializes the ConfigCpuNr instance and configures the
            number of available CPU cores based on the host environment.

        set_core_numbers(self) -> int
            Determines the appropriate number of CPU cores based on
            the hostname and specific rules for different computing
            environments.

        get_hostname(self) -> str
            Retrieves and returns the hostname of the machine.

        get_core_nr(self) -> int
            Retrieves the total number of available CPU cores.

        write_log_msg(self, log: logger.logging.Logger) -> None
            Logs informative messages about core configuration to the
            provided logger instance and prints the messages to the
            console.

    Usage Example:
    --------------

    from ConfigCpuNr import ConfigCpuNr
    import logger

    if __name__ == '__main__':
        # Create a logger instance
        log = logger.setup_logger(log_name='ConfigeCpu.log')

        # Initialize ConfigCpuNr and set the number of CPU cores
        cpu_config = ConfigCpuNr(log=log)

        # Access the configured core count
        num_cores = cpu_config.cores_nr
        print(f"Number of cores configured for computation: {num_cores}")

Module Dependencies:
--------------------

- `socket`: Provides access to low-level networking interfaces.
- `multiprocessing`: Enables multi-core processing and CPU count
    retrieval.
- `logger`: An external logger module for logging purposes.
- `colors_text`: An external module for text color formatting.

Notes:
------

- To use this module effectively, ensure that the necessary dependencies
    are available.
- This module makes use of a custom logger module (not provided) and a
    text color formatting module (not provided) for enhanced logging and
    console output.

"""


import socket
import multiprocessing
import common.logger as logger
from common.colors_text import TextColor as bcolors


class ConfigCpuNr:
    """
    Find the number of core
    """

    info_msg: str = 'message from ConfigCpuNr:\n'  # Meesage in methods to log
    local_host: str = 'hmigws03'  # Name of the host in the office
    server_front_host: list[str] = ['glogin', 'blogin']  # Front names in HLRN
    # Name of the goettingen of HLRN
    server_host_list: list[str] = ['gcn', 'gfn', 'gsn', 'bcn', 'bfn', 'bsn']

    def __init__(self,
                 log: logger.logging.Logger
                 ) -> None:
        self.hostname: str = self.get_hostname()
        self.cores_nr: int = self.set_core_numbers()
        self.write_log_msg(log)

    def set_core_numbers(self) -> int:
        """set the nmbers of the cores based on the hostname"""
        aval_core_nr: int = self.get_core_nr()
        if self.hostname == self.local_host:
            # In local machine only using half of the cores
            cores_nr = int(aval_core_nr // 2)
        elif self.hostname[:6] in self.server_front_host:
            # On frontend use only 4 since it is for all
            cores_nr = 4
        elif self.hostname[:3] in self.server_host_list:
            # On the backends use all the physical cores
            cores_nr = int(aval_core_nr // 2)
        else:
            cores_nr = int(aval_core_nr)
        self.info_msg += (f'\t\tNumber of cores for this computation is'
                          f' set to: `{cores_nr}`\n')
        return cores_nr

    def get_hostname(self) -> str:
        """Retrieve the hostname of the machine."""
        try:
            hostname = socket.gethostname()
            self.info_msg += f'\t\tHostname is `{hostname}`\n'
        except socket.error as err:
            raise RuntimeError("Failed to retrieve hostname.") from err
        return hostname

    def get_core_nr(self) -> int:
        """return numbers of cores"""
        aval_core_nr: int = multiprocessing.cpu_count()
        self.info_msg += \
            f'\t\tNumber of available cores of the host is: `{aval_core_nr}`\n'
        return aval_core_nr

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == '__main__':
    ConfigCpuNr(log=logger.setup_logger(log_name='ConfigCpu.log'))
