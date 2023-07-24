"""This module enables support for log-level messaging throughout PennyLane, following the native Python logging framework interface. Please see the official documentation for details on usage https://docs.python.org/3/library/logging.html"""
import logging
import logging.config
import os
from importlib import import_module
from importlib.util import find_spec

has_toml = False
toml_libs = ["tomllib", "tomli", "tomlkit"]
for pkg in toml_libs:
    spec = find_spec(pkg)
    if spec:
        tomllib = import_module(pkg)
        has_toml = True
        break

# Define absolute path to this file in source tree
_path = os.path.dirname(__file__)

# Define a more verbose mode for the messages. Not currently controlled by internal log configurations.
TRACE = logging.DEBUG // 2


def enable_logging():
    """
    This method allows top selectively enable logging throughout PennyLane.
    All configurations are read through the `log_config.toml` file.
    """
    if not has_toml:
        raise ImportError(
            "A TOML parser is required to enable PennyLane logging defaults. "
            "You can install tomli via `pip install tomli`, "
            "install tomlkit via `pip install tomlkit`, "
            "or use Python 3.11 which natively offers the tomllib library."
        )

    with open(os.path.join(_path, "log_config.toml"), "rb") as f:
        pl_config = tomllib.load(f)
        logging.config.dictConfig(pl_config)

    def trace(self, message, *args, **kws):
        """Enable a more verbose mode than DEBUG. Used to enable inspection of function definitions in log messages."""

        # Due to limitations in how the logging module exposes support for custom levels, accessing the private method `_log` has no alternative.
        # pylint: disable=protected-access
        self._log(TRACE, message, args, **kws)

    logging.addLevelName(TRACE, "TRACE")
    logging.TRACE = TRACE
    lc = logging.getLoggerClass()
    lc.trace = trace
