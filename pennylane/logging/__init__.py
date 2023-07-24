"""This module enables support for log-level messaging throughout PennyLane, following the native Python logging framework interface. Please see the official documentation for details on usage https://docs.python.org/3/library/logging.html"""
import logging
import os

import pytoml

# Define a more verbose mode for the messages. Not currently controlled by internal log configurations.
TRACE = 1
_path = os.path.dirname(__file__)


def enable_logging():
    """
    This method allows top selectively enable logging throughout PennyLane.
    All configurations are read through the `log_config.toml` file.
    """

    with open(os.path.join(_path, "log_config.toml"), "r") as f:
        config = pytoml.load(f)
        logging.config.dictConfig(config)

    # Enable a more verbose mode than DEBUG.
    # Used to enable inspection of function definitions in log messages.
    def trace(self, message, *args, **kws):
        self._log(TRACE, message, args, **kws)

    logging.addLevelName(TRACE, "TRACE")
    lc = logging.getLoggerClass()
    lc.trace = trace
