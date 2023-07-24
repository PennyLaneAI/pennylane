"""This module enables support for log-level messaging throughout PennyLane, following the native Python logging framework interface. Please see the official documentation for details on usage https://docs.python.org/3/library/logging.html"""
import logging
import os

import pytoml

_path = os.path.dirname(__file__)


def enable_logging():
    """
    This method allows top selectively enable logging throughout PennyLane.
    All configurations are read through the `log_config.toml` file.
    """

    with open(os.path.join(_path, "log_config.toml"), "r") as f:
        config = pytoml.load(f)
        logging.config.dictConfig(config)
