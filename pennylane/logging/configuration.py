# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains support methods for configuring the logging functionality.
"""
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

# Define a more verbose log-level. Not currently controlled by internal log configurations.
TRACE = logging.DEBUG // 2


def _add_trace_level():
    "Wrapper to define custom TRACE level for PennyLane logging"

    def trace(self, message, *args, **kws):
        """Enable a more verbose mode than DEBUG. Used to enable inspection of function definitions in log messages."""

        # Due to limitations in how the logging module exposes support for custom levels,
        # accessing the private method `_log` has no alternative.
        # pylint: disable=protected-access
        self._log(TRACE, message, args, **kws)

    logging.addLevelName(TRACE, "TRACE")
    logging.TRACE = TRACE
    lc = logging.getLoggerClass()
    lc.trace = trace


def _configure_logging(config_file):
    """
    This method allows custom logging configuration throughout PennyLane.
    All configurations are read through the ``log_config.toml`` file.
    """
    if not has_toml:
        raise ImportError(
            "A TOML parser is required to enable PennyLane logging defaults. "
            "We support any of the following TOML parsers: [tomli, tomlkit, tomllib] "
            "You can install either tomli via `pip install tomli`, "
            "tomlkit via `pip install tomlkit`, or use Python 3.11 "
            "or above which natively offers the tomllib library."
        )
    with open(os.path.join(_path, config_file), "rb") as f:
        pl_config = tomllib.load(f)
        logging.config.dictConfig(pl_config)


def enable_logging():
    """
    This method allows to selectively enable logging throughout PennyLane, following the configuration options defined in the ``log_config.toml`` file.

    Enabling logging through this method will override any externally defined logging configurations.

    **Example**

    >>> qml.logging.enable_logging()
    """
    _add_trace_level()
    _configure_logging("log_config.toml")


def config_path():
    """
    This method returns the full absolute path to the the ``log_config.toml`` configuration file.

    Returns:
        str: System path to the ``log_config.toml`` file.

    **Example**

    >>> config_path()
    /home/user/pyenv/lib/python3.10/site-packages/pennylane/logging/log_config.toml
    """
    path = os.path.join(_path, "log_config.toml")
    return path
