# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This is the top level module from which all basic functions and classes of
PennyLane can be directly imported.
"""

import importlib as _importlib

from pennylane._version import __version__
from pennylane.configuration import Configuration

# Look for an existing configuration file
default_config = Configuration("config.toml")


class DeviceError(Exception):
    """Exception raised when it encounters an illegal operation in the quantum circuit."""


class QuantumFunctionError(Exception):
    """Exception raised when an illegal operation is defined in a quantum function."""


class PennyLaneDeprecationWarning(UserWarning):
    """Warning raised when a PennyLane feature is being deprecated."""


class ExperimentalWarning(UserWarning):
    """Warning raised to indicate experimental/non-stable feature or support."""


top_level_accessible = {
    # Define what modules are accessible at the top level
}

submodules = [
    "numpy",
    "compiler",
    "capture",
    "control_flow",
    "kernels",
    "math",
    "operation",
    "decomposition",
    "qnn",
    "templates",
    "pauli",
    "resource",
    "qchem",
    "qaoa",
    "pulse",
    "fourier",
    "gradients",
    "logging",
    "data",
    "noise",
    "liealg",
    "spin",
]


# pylint: disable=no-else-return
def __getattr__(name):

    # pylint: disable=import-outside-toplevel
    if name == "plugin_devices":
        from pennylane.devices.device_constructor import plugin_devices

        return plugin_devices

    if name in submodules:
        return _importlib.import_module(f"pennylane.{name}")
    elif name in top_level_accessible:
        return _importlib.import_module(f"{top_level_accessible[name]}.{name}")
    else:
        try:
            return globals()[name]
        except KeyError as exc:
            raise AttributeError(f"module 'pennylane' has no attribute '{name}'") from exc


def version():
    """Returns the PennyLane version number."""
    return __version__
