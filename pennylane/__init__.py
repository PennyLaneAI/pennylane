# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
from importlib import reload
import pkg_resources

import numpy as _np

from semantic_version import Version, Spec

# QueuingContext needs to be imported before all other pennylane imports
from ._queuing import QueuingContext  # pylint: disable=wrong-import-order
import pennylane.operation

import pennylane.init
import pennylane.templates
import pennylane.qnn
import pennylane.qaoa as qaoa
from pennylane.templates import template, broadcast, layer
from pennylane.about import about
from pennylane.vqe import Hamiltonian, ExpvalCost, VQECost

from .circuit_graph import CircuitGraph
from .configuration import Configuration
from ._device import Device, DeviceError
from .collections import apply, map, sum, dot, QNodeCollection
from ._qubit_device import QubitDevice
from .measure import expval, var, sample, probs
from .ops import *
from .optimize import *
from .qnodes import qnode, QNode, QuantumFunctionError
from .utils import inv
from ._version import __version__
from .io import *
from ._grad import jacobian, grad

import pennylane.math  # pylint: disable=wrong-import-order
import pennylane.tape  # pylint: disable=wrong-import-order
from .tape import enable_tape, disable_tape, tape_mode_active
from .tape.qnode import draw, metric_tensor

# Look for an existing configuration file
default_config = Configuration("config.toml")


def _get_device_entrypoints():
    """Returns a dictionary mapping the device short name to the
    loadable entrypoint"""
    return {entry.name: entry for entry in pkg_resources.iter_entry_points("pennylane.plugins")}


def refresh_devices():
    """Scan installed PennyLane plugins to refresh the device list."""

    # This function does not return anything; instead, it has a side effect
    # which is to update the global plugin_devices variable.

    # We wish to retain the behaviour of a global plugin_devices dictionary,
    # as re-importing pkg_resources can be a very slow operation on systems
    # with a large number of installed packages.
    global plugin_devices  # pylint:disable=global-statement

    reload(pkg_resources)
    plugin_devices = _get_device_entrypoints()


# get list of installed devices
plugin_devices = _get_device_entrypoints()


# get chemistry plugin
class NestedAttrError:
    """This class mocks out the qchem module in case
    it is not installed. Any attempt to print an instance
    of this class, or to access an attribute of this class,
    results in an import error, directing the user to the installation
    instructions for PennyLane Qchem"""

    error_msg = (
        "PennyLane-QChem not installed. \n\nTo access the qchem "
        "module, you can install PennyLane-QChem via pip:"
        "\n\npip install pennylane-qchem"
        "\n\nFor more details, see the quantum chemistry documentation:"
        "\nhttps://pennylane.readthedocs.io/en/stable/introduction/chemistry.html"
    )

    def __str__(self):
        raise ImportError(self.error_msg) from None

    def __getattr__(self, name):
        raise ImportError(self.error_msg) from None

    __repr__ = __str__


qchem = NestedAttrError()

for entry in pkg_resources.iter_entry_points("pennylane.qchem"):
    if entry.name == "OpenFermion":
        qchem = entry.load()


def device(name, *args, **kwargs):
    r"""device(name, wires=1, *args, **kwargs)
    Load a :class:`~.Device` and return the instance.

    This function is used to load a particular quantum device,
    which can then be used to construct QNodes.

    PennyLane comes with support for the following devices:

    * :mod:`'default.qubit' <pennylane.devices.default_qubit>`: a simple
      state simulator of qubit-based quantum circuit architectures.

    * :mod:`'default.gaussian' <pennylane.devices.default_gaussian>`: a simple simulator
      of Gaussian states and operations on continuous-variable circuit architectures.

    * :mod:`'default.qubit.tf' <pennylane.devices.default_qubit_tf>`: a state simulator
      of qubit-based quantum circuit architectures written in TensorFlow, which allows
      automatic differentiation through the simulation.

    * :mod:`'default.qubit.autograd' <pennylane.devices.default_qubit_autograd>`: a state simulator
      of qubit-based quantum circuit architectures which allows
      automatic differentiation through the simulation via python's autograd library.

    In addition, additional devices are supported through plugins — see
    the  `available plugins <https://pennylane.ai/plugins.html>`_ for more
    details.

    All devices must be loaded by specifying their **short-name** as listed above,
    followed by the **wires** (subsystems) you wish to initialize. The *wires*
    argument can be an integer, in which case the wires of the device are addressed
    by consecutive integers:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=5)

        def circuit():
           qml.Hadamard(wires=1)
           qml.Hadamard(wires=[0])
           qml.CNOT(wires=[3, 4])
           ...

    The *wires* argument can also be a sequence of unique numbers or strings, specifying custom wire labels
    that the user employs to address the wires:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=['ancilla', 'q11', 'q12', -1, 1])

        def circuit():
           qml.Hadamard(wires='q11')
           qml.Hadamard(wires=['ancilla'])
           qml.CNOT(wires=['q12', -1] )
           ...


    Some devices may accept additional arguments. For instance,
    ``default.gaussian`` accepts the keyword argument ``hbar``, to set
    the convention used in the commutation relation :math:`[\x,\p]=i\hbar`
    (by default set to 2).

    Please refer to the documentation for the individual devices to see any
    additional arguments that might be required or supported.

    Args:
        name (str): the name of the device to load
        wires (int): the number of wires (subsystems) to initialise
            the device with

    Keyword Args:
        config (pennylane.Configuration): a PennyLane configuration object
            that contains global and/or device specific configurations.
    """
    if name not in plugin_devices:
        # Device does not exist in the loaded device list.
        # Attempt to refresh the devices, in case the user
        # installed the plugin during the current Python session.
        refresh_devices()

    if name in plugin_devices:
        options = {}

        # load global configuration settings if available
        config = kwargs.get("config", default_config)

        if config:
            # combine configuration options with keyword arguments.
            # Keyword arguments take preference, followed by device options,
            # followed by plugin options, followed by global options.
            options.update(config["main"])
            options.update(config[name.split(".")[0] + ".global"])
            options.update(config[name])

        kwargs.pop("config", None)
        options.update(kwargs)

        # loads the device class
        plugin_device_class = plugin_devices[name].load()

        if Version(version()) not in Spec(plugin_device_class.pennylane_requires):
            raise DeviceError(
                "The {} plugin requires PennyLane versions {}, however PennyLane "
                "version {} is installed.".format(
                    name, plugin_device_class.pennylane_requires, __version__
                )
            )

        # load device
        return plugin_device_class(*args, **options)

    raise DeviceError("Device does not exist. Make sure the required plugin is installed.")


def version():
    """Returns the PennyLane version number."""
    return __version__


enable_tape()
