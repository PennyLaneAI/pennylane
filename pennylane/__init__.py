# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
import types
import pkg_resources

import numpy as _np
from semantic_version import Spec, Version

from pennylane.boolean_fn import BooleanFn
from pennylane.queuing import apply, QueuingContext

import pennylane.fourier
import pennylane.kernels
import pennylane.math
import pennylane.operation
import pennylane.qnn
import pennylane.templates
import pennylane.hf
from pennylane._device import Device, DeviceError
from pennylane._grad import grad, jacobian, finite_diff
from pennylane._qubit_device import QubitDevice
from pennylane._version import __version__
from pennylane.about import about
from pennylane.circuit_graph import CircuitGraph
from pennylane.configuration import Configuration
from pennylane.tracker import Tracker
from pennylane.io import *
from pennylane.measurements import density_matrix, measure, expval, probs, sample, state, var
from pennylane.ops import *
from pennylane.templates import broadcast, layer
from pennylane.templates.embeddings import *
from pennylane.templates.layers import *
from pennylane.templates.tensornetworks import *
from pennylane.templates.state_preparations import *
from pennylane.templates.subroutines import *
from pennylane import qaoa
from pennylane.qnode import QNode, qnode
import pennylane.qnode_old
from pennylane.transforms import (
    adjoint,
    adjoint_metric_tensor,
    batch_params,
    batch_transform,
    draw,
    draw_mpl,
    ControlledOperation,
    compile,
    ctrl,
    cond,
    defer_measurements,
    measurement_grouping,
    metric_tensor,
    specs,
    qfunc_transform,
    op_transform,
    single_tape_transform,
    quantum_monte_carlo,
    apply_controlled_Q,
    commutation_dag,
    is_commuting,
    simplify,
)
from pennylane.ops.functions import *
from pennylane.optimize import *
from pennylane.vqe import ExpvalCost, VQECost
from pennylane.debugging import snapshots

# QueuingContext and collections needs to be imported after all other pennylane imports
from .collections import QNodeCollection, dot, map, sum
import pennylane.grouping  # pylint:disable=wrong-import-order
import pennylane.gradients  # pylint:disable=wrong-import-order
from pennylane.interfaces.batch import execute  # pylint:disable=wrong-import-order

# Look for an existing configuration file
default_config = Configuration("config.toml")


class QuantumFunctionError(Exception):
    """Exception raised when an illegal operation is defined in a quantum function."""


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

    Additional devices are supported through plugins — see
    the  `available plugins <https://pennylane.ai/plugins.html>`_ for more
    details.

    Args:
        name (str): the name of the device to load
        wires (int): the number of wires (subsystems) to initialise
            the device with

    Keyword Args:
        config (pennylane.Configuration): a PennyLane configuration object
            that contains global and/or device specific configurations.
        custom_decomps (Dict[Union(str, qml.Operator), Callable]): Custom
            decompositions to be applied by the device at runtime.
        decomp_depth (int): For when custom decompositions are specified,
            the maximum expansion depth used by the expansion function.

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

    Most devices accept a ``shots`` argument which specifies how many circuit executions
    are used to estimate stochastic return values. In particular, ``qml.sample()`` measurements
    will return as many samples as specified in the shots argument. The shots argument can be
    changed on a per-call basis using the built-in ``shots`` keyword argument.

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1, shots=10)

        @qml.qnode(dev)
        def circuit(a):
          qml.RX(a, wires=0)
          return qml.sample(qml.PauliZ(wires=0))

    >>> circuit(0.8)  # 10 samples are returned
    [ 1  1  1 -1 -1  1  1  1  1  1]
    >>> circuit(0.8, shots=3))  # default is overwritten for this call
    [1 1 1]
    >>> circuit(0.8)  # back to default of 10 samples
    [ 1  1  1 -1 -1  1  1  1  1  1]

    When constructing a device, we may optionally pass a dictionary of custom
    decompositions to be applied to certain operations upon device execution.
    This is useful for enabling support of gates on devices where they would normally
    be unsupported.

    For example, suppose we are running on an ion trap device which does not
    natively implement the CNOT gate, but we would still like to write our
    circuits in terms of CNOTs. On a ion trap device, CNOT can be implemented
    using the ``IsingXX`` gate. We first define a decomposition function
    (such functions have the signature ``decomposition(*params, wires)``):

    .. code-block:: python

        def ion_trap_cnot(wires):
            return [
                qml.RY(np.pi/2, wires=wires[0]),
                qml.IsingXX(np.pi/2, wires=wires),
                qml.RX(-np.pi/2, wires=wires[0]),
                qml.RY(-np.pi/2, wires=wires[0]),
                qml.RY(-np.pi/2, wires=wires[1])
            ]

    Next, we create a device, and a QNode for testing. When constructing the
    QNode, we can set the expansion strategy to ``"device"`` to ensure the
    decomposition is applied and will be viewable when we draw the circuit.

    .. code-block:: python

        # As the CNOT gate normally has no decomposition, we can use default.qubit
        # here for expository purposes.
        dev = qml.device(
            'default.qubit', wires=2, custom_decomps={"CNOT" : ion_trap_cnot}
        )

        @qml.qnode(dev, expansion_strategy="device")
        def run_cnot():
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=1))

    >>> print(qml.draw(run_cnot)())
    0: ──RY(1.57)─╭IsingXX(1.57)──RX(-1.57)──RY(-1.57)─┤
    1: ───────────╰IsingXX(1.57)──RY(-1.57)────────────┤  <X>

    Some devices may accept additional arguments. For instance,
    ``default.gaussian`` accepts the keyword argument ``hbar``, to set
    the convention used in the commutation relation :math:`[\x,\p]=i\hbar`
    (by default set to 2).

    Please refer to the documentation for the individual devices to see any
    additional arguments that might be required or supported.
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

        # Pop the custom decomposition keyword argument; we will use it here
        # only and not pass it to the device.
        custom_decomps = kwargs.pop("custom_decomps", None)
        decomp_depth = kwargs.pop("decomp_depth", 10)

        kwargs.pop("config", None)
        options.update(kwargs)

        # loads the device class
        plugin_device_class = plugin_devices[name].load()

        if Version(version()) not in Spec(plugin_device_class.pennylane_requires):
            raise DeviceError(
                f"The {name} plugin requires PennyLane versions {plugin_device_class.pennylane_requires}, "
                f"however PennyLane version {__version__} is installed."
            )

        # Construct the device
        dev = plugin_device_class(*args, **options)

        # Once the device is constructed, we set its custom expansion function if
        # any custom decompositions were specified.
        if custom_decomps is not None:
            custom_decomp_expand_fn = pennylane.transforms.create_decomp_expand_fn(
                custom_decomps, dev, decomp_depth=decomp_depth
            )
            dev.custom_expand(custom_decomp_expand_fn)

        return dev

    raise DeviceError("Device does not exist. Make sure the required plugin is installed.")


def version():
    """Returns the PennyLane version number."""
    return __version__


# add everything as long as it's not a module and not prefixed with _
_all = sorted(
    [
        name
        for name, function in globals().items()
        if not (name.startswith("_") or isinstance(function, types.ModuleType))
    ]
)


_qchem = None


def __getattr__(name):
    """Ensure that the qchem module is imported lazily"""
    if name == "qchem":
        global _qchem  # pylint: disable=global-statement

        if _qchem is None:

            for entry in pkg_resources.iter_entry_points("pennylane.qchem"):
                if entry.name == "OpenFermion":
                    _qchem = entry.load()

            if _qchem is None:
                raise ImportError(
                    "PennyLane-QChem not installed. \n\nTo access the qchem "
                    "module, you can install PennyLane-QChem via pip:"
                    "\n\npip install pennylane-qchem"
                    "\n\nFor more details, see the quantum chemistry documentation:"
                    "\nhttps://pennylane.readthedocs.io/en/stable/introduction/chemistry.html"
                )

        return _qchem

    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():  # pragma: no cover
    return _all + ["qchem"]
