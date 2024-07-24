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
This module contains code for the main device construction delegation logic.
"""
import warnings
from importlib import metadata
from sys import version_info

from semantic_version import SimpleSpec, Version

import pennylane as qml


def _get_device_entrypoints():
    """Returns a dictionary mapping the device short name to the
    loadable entrypoint"""

    entries = (
        metadata.entry_points()["pennylane.plugins"]
        if version_info[:2] == (3, 9)
        # pylint:disable=unexpected-keyword-arg
        else metadata.entry_points(group="pennylane.plugins")
    )
    return {entry.name: entry for entry in entries}


# get list of installed devices
plugin_devices = _get_device_entrypoints()


def refresh_devices():
    """Scan installed PennyLane plugins to refresh the device list."""

    # This function does not return anything; instead, it has a side effect
    # which is to update the global plugin_devices variable.

    # We wish to retain the behaviour of a global plugin_devices dictionary,
    # as re-importing metadata can be a very slow operation on systems
    # with a large number of installed packages.

    global plugin_devices  # pylint:disable=global-statement
    plugin_devices = _get_device_entrypoints()


# pylint: disable=protected-access
def device(name, *args, **kwargs):
    r"""
    Load a device and return the instance.

    This function is used to load a particular quantum device,
    which can then be used to construct QNodes.

    PennyLane comes with support for the following devices:

    * :mod:`'default.qubit' <pennylane.devices.default_qubit>`: a simple
      state simulator of qubit-based quantum circuit architectures.

    * :mod:`'default.mixed' <pennylane.devices.default_mixed>`: a mixed-state
      simulator of qubit-based quantum circuit architectures.

    * ``'lightning.qubit'``: a more performant state simulator of qubit-based
      quantum circuit architectures written in C++.

    * :mod:`'default.qutrit' <pennylane.devices.default_qutrit>`: a simple
      state simulator of qutrit-based quantum circuit architectures.

    * :mod:`'default.qutrit.mixed' <pennylane.devices.default_qutrit_mixed>`: a
      mixed-state simulator of qutrit-based quantum circuit architectures.

    * :mod:`'default.gaussian' <pennylane.devices.default_gaussian>`: a simple simulator
      of Gaussian states and operations on continuous-variable circuit architectures.

    * :mod:`'default.clifford' <pennylane.devices.default_clifford>`: an efficient
      simulator of Clifford circuits.

    * :mod:`'default.tensor' <pennylane.devices.default_tensor>`: a simulator
      of quantum circuits based on tensor networks.

    Additional devices are supported through plugins — see
    the  `available plugins <https://pennylane.ai/plugins.html>`_ for more
    details. To list all currently installed devices, run
    :func:`qml.about <pennylane.about>`.

    Args:
        name (str): the name of the device to load
        wires (int): the number of wires (subsystems) to initialise
            the device with. Note that this is optional for certain
            devices, such as ``default.qubit``

    Keyword Args:
        config (pennylane.Configuration): a PennyLane configuration object
            that contains global and/or device specific configurations.
        custom_decomps (Dict[Union(str, Operator), Callable]): Custom
            decompositions to be applied by the device at runtime.
        decomp_depth (int): For when custom decompositions are specified,
            the maximum expansion depth used by the expansion function.

    .. warning::

        The ``decomp_depth`` argument is deprecated and will be removed in version 0.39.

    All devices must be loaded by specifying their **short-name** as listed above,
    followed by the **wires** (subsystems) you wish to initialize. The ``wires``
    argument can be an integer, in which case the wires of the device are addressed
    by consecutive integers:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=5)

        def circuit():
           qml.Hadamard(wires=1)
           qml.Hadamard(wires=[0])
           qml.CNOT(wires=[3, 4])
           ...

    The ``wires`` argument can also be a sequence of unique numbers or strings, specifying custom wire labels
    that the user employs to address the wires:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=['ancilla', 'q11', 'q12', -1, 1])

        def circuit():
           qml.Hadamard(wires='q11')
           qml.Hadamard(wires=['ancilla'])
           qml.CNOT(wires=['q12', -1])
           ...

    On some newer devices, such as ``default.qubit``, the ``wires`` argument can be omitted altogether,
    and instead the wires will be computed when executing a circuit depending on its contents.

    >>> dev = qml.device("default.qubit")

    Most devices accept a ``shots`` argument which specifies how many circuit executions
    are used to estimate stochastic return values. As an example, ``qml.sample()`` measurements
    will return as many samples as specified in the shots argument. The shots argument can be
    changed on a per-call basis using the built-in ``shots`` keyword argument. Note that the
    ``shots`` argument can be a single integer or a list of shot values.

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1, shots=10)

        @qml.qnode(dev)
        def circuit(a):
          qml.RX(a, wires=0)
          return qml.sample(qml.Z(0))

    >>> circuit(0.8)  # 10 samples are returned
    array([ 1,  1,  1,  1, -1,  1,  1, -1,  1,  1])
    >>> circuit(0.8, shots=[3, 4, 4])   # default is overwritten for this call
    (array([1, 1, 1]), array([ 1, -1,  1,  1]), array([1, 1, 1, 1]))
    >>> circuit(0.8)  # back to default of 10 samples
    array([ 1, -1,  1,  1, -1,  1,  1,  1,  1,  1])

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

        def ion_trap_cnot(wires, **_):
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
    Note that custom decompositions should accept keyword arguments even when
    it is not used.

    .. code-block:: python

        # As the CNOT gate normally has no decomposition, we can use default.qubit
        # here for expository purposes.
        dev = qml.device(
            'default.qubit', wires=2, custom_decomps={"CNOT" : ion_trap_cnot}
        )

        @qml.qnode(dev, expansion_strategy="device")
        def run_cnot():
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.X(1))

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
        config = kwargs.get("config", qml.default_config)

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
        decomp_depth = kwargs.pop("decomp_depth", None)

        if decomp_depth is not None:
            warnings.warn(
                "The decomp_depth argument is deprecated and will be removed in version 0.39. ",
                qml.PennyLaneDeprecationWarning,
            )
        else:
            decomp_depth = 10

        kwargs.pop("config", None)
        options.update(kwargs)

        # loads the device class
        plugin_device_class = plugin_devices[name].load()

        if hasattr(plugin_device_class, "pennylane_requires") and Version(
            qml.version()
        ) not in SimpleSpec(plugin_device_class.pennylane_requires):
            raise qml.DeviceError(
                f"The {name} plugin requires PennyLane versions {plugin_device_class.pennylane_requires}, "
                f"however PennyLane version {qml.version()} is installed."
            )

        # Construct the device
        dev = plugin_device_class(*args, **options)

        # Once the device is constructed, we set its custom expansion function if
        # any custom decompositions were specified.

        if custom_decomps is not None:
            if isinstance(dev, qml.devices.LegacyDevice):
                custom_decomp_expand_fn = qml.transforms.create_decomp_expand_fn(
                    custom_decomps, dev, decomp_depth=decomp_depth
                )
                dev.custom_expand(custom_decomp_expand_fn)
            else:
                custom_decomp_preprocess = qml.transforms.tape_expand._create_decomp_preprocessing(
                    custom_decomps, dev, decomp_depth=decomp_depth
                )
                dev.preprocess = custom_decomp_preprocess

        return dev

    raise qml.DeviceError(
        f"Device {name} does not exist. Make sure the required plugin is installed."
    )
