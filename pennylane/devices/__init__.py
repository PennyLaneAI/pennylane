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
This subpackage provides default devices for PennyLane, which do not need external plugins to be installed.
The default devices provide basic built-in qubit
and CV circuit simulators that can be used with PennyLane without the need for additional
dependencies. They may also be used in the PennyLane test suite in order
to verify and test quantum gradient computations.



.. currentmodule:: pennylane.devices
.. autosummary::
    :toctree: api


    default_qubit
    default_qubit_legacy
    default_qubit_jax
    default_qubit_torch
    default_qubit_tf
    default_qubit_autograd
    default_gaussian
    default_mixed
    default_qutrit
    default_qutrit_mixed
    default_clifford
    null_qubit
    tests

Next generation devices
-----------------------

:class:`pennylane.devices.Device` is the latest interface for the next generation of devices that
replaces :class:`pennylane.Device` and :class:`pennylane.QubitDevice`.

While the previous interface :class:`pennylane.Device` is imported top level, the new :class:`pennylane.devices.Device` is
accessible from the ``pennylane.devices`` submodule.

.. currentmodule:: pennylane.devices
.. autosummary::
    :toctree: api

    ExecutionConfig
    Device
    DefaultQubit
    NullQubit
    DefaultQutritMixed

Preprocessing Transforms
------------------------

The ``preprocess`` module offers several transforms that can be used in constructing the :meth:`~.devices.Device.preprocess`
method for devices.

.. currentmodule:: pennylane.devices.preprocess
.. autosummary::
    :toctree: api

    decompose
    validate_observables
    validate_measurements
    validate_device_wires
    validate_multiprocessing_workers
    validate_adjoint_trainable_params
    no_sampling

Other transforms that may be relevant to device preprocessing include:

.. currentmodule:: pennylane
.. autosummary::
    :toctree: api

    defer_measurements
    transforms.broadcast_expand
    transforms.sum_expand
    transforms.split_non_commuting
    transforms.hamiltonian_expand


Modifiers
---------

The ``modifiers`` allow for the easy addition of default behaviour to a device.

.. currentmodule:: pennylane.devices.modifiers
.. autosummary::
    :toctree: api

    single_tape_support
    simulator_tracking

For example with a custom device we can add simulator-style tracking and the ability
to handle a single circuit. See the documentation for each modifier for more details.

.. code-block:: python

    @simulator_tracking
    @single_tape_support
    class MyDevice(qml.devices.Device):

        def execute(self, circuits, execution_config = qml.devices.DefaultExecutionConfig):
            return tuple(0.0 for _ in circuits)

>>> dev = MyDevice()
>>> tape = qml.tape.QuantumTape([qml.S(0)], [qml.expval(qml.X(0))])
>>> with dev.tracker:
...     out = dev.execute(tape)
>>> out
0.0
>>> dev.tracker.history
{'batches': [1],
 'simulations': [1],
 'executions': [1],
 'results': [0.0],
 'resources': [Resources(num_wires=1, num_gates=1,
 gate_types=defaultdict(<class 'int'>, {'S': 1}),
 gate_sizes=defaultdict(<class 'int'>, {1: 1}), depth=1,
 shots=Shots(total_shots=None, shot_vector=()))]}


Qubit Simulation Tools
----------------------

.. currentmodule:: pennylane.devices.qubit
.. automodule:: pennylane.devices.qubit


Qutrit Mixed-State Simulation Tools
-----------------------------------

.. currentmodule:: pennylane.devices.qutrit_mixed
.. automodule:: pennylane.devices.qutrit_mixed

"""

from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .device_api import Device
from .default_qubit import DefaultQubit

# DefaultQubitTF and DefaultQubitAutograd not imported here since this
# would lead to an automatic import of tensorflow and autograd, which are
# not PennyLane core dependencies
from .default_qubit_legacy import DefaultQubitLegacy
from .default_gaussian import DefaultGaussian
from .default_mixed import DefaultMixed
from .default_clifford import DefaultClifford
from .null_qubit import NullQubit
from .default_qutrit_mixed import DefaultQutritMixed
