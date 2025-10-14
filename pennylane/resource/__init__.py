# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
The ``resource`` module provides classes and functionality to track the quantum resources
(number of qubits, circuit depth, etc.) required to implement advanced quantum algorithms.

.. seealso::
    The :mod:`~.estimator` module for higher level resource estimation of quantum programs.

Expectation Value Functions
---------------------------

.. currentmodule:: pennylane.resource

.. autosummary::
    :toctree: api

    ~estimate_error
    ~estimate_shots

Circuit specifications
----------------------

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~specs


Quantum Phase Estimation Resources
----------------------------------

.. currentmodule:: pennylane.resource

.. autosummary::
    :toctree: api

    ~FirstQuantization
    ~DoubleFactorization

Error Tracking
--------------

.. currentmodule:: pennylane.resource

.. autosummary::
    :toctree: api

    ~AlgorithmicError
    ~SpectralNormError
    ~ErrorOperation

Resource Classes
----------------

.. currentmodule:: pennylane.resource

.. autosummary::
    :toctree: api

    ~Resources
    ~ResourcesOperation

Resource Functions
~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.resource

.. autosummary::
    :toctree: api

    ~add_in_series
    ~add_in_parallel
    ~mul_in_series
    ~mul_in_parallel
    ~substitute

Tracking Resources for Custom Operations
----------------------------------------

We can use the :code:`null.qubit` device with the :class:`pennylane.Tracker` to track the resources
used in a quantum circuit with custom operations without execution.

.. code-block:: python

    from functools import partial
    from pennylane import numpy as pnp

    class MyCustomAlgorithm(ResourcesOperation):
        num_wires = 2

        def resources(self):
            return Resources(
                num_wires=self.num_wires,
                num_gates=5,
                gate_types={"Hadamard": 2, "CNOT": 1, "PauliZ": 2},
                gate_sizes={1: 4, 2: 1},
                depth=3,
            )

    dev = qml.device("null.qubit", wires=[0, 1, 2])

    @partial(qml.set_shots, shots=100)
    @qml.qnode(dev)
    def circuit(theta):
        qml.RZ(theta, wires=0)
        qml.CNOT(wires=[0,1])
        MyCustomAlgorithm(wires=[1, 2])
        return qml.expval(qml.Z(1))

    x = pnp.array(1.23, requires_grad=True)

    with qml.Tracker(dev) as tracker:
        circuit(x)

We can examine the resources by accessing the :code:`resources` key:

>>> resources_lst = tracker.history['resources']
>>> print(resources_lst[0])
num_wires: 3
num_gates: 7
depth: 5
shots: Shots(total=100)
gate_types:
{'RZ': 1, 'CNOT': 2, 'Hadamard': 2, 'PauliZ': 2}
gate_sizes:
{1: 5, 2: 2}
"""
from .error import AlgorithmicError, ErrorOperation, SpectralNormError
from .first_quantization import FirstQuantization
from .measurement import estimate_error, estimate_shots
from .resource import (
    Resources,
    ResourcesOperation,
    add_in_series,
    add_in_parallel,
    mul_in_series,
    mul_in_parallel,
    substitute,
)
from .second_quantization import DoubleFactorization
from .specs import specs
