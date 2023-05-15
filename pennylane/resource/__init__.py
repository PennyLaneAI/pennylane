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
The ``resource`` module provides classes and functionality to estimate the quantum resources
(number of qubits, circuit depth, etc.) required to implement advanced quantum algorithms.


Expectation Value Functions
---------------------------

.. currentmodule:: pennylane.resource

.. autosummary::
    :toctree: api

    ~estimate_error
    ~estimate_shots

Quantum Phase Estimation Resources
----------------------------------

.. currentmodule:: pennylane.resource

.. autosummary::
    :toctree: api

    ~FirstQuantization
    ~DoubleFactorization

Resource Classes
----------------

.. currentmodule:: pennylane.resource

.. autosummary::
    :toctree: api

    ~Resources
    ~ResourcesOperation

Tracking Resources for Custom Operations
----------------------------------------

We can use the :code:`null.qubit` device with the :code:`qml.Tracker` to track the resources
used in a quantum circuit with custom operations without execution.

.. code-block:: python3

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

    dev = qml.device("null.qubit", wires=[0, 1, 2], shots=100)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RZ(theta, wires=0)
        qml.CNOT(wires=[0,1])
        MyCustomAlgorithm(wires=[1, 2])
        return qml.expval(qml.PauliZ(wires=1))

    x = np.array(1.23, requires_grad=True)

    with qml.Tracker(dev) as tracker:
        circuit(x)

We can examine the resources by accessing the :code:`resources` key:

>>> resources_lst = tracker.history['resources']
>>> print(resources_lst[0])
wires: 3
gates: 7
depth: 5
shots: Shots(None)
gate_types:
{"RZ": 1, "CNOT": 2, "Hadamard": 2, "PauliZ": 2}
gate_sizes:
{1: 5, 2: 2}
"""
from .resource import Resources, ResourcesOperation
from .first_quantization import FirstQuantization
from .second_quantization import DoubleFactorization
from .measurement import estimate_error, estimate_shots
