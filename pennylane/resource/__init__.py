# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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

Circuit Specifications (specs)
------------------------------

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~specs

Circuit Specification Classes and Utilities
-------------------------------------------

.. currentmodule:: pennylane.resource

.. autosummary::
    :toctree: api

    ~CircuitSpecs
    ~SpecsResources

    ~resources_from_tape

Error Tracking
--------------

.. currentmodule:: pennylane.resource

.. autosummary::
    :toctree: api

    ~AlgorithmicError
    ~SpectralNormError
    ~ErrorOperation
    ~algo_error

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

We can use the :mod:`null.qubit <pennylane.devices.null_qubit>` device with :class:`pennylane.Tracker`
to track the resources used in a quantum circuit with custom operations without execution.

.. code-block:: python

    from pennylane import numpy as pnp
    from pennylane.resource import Resources, ResourcesOperation

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

    @qml.set_shots(shots=100)
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
    Total wire allocations: 3
    Total gates: 7
    Circuit depth: 5
    <BLANKLINE>
    Gate types:
      RZ: 1
      CNOT: 2
      Hadamard: 2
      PauliZ: 2
    <BLANKLINE>
    Measurements:
      expval(PauliZ): 1
"""
from .error import AlgorithmicError, ErrorOperation, SpectralNormError, algo_error
from .resource import (
    Resources,
    ResourcesOperation,
    SpecsResources,
    CircuitSpecs,
    add_in_series,
    add_in_parallel,
    mul_in_series,
    mul_in_parallel,
    resources_from_tape,
    substitute,
)
from .specs import specs
