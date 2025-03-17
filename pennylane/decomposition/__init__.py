# Copyright 2025 Xanadu Quantum Technologies Inc.

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
.. currentmodule:: pennylane

This module implements the infrastructure for PennyLane's new graph-based decomposition system.

.. warning::

    This module is experimental and is subject to change in the future.

.. currentmodule:: pennylane.decomposition

.. autosummary::
    :toctree: api

    ~register_resources
    ~resource_rep
    ~controlled_resource_rep
    ~adjoint_resource_rep
    ~add_decomps
    ~list_decomps
    ~has_decomp
    ~Resources
    ~CompressedResourceOp
    ~DecompositionRule

.. TODO::
    Add section here explaining how to enable and disable the new decomposition system. [sc-83993]

**Defining Decomposition Rules**

In the new decomposition system, a decomposition rule must be defined as a quantum function that
accepts ``(*op.parameters, op.wires, **op.hyperparameters)`` as arguments, where ``op`` is an
instance of the operator type that the decomposition is for. Additionally, a resource estimate
in the form of a gate count must be registered with this function:

.. code-block:: python

    import pennylane as qml

    @qml.register_resources({qml.H: 2, qml.CZ: 1})
    def my_cnot(wires):
        qml.H(wires=wires[1])
        qml.CZ(wires=wires)
        qml.H(wires=wires[1])

.. code-block:: pycon

    >>> with qml.queuing.AnnotatedQueue() as q:
    ...     my_cnot(wires=[0, 1])
    >>> q.queue
    [H(1), CZ(wires=[0, 1]), H(1)]
    >>> my_cnot.compute_resources()
    num_gates=3, gate_counts={H: 2, CZ: 1}

**Resource Functions and Dynamic Resource Requirements**

In many cases, the resource requirement of an operator's decomposition is not static. For example,
the number of gates in the decomposition for ``qml.MultiRZ`` varies based on the number of wires
it acts on. The set of parameters that affects the resource requirements of an operator is defined
in the operator's ``resource_keys`` attribute:

.. code-block:: pycon

    >>> qml.CNOT.resource_keys
    {}
    >>> qml.MultiRZ.resource_keys
    {'num_wires'}

For operators with dynamic resource requirements such as ``qml.MultiRZ``, their decompositions
must be registered with a resource function (as opposed to a static dictionary) that accepts those
exact arguments and returns a dictionary:

.. code-block:: python

    def _multi_rz_resources(num_wires):
        return {
            qml.CNOT: 2 * (num_wires - 1),
            qml.RZ: 1
        }

    @qml.register_resources(_multi_rz_resources)
    def multi_rz_decomposition(theta, wires, **__):
        for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
            qml.CNOT(wires=(w0, w1))
        qml.RZ(theta, wires=wires[0])
        for w0, w1 in zip(wires[1:], wires[:-1]):
            qml.CNOT(wires=(w0, w1))

.. code-block:: pycon

    >>> multi_rz_decomposition.compute_resources(num_wires=3)
    num_gates=5, gate_counts={CNOT: 4, RZ: 1}

Additionally, when a decomposition rule contains an operator with dynamic resource requirements,
the existence of such an operator must be declared along with the required information:

.. code-block:: python

    def _my_resource(num_wires):
        return {
            qml.resource_rep(qml.MultiRZ, num_wires=3): 2,
            qml.resource_rep(qml.MultiRZ, num_wires=num_wires - 1): 1
        }

    def my_decomp(thata, wires):
        qml.MultiRZ(theta, wires=wires[:3])
        qml.MultiRZ(theta, wires=wires[1:])
        qml.MultiRZ(theta, wires=wires[:3])

where ``qml.resource_rep`` is a utility function that wraps an operator type and any additional
information relevant to its resource estimate into a compressed data structure.

.. TODO::
    Add section here explaining the decomposition graph [sc-84329]

.. TODO::
    Add section here explaining how this new system interacts with the `decompose` transform. [sc-83993]

"""

from .decomposition_graph import DecompositionGraph
from .resources import (
    Resources,
    CompressedResourceOp,
    resource_rep,
    controlled_resource_rep,
    adjoint_resource_rep,
)
from .decomposition_rule import (
    register_resources,
    DecompositionRule,
    add_decomps,
    list_decomps,
    has_decomp,
)
