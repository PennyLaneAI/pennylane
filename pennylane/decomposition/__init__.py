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

.. TODO::
    Add section here explaining how to enable and disable the new decomposition system. [sc-83993]

Defining Decomposition Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.decomposition

.. autosummary::
    :toctree: api

    ~register_resources
    ~resource_rep
    ~controlled_resource_rep
    ~adjoint_resource_rep
    ~DecompositionRule
    ~Resources

In the new decomposition system, a decomposition rule must be defined as a quantum function that
accepts ``(*op.parameters, op.wires, **op.hyperparameters)`` as arguments, where ``op`` is an
instance of the operator type that the decomposition is for. Additionally, a decomposition rule
must declare its resource requirements using the ``register_resources`` decorator:

.. code-block:: python

    import pennylane as qml

    @qml.register_resources({qml.H: 2, qml.CZ: 1})
    def my_cnot(wires):
        qml.H(wires=wires[1])
        qml.CZ(wires=wires)
        qml.H(wires=wires[1])

Inspecting and Managing Decomposition Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~add_decomps
    ~list_decomps
    ~has_decomp

PennyLane maintains a global dictionary of decomposition rules. New decomposition rules can be
registered under an operator using ``add_decomps``, and ``list_decomps`` can be called to inspect
a list of known decomposition rules for a given operator.

Graph-based Decomposition Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~DecompositionGraph

The decomposition graph is a directed graph of operators and decomposition rules. Dijkstra's
algorithm is used to explore the graph and find the most efficient decomposition of a given
operator towards a target gate set.

.. code-block:: python

    op = qml.CRX(0.5, wires=[0, 1])
    graph = DecompositionGraph(
        operations=[op],
        target_gate_set={"RZ", "RX", "CNOT", "GlobalPhase"},
    )
    graph.solve()

.. code-block:: pycon

    >>> with qml.queuing.AnnotatedQueue() as q:
    ...     graph.decomposition(op)(0.5, wires=[0, 1])
    >>> q.queue
    [H(1), CRZ(0.5, wires=Wires([0, 1])), H(1)]
    >>> graph.resource_estimate(op)
    <num_gates=14, gate_counts={RZ: 6, GlobalPhase: 4, RX: 2, CNOT: 2}>

Integration with the Decompose Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. TODO::
    Add section here explaining how this new system integrates with the `decompose` transform. [sc-83993]

"""

from .decomposition_graph import DecompositionGraph
from .resources import (
    Resources,
    # TODO: add CompressedResourceOp once the conflict with labs is resolved.
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
