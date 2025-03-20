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

To activate and deactivate the new experimental graph-based decomposition
system, use the switches ``qml.decomposition.enable_graph`` and
``qml.decomposition.disable_graph``.

Whether or not the graph-based decomposition is currently being used can be
queried with ``qml.decomposition.enabled_graph``.
By default, the mechanism is disabled.

.. currentmodule:: pennylane.decomposition

.. autosummary::
    :toctree: api

    ~enable_graph
    ~disable_graph
    ~enabled_graph

.. code-block:: pycon

    >>> import pennylane as qml
    >>> qml.decomposition.enabled_garph()
    False
    >>> qml.decomposition.enable_graph()
    >>> qml.decomposition.enabled_garph()
    True
    >>> qml.decomposition.disable_graph()
    >>> qml.decomposition.enabled_garph()
    False

.. _decomps_rules:

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

.. _decomps_management:

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

The graph decompostiion system can be used in conjunction with the decompose transform ``qml.transforms.decompose``.
This new way of performing decomposition is generally more resource-efficient and allows for specifying custom decompositions.

The new system can handle more complex decompositions pathways and is more flexible. For example,
in the old system when ``qml.decomposition.disable_graph``, if ``qml.CZ`` is in the target gate set instead of ``qml.CNOT``,
you won't be able to decompose a ``qml.CRX`` to the desired target gate set, but with ``qml.decomposition.enable_graph``,
you do get the expected result.

**Custom Decompositions**

Additionally, the ``fixed_decomps` and ``alt_decomps`` arguments are functional with the new system enabled.
These two keyword arguments allow for customizing how gates get decomposed; decompositions in ``alt_decomps``
are optional for the algorithm to choose if it's the most resource-efficient, and decompositions in
``fixed_decomps`` force the algorithm to choose those decompositions, regardless of their efficiency.

Creating custom decompositions that the system can use involves a quantum function that represents
the decomposition and a resource data structure that tracks gate counts in the custom decomposition.

See also :ref:`Defining Decomposition Rules <decomps_rules>` section.

In the following example, we define custom decompositions for both ``qml.CNOT`` and ``qml.IsingXX``,
and we use the ``qml.transforms.decompose`` transform to decompose a quantum circuit to the target gate set
``{"RZ", "RX", "CZ", "GlobalPhase"}``.

We use ``fixed_decomps`` to force the system to use the custom decompositions for ``qml.IsingXX``
and ``alt_decomps`` to provide alternative decompositions for the ``qml.CNOT`` gate.

.. code-block:: python

    import pennylane as qml

    qml.decomposition.enable_graph()

    @qml.register_resources({qml.CNOT: 2, qml.RX: 1})
    def isingxx_decomp(phi, wires, **__):
        qml.CNOT(wires=wires)
        qml.RX(phi, wires=[wires[0]])
        qml.CNOT(wires=wires)

    @qml.register_resources({qml.H: 2, qml.CZ: 1})
    def my_cnot1(wires, **__):
        qml.H(wires=wires[1])
        qml.CZ(wires=wires)
        qml.H(wires=wires[1])

    @qml.register_resources({qml.RY: 2, qml.CZ: 1, qml.Z: 2})
    def my_cnot2(wires, **__):
        qml.RY(np.pi/2, wires[1])
        qml.Z(wires[1])
        qml.CZ(wires=wires)
        qml.RY(np.pi/2, wires[1])
        qml.Z(wires[1])

    @partial(
        qml.transforms.decompose,
        gate_set={"RX", "RZ", "CZ", "GlobalPhase"},
        alt_decomps={qml.CNOT: [my_cnot1, my_cnot2]},
        fixed_decomps={qml.IsingXX: isingxx_decomp},
    )
    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        qml.CNOT(wires=[0, 1])
        qml.IsingXX(0.5, wires=[0, 1])
        return qml.state()


.. code-block:: pycon

    >>> qml.specs(circuit)()["resources"].gate_types
    defaultdict(int, {'RZ': 12, 'RX': 7, 'GlobalPhase': 6, 'CZ': 3})


Alternative decompositions for the system to choose can also be specified globally with :func:`add_decomps`.
See also :ref:`Inspecting and Managing Decomposition Rules <decomps_management>` section for more information.

"""

from .utils import DecompositionError, enable_graph, disable_graph, enabled_graph

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
