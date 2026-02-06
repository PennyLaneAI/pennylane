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

To activate and deactivate the new experimental graph-based decomposition system, use the switches
:func:`~pennylane.decomposition.enable_graph` and :func:`~pennylane.decomposition.disable_graph`. Whether the graph-based
decomposition system is currently being used can be queried with :func:`~pennylane.decomposition.enabled_graph`.
By default, this system is disabled.

.. currentmodule:: pennylane.decomposition

.. autosummary::
    :toctree: api

    ~enable_graph
    ~disable_graph
    ~enabled_graph

>>> qp.decomposition.enabled_graph()
False
>>> qp.decomposition.enable_graph()
>>> qp.decomposition.enabled_graph()
True
>>> qp.decomposition.disable_graph()
>>> qp.decomposition.enabled_graph()
False

.. _decomps_rules:

Defining Decomposition Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.decomposition

.. autosummary::
    :toctree: api

    ~register_resources
    ~register_condition
    ~resource_rep
    ~controlled_resource_rep
    ~adjoint_resource_rep
    ~pow_resource_rep
    ~change_op_basis_resource_rep
    ~DecompositionRule
    ~Resources
    ~CompressedResourceOp
    ~null_decomp

In the new decomposition system, a decomposition rule must be defined as a quantum function that
accepts ``(*op.parameters, op.wires, **op.hyperparameters)`` as arguments, where ``op`` is an
instance of the operator type that the decomposition is for. Additionally, a decomposition rule
must declare its resource requirements using the ``register_resources`` decorator:

.. code-block:: python

    @qp.register_resources({qp.H: 2, qp.CZ: 1})
    def my_cnot(wires):
        qp.H(wires=wires[1])
        qp.CZ(wires=wires)
        qp.H(wires=wires[1])

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
a list of known decomposition rules for a given operator. In the new system, an operator can be
associated with multiple decomposition rules, and the one that leads to the most resource-efficient
decomposition towards a target gate set is chosen.

Integration with the Decompose Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~pennylane.transforms.decompose` transform takes advantage of this new graph-based
decomposition algorithm when :func:`~pennylane.decomposition.enable_graph` is present, and allows for more
flexible decompositions towards any target gate set. For example, the current system does not
guarantee a decomposition to the desired target gate set:

.. code-block:: python

    with qp.queuing.AnnotatedQueue() as q:
        qp.CRX(0.5, wires=[0, 1])

    tape = qp.tape.QuantumScript.from_queue(q)
    [new_tape], _ = qp.transforms.decompose([tape], gate_set={"RX", "RY", "RZ", "CZ"})

>>> new_tape.operations
[RZ(1.5707963267948966, wires=[1]),
    RY(0.25, wires=[1]),
    CNOT(wires=[0, 1]),
    RY(-0.25, wires=[1]),
    CNOT(wires=[0, 1]),
    RZ(-1.5707963267948966, wires=[1])]

With the new system enabled, the transform produces the expected outcome.

>>> qp.decomposition.enable_graph()
>>> [new_tape], _ = qp.transforms.decompose([tape], gate_set={"RX", "RY", "RZ", "CZ"})
>>> new_tape.operations
[RX(0.25, wires=[1]), CZ(wires=[0, 1]), RX(-0.25, wires=[1]), CZ(wires=[0, 1])]

**Customizing Decompositions**

The new system also enables specifying custom decomposition rules. When :func:`~pennylane.decomposition.enable_graph`
is present, the :func:`~pennylane.transforms.decompose` transform accepts two additional keyword
arguments: ``fixed_decomps`` and ``alt_decomps``. The user can define custom decomposition rules
as explained in the :ref:`Defining Decomposition Rules <decomps_rules>` section, and provide them
to the transform via these arguments.

``fixed_decomps`` forces the transform to use the specified decomposition rules for
certain operators, whereas ``alt_decomps`` is used to provide alternative decomposition rules
for operators that may be chosen if they lead to a more resource-efficient decomposition.

In the following example, ``isingxx_decomp`` will always be used to decompose ``qp.IsingXX``
gates; when it comes to ``qp.CNOT``, the system will choose the most efficient decomposition rule
among ``my_cnot1``, ``my_cnot2``, and all existing decomposition rules defined for ``qp.CNOT``.

.. code-block:: python

    qp.decomposition.enable_graph()

    @qp.register_resources({qp.CNOT: 2, qp.RX: 1})
    def isingxx_decomp(phi, wires, **__):
        qp.CNOT(wires=wires)
        qp.RX(phi, wires=[wires[0]])
        qp.CNOT(wires=wires)

    @qp.register_resources({qp.H: 2, qp.CZ: 1})
    def my_cnot1(wires, **__):
        qp.H(wires=wires[1])
        qp.CZ(wires=wires)
        qp.H(wires=wires[1])

    @qp.register_resources({qp.RY: 2, qp.CZ: 1, qp.Z: 2})
    def my_cnot2(wires, **__):
        qp.RY(np.pi/2, wires[1])
        qp.Z(wires[1])
        qp.CZ(wires=wires)
        qp.RY(np.pi/2, wires[1])
        qp.Z(wires[1])

    @qp.transforms.decompose(
        gate_set={"RX", "RZ", "CZ", "GlobalPhase"},
        alt_decomps={qp.CNOT: [my_cnot1, my_cnot2]},
        fixed_decomps={qp.IsingXX: isingxx_decomp},
    )
    @qp.qnode(qp.device("default.qubit"))
    def circuit():
        qp.CNOT(wires=[0, 1])
        qp.IsingXX(0.5, wires=[0, 1])
        return qp.state()

>>> qp.specs(circuit)()["resources"].gate_types
defaultdict(int, {'RZ': 12, 'RX': 7, 'GlobalPhase': 6, 'CZ': 3})

To register alternative decomposition rules under an operator to be used globally, use
:func:`~pennylane.add_decomps`. See :ref:`Inspecting and Managing Decomposition Rules <decomps_management>`
for details.

Graph-based Decomposition Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~DecompositionGraph
    ~DecompGraphSolution

The decomposition graph is a directed graph of operators and decomposition rules. Dijkstra's
algorithm is used to explore the graph and find the most efficient decomposition of a given
operator towards a target gate set.

.. code-block:: python

    op = qp.CRX(0.5, wires=[0, 1])
    graph = DecompositionGraph(
        operations=[op],
        gate_set={"RZ", "RX", "CNOT", "GlobalPhase"},
    )
    solution = graph.solve()

>>> with qp.queuing.AnnotatedQueue() as q:
...     solution.decomposition(op)(0.5, wires=[0, 1])
>>> q.queue
[RZ(1.5707963267948966, wires=[1]),
    RY(0.25, wires=[1]),
    CNOT(wires=[0, 1]),
    RY(-0.25, wires=[1]),
    CNOT(wires=[0, 1]),
    RZ(-1.5707963267948966, wires=[1])]
>>> graph.resource_estimate(op)
<num_gates=10, gate_counts={RZ: 6, CNOT: 2, RX: 2}>

Utility Classes
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~DecompositionError

"""

from pennylane.exceptions import DecompositionError
from .utils import (
    enable_graph,
    disable_graph,
    enabled_graph,
)
from .decomposition_graph import DecompositionGraph, DecompGraphSolution
from .resources import (
    Resources,
    resource_rep,
    controlled_resource_rep,
    adjoint_resource_rep,
    pow_resource_rep,
    CompressedResourceOp,
    change_op_basis_resource_rep,
)
from .decomposition_rule import (
    register_resources,
    register_condition,
    DecompositionRule,
    null_decomp,
    add_decomps,
    list_decomps,
    has_decomp,
)
