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

r"""This module contains the GraphStatePrep template."""

from typing import Optional, Union

import networkx as nx

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires

from .qubit_graph import QubitGraph


class GraphStatePrep(Operation):
    r"""
    Encode a graph state with a single graph operation applied on each qubit, and an entangling
    operation applied on nearest-neighbor qubits defined by the graph connectivity.
    The initial graph is :math:`|0\rangle^{\otimes V}`, given each qubit or graph vertex node
    (:math:`V`) in the graph is in the :math:`|0\rangle` state and is not entangled with any
    other qubit.
    The target graph state :math:`| \psi \rangle` is:
    :math:`| \psi \rangle = \prod\limits_{\{a, b\} \in E} U_{ab}|+\rangle^{\otimes V}`
    where :math:`U_{ab}` is a phase gate applied to the vertices :math:`a`, :math:`b` of a edge
    :math:`E` in the graph as illustrated in eq. (24)
    in `arxiv:quant-ph/0602096 <https://arxiv.org/pdf/quant-ph/0602096>`_.

    The target graph state can be prepared as below:

        1. Each qubit is prepared as :math:`|+\rangle^{\otimes V}` state by applying the
        ``one_qubit_ops`` (:class:`~.pennylane.H` gate) operation.

        2. Entangle every nearest qubit pair in the graph with ``two_qubit_ops``
        (:class:`~.pennylane.CZ` gate) operation.

    Args:
        graph (Union[QubitGraph, networkx.Graph]): QubitGraph or networkx.Graph object mapping qubit to wires.
        one_qubit_ops (Operation): Operator to prepare the initial state of each qubit. Default to :class:`~.pennylane.H`.
        two_qubit_ops (Operation): Operator to entangle nearest qubits. Default to :class:`~.pennylane.CZ`.
        wires (Optional[Wires]): Wires the operator applies on. Wires will be mapped 1:1 to graph nodes. Optional only `graph`
          is a QubitGraph. If no wires are provided, the ``children`` of the provided ``QubitGraph`` will be used as wires.

    .. todo::

        1. To define more complex starting states not relying on a single ops (``one_qubit_ops``
            and ``two_qubit_ops``).
        2. Ensure ``wires`` works with multiple dimensional ``nx.Graph()`` object after the wires
           indexing scheme is added to the ``ftqc`` module.

    **Example:**
        The graph state preparation layer can be customized by the user.

        .. code-block:: python3

            import pennylane as qml
            from pennylane.ftqc import generate_lattice, GraphStatePrep, QubitGraph

            dev = qml.device('default.qubit')

            @qml.qnode(dev)
            def circuit(q, one_qubit_ops, two_qubit_ops, wires = None):
                GraphStatePrep(graph=q, one_qubit_ops=one_qubit_ops, two_qubit_ops=two_qubit_ops, wires = wires)
                return qml.probs()

            lattice = generate_lattice([2, 2], "square")
            q = QubitGraph(lattice.graph, id="square")

            one_qubit_ops = qml.Y
            two_qubit_ops = qml.CNOT

        If the wires argument is not explicitly passed to the circuit, the child nodes of the
        ``QubitGraph`` are used as the wires. The resulting circuit after applying the
        ``GraphStatePrep`` template is:

        >>> print(qml.draw(circuit, level="device")(q, one_qubit_ops, two_qubit_ops))
        QubitGraph<id=(0, 0), loc=[square]>: ──Y─╭●─╭●───────┤  Probs
        QubitGraph<id=(0, 1), loc=[square]>: ──Y─│──╰X─╭●────┤  Probs
        QubitGraph<id=(1, 0), loc=[square]>: ──Y─╰X────│──╭●─┤  Probs
        QubitGraph<id=(1, 1), loc=[square]>: ──Y───────╰X─╰X─┤  Probs

        The circuit wires can also be customized by passing a wires argument to the circuit as follows:

        >>> print(qml.draw(circuit, level="device")(q, one_qubit_ops, two_qubit_ops, wires=[0, 1, 2, 3]))
        0: ──Y─╭●─╭●───────┤  Probs
        1: ──Y─│──╰X─╭●────┤  Probs
        2: ──Y─╰X────│──╭●─┤  Probs
        3: ──Y───────╰X─╰X─┤  Probs

    """

    def __init__(
        self,
        graph: Union[nx.Graph, QubitGraph],
        one_qubit_ops: Operation = qml.H,
        two_qubit_ops: Operation = qml.CZ,
        wires: Optional[Wires] = None,
    ):
        self.hyperparameters["graph"] = graph
        self.hyperparameters["one_qubit_ops"] = one_qubit_ops
        self.hyperparameters["two_qubit_ops"] = two_qubit_ops

        if isinstance(graph, QubitGraph):
            if wires is not None and len(set(wires)) != len(set(graph.node_labels)):
                raise ValueError(
                    "Please ensure the length of wires objects match the number of children in QubitGraph."
                )
            super().__init__(wires=wires if wires is not None else list(graph.children))
        else:
            if wires is None:
                raise ValueError("Please ensure wires is specified.")
            if len(wires) != len(set(graph.nodes)):
                raise ValueError(
                    "Please ensure the length of wires objects match that of labels in graph."
                )
            super().__init__(wires=wires)

    def label(self, *args, **kwargs) -> str:  # pylint: disable=unused-argument
        r"""Defines how the graph state preparation is represented in diagrams and drawings.

        Args:
            *args (Optional[Union[int, str]]): positional arguments for decimals and base_label.
            **kwargs (Optional[dict]): keyword arguments for cache.

        Returns:
            str: label to use in drawings
        """
        return repr(self)

    def __repr__(self):
        """Method defining the string representation of this class."""
        return f"GraphStatePrep({self.hyperparameters['one_qubit_ops'](wires=0).name}, {self.hyperparameters['two_qubit_ops'].name})"

    @staticmethod
    def compute_decomposition(
        wires: Wires,
        graph: Union[nx.Graph, QubitGraph],
        one_qubit_ops: Operation = qml.H,
        two_qubit_ops: Operation = qml.CZ,
    ):  # pylint: disable=arguments-differ, unused-argument
        r"""Representation of the operator as a product of other operators (static method).

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            wires (Wires): Wires the decomposition applies on. Wires will be mapped 1:1 to graph nodes.
            graph (Union[nx.Graph, QubitGraph]): QubitGraph or nx.Graph object mapping qubit to wires.
            one_qubit_ops (Operation): Operator to prepare the initial state of each qubit. Default to :class:`~.pennylane.H`.
            two_qubit_ops (Operation): Operator to entangle nearest qubits. Default to :class:`~.pennylane.CZ`.

        Returns:
            list[Operator]: decomposition of the operator
        """

        op_list = []

        edges = graph.edge_labels if isinstance(graph, QubitGraph) else graph.edges
        nodes = graph.node_labels if isinstance(graph, QubitGraph) else graph.nodes

        if set(wires) != set(nodes):
            wire_map = dict(zip(nodes, wires))
            edges = [(wire_map[edge[0]], wire_map[edge[1]]) for edge in edges]

        for wire in wires:
            op_list.append(one_qubit_ops(wires=wire))
        for edge in edges:
            op_list.append(two_qubit_ops(wires=edge))
        return op_list
