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
    Encode a graph state with single-qubit operations applied on each qubit, and entangling operations applied on nearest-neighbor qubits defined by the graph connectivity.
    The initial graph is :math:`|0\rangle^{\otimes V}`, given each qubit or graph vertex node (:math:`V`) in the graph is in the :math:`|0\rangle` state and is not entangled with any other qubit.
    The target graph state :math:`| \psi \rangle` is:
    :math:`| \psi \rangle = \prod\limits_{\{a, b\} \in E} U_{ab}|+\rangle^{\otimes V}`
    where :math:`U_{ab}` is a phase gate applied to the vertices :math:`a`, :math:`b` of a edge :math:`E` in the graph as illustrated in eq. (24)
    in `arxiv:quant-ph/0602096 <https://arxiv.org/pdf/quant-ph/0602096>`_.

    The target graph state can be prepared as below:

        1. Each qubit is prepared as :math:`|+\rangle^{\otimes V}` state by applying the ``one_qubit_ops`` (``Hadamard`` gate) operation.
        2. Entangle every nearest qubit pair in the graph with ``two_qubit_ops`` (``CZ`` gate) operation.

    Args:
        graph (Union[QubitGraph, nx.Graph]): QubitGraph or nx.Graph object mapping qubit to wires.
        one_qubit_ops (Operation): Operator to prepare the initial state of each qubit. Default to :class:`~.pennylane.H`. #TODO: To define more complex starting states not relying on a single ops.
        two_qubit_ops (Operation): Operator to entangle nearest qubits. Default to :class:`~.pennylane.CZ`.
        wires (Optional[Wires]): Wires the graph state preparation to apply on. Default to None. #TODO: Ensure wires works with multiple dimensional nx.Graph() object after the wires indexing scheme is added to the ``ftqc`` module.

    **Example:**
        The graph state preparation layer can be customized by the user.

        .. code-block:: python3

            import pennylane as qml
            from pennylane.ftqc import generate_lattice, GraphStatePrep, QubitGraph

            dev = qml.device('default.qubit')

            @qml.qnode(dev)
            def circuit(q, one_qubit_ops, two_qubit_ops):
                GraphStatePrep(qubit_graph=q, one_qubit_ops=one_qubit_ops, two_qubit_ops=two_qubit_ops)
                return qml.probs()

            lattice = generate_lattice([2, 2], "square")
            q = QubitGraph(lattice.graph, id="square")

            one_qubit_ops = qml.Y
            two_qubit_ops = qml.CNOT

        The resulting circuit after applying the ``GraphStatePrep`` template is:

        >>> print(qml.draw(circuit, level="device")(q, one_qubit_ops, two_qubit_ops))
        QubitGraph<id=(0, 1), loc=[square]>: ──Y────╭X─╭●────┤  Probs
        QubitGraph<id=(1, 0), loc=[square]>: ──Y─╭X─│──│──╭●─┤  Probs
        QubitGraph<id=(1, 1), loc=[square]>: ──Y─│──│──╰X─╰X─┤  Probs
        QubitGraph<id=(0, 0), loc=[square]>: ──Y─╰●─╰●───────┤  Probs
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
            if wires is not None and set(wires) != set(graph.node_labels):
                raise ValueError("Please ensure wires objects match labels in QubitGraph")
            super().__init__(wires=wires if wires is not None else set(graph.graph))
        else:
            if wires is None:
                raise ValueError("Please ensure wires is specified.")
            if wires is not None and set(wires) != set(graph.nodes):
                raise ValueError("Please ensure wires objects match labels in graph")
            super().__init__(wires=wires)

    def label(self) -> str:  # pylint: disable=arguments-differ
        r"""Defines how the graph state preparation is represented in diagrams and drawings.

        Returns:
            str: label to use in drawings
        """
        return self.__repr__()

    def __repr__(self):
        """Method defining the string representation of this class."""
        return f"GraphStatePrep({self.hyperparameters['one_qubit_ops'](wires=0).name}, {self.hyperparameters['two_qubit_ops'].name})"

    def decomposition(self) -> list["Operator"]:
        r"""Representation of the operator as a product of other operators.

        Returns:
            list[Operator]: decomposition of the operator
        """
        return self.compute_decomposition(wires=self.wires, **self.hyperparameters)

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
            wires (Wires): Wires the decomposition applies on.
            graph (Union[nx.Graph, QubitGraph]): QubitGraph or nx.Graph object mapping qubit to wires.
            one_qubit_ops (Operation): Operator to prepare the initial state of each qubit. Default to :class:`~.pennylane.H`.
            two_qubit_ops (Operation): Operator to entangle nearest qubits. Default to :class:`~.pennylane.CZ`.

        Returns:
            list[Operator]: decomposition of the operator
        """

        op_list = []

        # Add two_qubit_ops for each pair of nearest qubits in the graph
        if isinstance(graph, QubitGraph):
            # Add one_qubit_ops for each qubit in the graph
            for wire in wires:
                op_list.append(one_qubit_ops(wires=graph[wire]))

            for qubit0, qubit1 in graph.graph.edges:
                op_list.append(two_qubit_ops(wires=[graph[qubit0], graph[qubit1]]))
        else:
            # Add one_qubit_ops for each qubit in the graph
            for wire in wires:
                op_list.append(one_qubit_ops(wires=wire))

            for qubit0, qubit1 in graph.edges:
                op_list.append(two_qubit_ops(wires=[qubit0, qubit1]))
        return op_list
