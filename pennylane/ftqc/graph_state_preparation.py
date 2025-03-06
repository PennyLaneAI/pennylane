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

from typing import Union

import networkx as nx

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires

from .qubit_graph import QubitGraph


class GraphStatePrep(Operation):
    r"""
    Encode a graph state with the specified lattice structure, operations on each qubit, entanglement operations for nearest qubits and qubit graph.
    The initial graph is :math:`|0\rangle^{V}`, given each qubit in the graph is in the :math:`|0\rangle` state and is not entangled with each other.
    The target graph state :math:`| \psi \rangle = \prod\limits_{\{a, b\} \in E} U_{ab}|+\rangle^{V}` is prepared as below:
    1. Each qubit is prepared as :math:`|+\rangle^{V}` state by applying the `qubit_ops` (`Hadamard` gate) operation.
    2. Entangle every nearest qubit pair in the graph with `entanglement_ops` (`CZ` gate) operation.

    Args:
        graph (Union[QubitGraph, nx.Graph]): QubitGraph or nx.Graph object mapping qubit to wires.
        qubit_ops (Operation): Operator to prepare the initial state of each qubit. Default as ``qml.H``.
        entanglement_ops (Operation): Operator to entangle nearest qubits. Default as ``qml.CZ``.
        wires: Wires the graph state preparation to apply on. Default as None.

    **Example:**
        The graph state preparation layer can be customized by the user.

        .. code-block:: python3

            import pennylane as qml
            from pennylane.ftqc import generate_lattice, GraphStatePrep, QubitGraph

            dev = qml.device('default.qubit')

            @qml.qnode(dev)
            def circuit(q, qubit_ops, entangle_ops):
                GraphStatePrep(qubit_graph=q, qubit_ops=qubit_ops, entanglement_ops=entangle_ops)
                return qml.probs()

            lattice = generate_lattice([2, 2], "square")
            q = QubitGraph("square", lattice.graph)

            qubit_ops = qml.Y
            entangle_ops = qml.CNOT

        The resulting circuit after applying the ``GraphStatePrep`` template is:

        >>> print(qml.draw(circuit, level="device")(q, qubit_ops, entangle_ops))
        QubitGraph<square, (0, 0)>: ──Y─╭●─╭●───────┤  Probs
        QubitGraph<square, (0, 1)>: ──Y─│──╰X─╭●────┤  Probs
        QubitGraph<square, (1, 0)>: ──Y─╰X────│──╭●─┤  Probs
        QubitGraph<square, (1, 1)>: ──Y───────╰X─╰X─┤  Probs
    """

    def __init__(
        self,
        graph: Union[nx.Graph, QubitGraph],
        qubit_ops: Operation = qml.H,
        entanglement_ops: Operation = qml.CZ,
        wires: Wires = None,
    ):
        self.hyperparameters["graph"] = graph
        self.hyperparameters["qubit_ops"] = qubit_ops
        self.hyperparameters["entanglement_ops"] = entanglement_ops

        if isinstance(graph, QubitGraph):
            if wires is not None and set(wires) != set(graph.graph):
                raise ValueError("Please ensure wires objects match labels in QubitGraph")
            self.hyperparameters["wires"] = wires if wires is not None else set(graph.graph)
            super().__init__(wires=wires if wires is not None else set(graph.graph))
        else:
            if wires is None:
                raise ValueError("Please ensure wires is specified.")
            if wires is not None and set(wires) != set(graph):
                raise ValueError("Please ensure wires objects match labels in graph")
            self.hyperparameters["wires"] = wires
        super().__init__(wires=self.hyperparameters["wires"])

    def label(
        self, decimals: int = None, base_label: str = None, cache: dict = None
    ):  # pylint: disable=unused-argument
        r"""How the graph state preparation is represented in diagrams and drawings.

        Args:
            decimals: If ``None``, no parameters are included. Else, how to round
                the parameters. Required to match general call signature. Not used.
            base_label: overwrite the non-parameter component of the label.
                Required to match general call signature. Not used.
            cache: dictionary that carries information between label calls in the
                same drawing. Required to match general call signature. Not used.

        Returns:
            str: label to use in drawings
        """
        return f"GraphStatePrep({self.hyperparameters['qubit_ops'](wires=0).name}, {self.hyperparameters['entanglement_ops'].name})"

    def __repr__(self):
        """Method defining the string representation of this class."""
        return f"GraphStatePrep({self.hyperparameters['qubit_ops'](wires=0).name}, {self.hyperparameters['entanglement_ops'].name})"

    def decomposition(self) -> list["Operator"]:
        r"""Representation of the operator as a product of other operators.

        Returns:
            list[Operator]: decomposition of the operator
        """
        return self.compute_decomposition(**self.hyperparameters)

    @staticmethod
    def compute_decomposition(
        wires: Wires,
        graph: Union[nx.Graph, QubitGraph],
        qubit_ops: Operation = qml.H,
        entanglement_ops: Operation = qml.CZ,
    ):  # pylint: disable=arguments-differ, unused-argument
        r"""Representation of the operator as a product of other operators (static method).
        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            wires : Wires the decomposition applies on.
            graph (Union[nx.Graph, QubitGraph]): QubitGraph object mapping qubit to wires.
            qubit_ops (Operation): Operator to prepare the initial state of each qubit. Default as ``qml.H``.
            entanglement_ops (Operation): Operator to entangle nearest qubits. Default as ``qml.CZ``.

        Returns:
            list[Operator]: decomposition of the operator
        """

        op_list = []

        # Add entanglement_ops for each pair of nearest qubits in the graph
        if isinstance(graph, QubitGraph):
            # Add qubit_ops for each qubit in the graph
            for wire in wires:
                op_list.append(qubit_ops(wires=graph[wire]))

            for qubit0, qubit1 in graph.graph.edges:
                op_list.append(entanglement_ops(wires=[graph[qubit0], graph[qubit1]]))
        else:
            # Add qubit_ops for each qubit in the graph
            for wire in wires:
                op_list.append(qubit_ops(wires=wire))

            for qubit0, qubit1 in graph.edges:
                op_list.append(entanglement_ops(wires=[qubit0, qubit1]))
        return op_list
