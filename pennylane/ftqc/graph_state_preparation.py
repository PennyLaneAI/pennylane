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

r"""This module contains the GraphStatePreparation template."""

import pennylane as qml
from pennylane.operation import Operation

from .lattice import Lattice
from .qubit_graph import QubitGraph


class GraphStatePrep(Operation):
    r"""
    Encode a graph state with the specified lattice structure, operations on each qubit, entanglement operations for nearest qubits and qubit graph.
    The initial graph is :math:`|0\rangle^{V}`, given each qubit in the graph is in the :math:`|0\rangle` state and is not entangled with each other.
    The target graph state :math:`| \psi \rangle = \prod\limits_{\{a, b\} \in E} U_{ab}|+\rangle^{V}` is prepared as below:
    1. Each qubit is prepared as :math:`|+\rangle^{V}` state by applying the `qubit_ops` (`Hadamard` gate) operation.
    2. Entangle every nearest qubit pair in the graph with `entanglement_ops` (`CZ` gate) operation.

    Args:
        lattice (Lattice): Graph representation of qubits connectivity.
        wires (QubitGraph): QubitGraph object mapping qubit to wires.
        qubit_ops (Operations): Operator to prepare the initial state of each qubit. Default as ``qml.H``.
        entanglement_ops (Operations): Operator to entangle nearest qubits. Default as ``qml.CZ``.
    """

    def __init__(
        self,
        lattice: Lattice,
        wires: list[QubitGraph],
        qubit_ops: Operation = qml.H,
        entanglement_ops: Operation = qml.CZ,
    ):
        if len(lattice.nodes) != len(wires):
            raise ValueError(
                f"The number of vertices in the lattice is : {len(lattice.get_nodes)} and does not match the number of wires {len(wires)}"
            )

        self.hyperparameters["lattice"] = lattice
        self.hyperparameters["wires"] = wires
        self.hyperparameters["qubit_ops"] = qubit_ops
        self.hyperparameters["entanglement_ops"] = entanglement_ops

    def decomposition(self) -> list["Operator"]:
        r"""Representation of the operator as a product of other operators.

        Returns:
            list[Operator]: decomposition of the operator
        """
        return self.compute_decomposition(**self.hyperparameters)

    @staticmethod
    def compute_decomposition(
        lattice: Lattice,
        wires: list[QubitGraph],
        qubit_ops: Operation = qml.H,
        entanglement_ops: Operation = qml.CZ,
    ):
        r"""Representation of the operator as a product of other operators (static method).
        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            lattice (Lattice): An instance of Lattice represents the connectivity of qubits.
            wires (list[QubitGraph]): QubitGraph object mapping qubit to wires.
            qubit_ops (Operations): Operator to prepare the initial state of each qubit. Default as ``qml.H``.
            entanglement_ops (Operations): Operator to entangle nearest qubits. Default as ``qml.CZ``.

        Returns:
            list[Operator]: decomposition of the operator
        """

        op_list = []
        # Add qubit_ops to the queue
        # traverse the nodes in the qubit graph
        for v in lattice.graph:
            op_list.append(qubit_ops(wires[v]))

        # Add entanglement_ops to the queue
        # traverse the edges in the qubit graph
        for v0, v1 in lattice.edges:
            op_list.append(entanglement_ops(wires[v0], wires[v1]))
        return op_list
