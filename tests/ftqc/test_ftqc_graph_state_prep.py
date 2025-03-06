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
# pylint: disable=no-name-in-module, no-self-use, protected-access
"""Unit tests for the GraphStatePrep module"""

import pytest

import pennylane as qml
import jax
from pennylane.ftqc import GraphStatePrep, QubitGraph, generate_lattice
from pennylane.ops.functions import assert_valid
from pennylane.transforms.decompose import decompose


class TestGraphStatePrep:
    """Test for graph state prep"""

    @pytest.mark.xfail(reason="Jax JIT requires wires to be integers.")
    def test_jaxjit_circuit_graph_state_prep(self):
        """Test if Jax JIT works with GraphStatePrep"""
        lattice = generate_lattice([2, 2], "square")
        q = QubitGraph("test", lattice.graph)
        dev = qml.device("default.qubit")
        @jax.jit
        @qml.qnode(dev)
        def circuit(q):
            GraphStatePrep(qubit_graph=q)
            return qml.probs()

        circuit(q)


    def test_circuit_accept_graph_state_prep(self):
        """Test if a quantum function accepts GraphStatePrep."""
        lattice = generate_lattice([2, 2], "square")
        q = QubitGraph("test", lattice.graph)
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(q):
            GraphStatePrep(qubit_graph=q)
            return qml.probs()

        circuit(q)
        assert_valid(GraphStatePrep(qubit_graph=q), skip_deepcopy=True, skip_pickle=True)

    @pytest.mark.parametrize(
        "dims, shape, expected",
        [
            ([5], "chain", 9),
            ([2, 2], "square", 8),
            ([2, 3], "rectangle", 13),
            ([1, 2], "triangle", 9),
            ([2, 1], "honeycomb", 21),
            ([2, 2, 2], "cubic", 20),
        ],
    )
    def test_compute_decompose(self, dims, shape, expected):
        """Test the compute_decomposition method of the GraphStatePrep class."""
        lattice = generate_lattice(dims, shape)
        q = QubitGraph("test", lattice.graph)
        queue = GraphStatePrep.compute_decomposition(q)
        assert len(queue) == expected

    @pytest.mark.parametrize(
        "qubit_ops, entangle_ops",
        [
            (qml.H, qml.CZ),
            (qml.X, qml.CNOT),
        ],
    )
    def test_decompose(self, qubit_ops, entangle_ops):
        """Test the decomposition method of the GraphStatePrep class."""
        lattice = generate_lattice([2, 2, 2], "cubic")
        q = QubitGraph("test", lattice.graph)
        op = GraphStatePrep(qubit_graph=q, qubit_ops=qubit_ops, entanglement_ops=entangle_ops)
        queue = op.decomposition()
        assert len(queue) == 20  # 8 ops for |0> -> |+> and 12 ops to entangle nearest qubits
        for op in queue[:8]:
            assert op.name == qubit_ops(0).name
            assert isinstance(op.wires[0], QubitGraph)
        for op in queue[8:]:
            assert op.name == entangle_ops.name
            assert all(isinstance(w, QubitGraph) for w in op.wires)

