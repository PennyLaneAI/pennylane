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
"""
Tests for the TemporaryAnd template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


class TestTemporaryAnd:
    """Tests specific to the TemporaryAnd operation"""

    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""

        op = qml.TemporaryAnd(wires=[0, "a", 2], control_values=(1, 0))
        qml.ops.functions.assert_valid(op)

    def test_correctness(self):
        """Tests the correctness of the TemporaryAnd operator.
        This is done by comparing the results with the Toffoli operator
        """

        dev = qml.device("default.qubit", wires=4)

        qs_and = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.TemporaryAnd([0, 1, 2], control_values=[0, 1]),
                qml.CNOT([2, 3]),
                qml.RX(1.2, 3),
                qml.adjoint(qml.TemporaryAnd([0, 1, 2], control_values=[0, 1])),
            ],
            [qml.state()],
        )

        qs_toffoli = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.X(0),
                qml.Toffoli([0, 1, 2]),
                qml.X(0),
                qml.CNOT([2, 3]),
                qml.RX(1.2, 3),
                qml.X(0),
                qml.Toffoli([0, 1, 2]),
                qml.X(0),
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs_and])
        output_and = dev.execute(tape[0])[0]

        tape = program([qs_toffoli])
        output_toffoli = dev.execute(tape[0])[0]
        assert np.allclose(output_toffoli, output_and)

        # Compare the contracted isometries with the third qubit fixed to |0>
        M_and = qml.matrix(qml.TemporaryAnd(wires=[0, 1, 2]))
        M_and_adj = qml.matrix(qml.adjoint(qml.TemporaryAnd(wires=[0, 1, 2])))
        M_toffoli = qml.matrix(qml.Toffoli(wires=[0, 1, 2]))

        # When the third qubit starts in |0>, we only check the odd columns
        iso_and = M_and[:, ::2]
        iso_toffoli = M_toffoli[:, ::2]

        # When the third qubit ends in |0>, we only check the odd rows
        iso_M_and_adj = M_and_adj[::2, :]
        iso_toffoli_adj = M_toffoli[::2, :]

        assert np.allclose(iso_and, iso_toffoli)
        assert np.allclose(iso_M_and_adj, iso_toffoli_adj)

    def test_and_decompositions(self):
        """Tests that TemporaryAnd is decomposed properly."""

        for rule in qml.list_decomps(qml.TemporaryAnd):
            _test_decomposition_rule(qml.TemporaryAnd([0, 1, 2], control_values=(0, 0)), rule)

    def test_compute_matrix(self):

        matrix = qml.TemporaryAnd([0, 1, "v"]).compute_matrix(control_values=(1, 1))
        matrix_target = qml.math.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, -1j, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, -1j, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1j, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1j],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )

        assert np.allclose(matrix, matrix_target)

    @pytest.mark.jax
    def test_jax_jit(self):
        """Tests that TemporaryAnd works with jax and jit"""
        import jax

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.TemporaryAnd(wires=[0, 1, 2], control_values=[0, 1])
            qml.CNOT(wires=[2, 3])
            qml.RY(1.2, 3)
            qml.adjoint(qml.TemporaryAnd([0, 1, 2]))
            return qml.probs([0, 1, 2, 3])

        jit_circuit = jax.jit(circuit)

        assert qml.math.allclose(circuit(), jit_circuit())
