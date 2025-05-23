# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the Elbow template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


class TestElbow:
    """Tests specific to the Elbow operation"""

    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""

        op = qml.Elbow(
            wires=[0, 1, 2],
        )
        qml.ops.functions.assert_valid(op)

    def test_correctness(self):
        """Tests the correctness of the Elbow operator.
        This is done by comparing the results with the Toffoli operator
        """

        dev = qml.device("default.qubit", wires=4)

        qs_elbow = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.Elbow([0, 1, 2]),
                qml.CNOT([2, 3]),
                qml.RX(1.2, 3),
                qml.adjoint(qml.Elbow([0, 1, 2])),
            ],
            [qml.state()],
        )

        qs_toffoli = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.Toffoli([0, 1, 2]),
                qml.CNOT([2, 3]),
                qml.RX(1.2, 3),
                qml.Toffoli([0, 1, 2]),
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs_elbow])
        output_elbow = dev.execute(tape[0])[0]

        tape = program([qs_toffoli])
        output_toffoli = dev.execute(tape[0])[0]
        assert np.allclose(output_toffoli, output_elbow)

    def test_elbow_decompositions(self):
        """Tests that Elbow is decomposed properly."""

        for rule in qml.list_decomps(qml.Elbow):
            _test_decomposition_rule(qml.Elbow([0, 1, 2]), rule)

    @pytest.mark.jax
    def test_jax_jit(self):
        """Tests that Elbow works with jax and jit"""
        import jax

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.Elbow(wires=[0, 1, 2])
            qml.CNOT(wires=[2, 3])
            qml.RY(1.2, 3)
            qml.adjoint(qml.Elbow([0, 1, 2]))
            return qml.probs([0, 1, 2, 3])

        jit_circuit = jax.jit(circuit)

        assert qml.math.allclose(circuit(), jit_circuit())
