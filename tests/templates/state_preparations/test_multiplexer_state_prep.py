# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for the MultiplexerStatePreparation template.
"""
import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as np


class TestMultiplexerStatePreparation:

    @pytest.mark.jax
    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""

        state = np.random.rand(2**4) * 1j
        state /= np.linalg.norm(state)

        wires = range(4)

        op = qml.MultiplexerStatePreparation(state_vector=state, wires=wires)

        qml.ops.functions.assert_valid(op, skip_differentiation=True)

    @pytest.mark.parametrize(
        ("state", "msg_match"),
        [
            (
                np.array([1.0, 0, 0]),
                "State vector must be of length",
            ),
            (
                np.array([1.0, 1, 0, 0]),
                "State vector must have",
            ),
        ],
    )
    def test_MultiplexerStatePrep_error(self, state, msg_match):
        """Test that proper errors are raised for MultiplexerStatePreparation"""
        with pytest.raises(ValueError, match=msg_match):
            qml.MultiplexerStatePreparation(state, wires=[0, 1])

    @pytest.mark.parametrize(
        ("state", "num_wires"),
        [
            (np.array([0, 0, 0, 1]), 2),
            (np.array([0, 0, 0, 1 / 2, 0, 1 / 2, 1 / 2, -1j / 2]), 3),
            (np.array([-0.84223628, -0.40036496, 0.08974619, -0.34970212]), 2),
            (
                np.array(
                    [
                        0.17157142 + 0.09585932j,
                        0.00852997 - 0.21056896j,
                        0.12986199 + 0.94654822j,
                        0.04206036 - 0.04873857j,
                    ]
                ),
                2,
            ),
        ],
    )
    def test_correctness(self, state, num_wires):

        wires = range(num_wires)

        dev = qml.device("default.qubit", wires=num_wires)

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexerStatePreparation(
                    state,
                    wires=wires,
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert np.allclose(state, output, atol=0.05)

    def test_decomposition(self):
        """Test that the correct gates are added in the decomposition"""

        wires = range(2)

        decomposition = qml.MultiplexerStatePreparation.compute_decomposition(
            np.array([1 / 2, 1j / 2, -1 / 2, -1j / 2]),
            wires=wires,
        )

        for gate in decomposition:
            assert gate.name in ["SelectPauliRot", "C(GlobalPhase)", "DiagonalQubitUnitary"]

    @pytest.mark.jax
    def test_interface_jax(self):
        """Test MultiplexerStatePreparation works with jax"""

        from qpjax import numpy as jnp

        state = [1 / 2, -1 / 2, 1j / 2, -1j / 2]

        wires = range(2)
        dev = qml.device("default.qubit", wires=2)

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexerStatePreparation(
                    jnp.array(state),
                    wires=wires,
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output_jax = dev.execute(tape[0])[0]

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexerStatePreparation(
                    state,
                    wires=wires,
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert qml.math.allclose(output, output_jax)

    @pytest.mark.torch
    def test_interface_torch(self):
        """Test MultiplexerStatePreparation works with torch"""

        import torch

        state = torch.tensor([1 / 2, -1 / 2, 1j / 2, -1j / 2])

        wires = range(2)
        dev = qml.device("default.qubit", wires=2)

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexerStatePreparation(
                    torch.tensor(state, dtype=torch.complex64),
                    wires=wires,
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output_torch = dev.execute(tape[0])[0]

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexerStatePreparation(
                    state,
                    wires=wires,
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert qml.math.allclose(output, output_torch)

    @pytest.mark.tf
    def test_interface_tf(self):
        """Test MultiplexerStatePreparation works with tensorflow"""

        import tensorflow as tf

        state = tf.Variable([1 / 2, -1 / 2, 1 / 2, -1 / 2])

        wires = range(2)
        dev = qml.device("default.qubit", wires=6)

        qs = qml.tape.QuantumScript(
            [qml.MultiplexerStatePreparation(tf.Variable(state), wires=wires)],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output_tf = dev.execute(tape[0])[0]

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexerStatePreparation(
                    state,
                    wires=wires,
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert qml.math.allclose(output, output_tf)

    @pytest.mark.jax
    def test_jit(self):
        """Tests the template correctly compiles with JAX JIT."""
        import qpjax

        state = qpjax.numpy.array([1 / 2j, -1 / 2, 1 / 2, -1 / 2])

        wires = range(2)
        dev = qml.device("default.qubit", wires=6)

        @qml.qnode(dev)
        def circuit(state):

            for wire in wires:
                qml.Hadamard(wire)

            qml.MultiplexerStatePreparation(
                state,
                wires=wires,
            )
            return qml.probs(wires)

        jit_circuit = qpjax.jit(circuit)

        assert qml.math.allclose(circuit(state), jit_circuit(state))
