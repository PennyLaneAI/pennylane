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
Unit tests for the QROMStatePreparation template.
"""
import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.state_preparations.qrom_state_prep import _float_to_binary


@pytest.mark.parametrize(
    ("val", "num_bits", "expected"),
    [
        (0.25, 4, "0100"),
        (0.5, 3, "100"),
        (0.125, 5, "00100"),
    ],
)
def test_float_to_binary(val, num_bits, expected):
    """Test _float_to_binary private function"""

    output = _float_to_binary(val, num_bits)
    assert output == expected


class TestQROMStatePreparation:

    @pytest.mark.jax
    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""

        state = np.random.rand(2**4)
        state /= np.linalg.norm(state)

        wires = qml.registers({"work_wires": 3, "precision_wires": 3, "embedding_wires": 4})

        op = qml.QROMStatePreparation(
            state_vector=state,
            wires=wires["embedding_wires"],
            precision_wires=wires["precision_wires"],
            work_wires=wires["work_wires"],
        )

        qml.ops.functions.assert_valid(op, skip_differentiation=True)

    @pytest.mark.parametrize(
        ("state", "msg_match"),
        [
            (
                np.array([1.0, 0, 0]),
                "State vectors must be of length",
            ),
            (
                np.array([1.0, 1, 0, 0]),
                "Input state vectors must have",
            ),
        ],
    )
    def test_QROMStatePrep_error(self, state, msg_match):
        """Test that proper errors are raised for QROMStatePreparation"""
        with pytest.raises(ValueError, match=msg_match):
            qml.QROMStatePreparation(state, wires=[0, 1], precision_wires=[2, 3], work_wires=[])

    @pytest.mark.parametrize(
        ("state", "num_wires", "num_work_wires", "num_precision_wires"),
        [
            (np.array([0, 0, 0, 1]), 2, 5, 5),
            (np.array([0, 0, 0, 1 / 2, 0, 1 / 2, 1 / 2, -1j / 2]), 3, 5, 5),
            (np.array([-0.84223628, -0.40036496, 0.08974619, -0.34970212]), 2, 1, 6),
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
                1,
                7,
            ),
        ],
    )
    def test_correctness(self, state, num_wires, num_work_wires, num_precision_wires):

        wires = qml.registers(
            {"work": num_work_wires, "precision": num_precision_wires, "state": num_wires}
        )

        dev = qml.device("default.qubit", wires=num_work_wires + num_precision_wires + num_wires)

        qs = qml.tape.QuantumScript(
            [
                qml.QROMStatePreparation(
                    state,
                    wires=wires["state"],
                    work_wires=wires["work"],
                    precision_wires=wires["precision"],
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0][: 2**num_wires]

        assert np.allclose(state, output, atol=0.05)

    def test_decomposition(self):
        """Test that the correct gates are added in the decomposition"""

        wires = qml.registers({"work": 3, "precision": 3, "state": 2})

        decomposition = qml.QROMStatePreparation.compute_decomposition(
            np.array([1 / 2, 1j / 2, -1 / 2, -1j / 2]),
            wires=range(8),
            input_wires=wires["state"],
            work_wires=wires["work"],
            precision_wires=wires["precision"],
        )

        for gate in decomposition:
            assert gate.name in ["QROM", "Adjoint(QROM)", "CRY", "C(GlobalPhase)"]

    @pytest.mark.jax
    def test_interface_jax(self):
        """Test QROMStatePreparation works with jax"""

        from jax import numpy as jnp

        state = [1 / 2, -1 / 2, 1j / 2, -1j / 2]

        wires = qml.registers({"work": 2, "precision": 2, "state": 2})
        dev = qml.device("default.qubit", wires=6)

        qs = qml.tape.QuantumScript(
            [
                qml.QROMStatePreparation(
                    jnp.array(state),
                    wires=wires["state"],
                    work_wires=wires["work"],
                    precision_wires=wires["precision"],
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output_jax = dev.execute(tape[0])[0]

        qs = qml.tape.QuantumScript(
            [
                qml.QROMStatePreparation(
                    state,
                    wires=wires["state"],
                    work_wires=wires["work"],
                    precision_wires=wires["precision"],
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
        """Test QROMStatePreparation works with torch"""

        import torch

        state = [1 / 2, -1 / 2, 1j / 2, -1j / 2]

        wires = qml.registers({"work": 2, "precision": 2, "state": 2})
        dev = qml.device("default.qubit", wires=6)

        qs = qml.tape.QuantumScript(
            [
                qml.QROMStatePreparation(
                    torch.tensor(state, dtype=torch.complex64),
                    wires=wires["state"],
                    work_wires=wires["work"],
                    precision_wires=wires["precision"],
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output_torch = dev.execute(tape[0])[0]

        qs = qml.tape.QuantumScript(
            [
                qml.QROMStatePreparation(
                    state,
                    wires=wires["state"],
                    work_wires=wires["work"],
                    precision_wires=wires["precision"],
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
        """Test QROMStatePreparation works with tensorflow"""

        import tensorflow as tf

        state = [1 / 2, -1 / 2, 1 / 2, -1 / 2]

        wires = qml.registers({"work": 2, "precision": 2, "state": 2})
        dev = qml.device("default.qubit", wires=6)

        qs = qml.tape.QuantumScript(
            [
                qml.QROMStatePreparation(
                    tf.Variable(state),
                    wires=wires["state"],
                    work_wires=wires["work"],
                    precision_wires=wires["precision"],
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output_tf = dev.execute(tape[0])[0]

        qs = qml.tape.QuantumScript(
            [
                qml.QROMStatePreparation(
                    state,
                    wires=wires["state"],
                    work_wires=wires["work"],
                    precision_wires=wires["precision"],
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert qml.math.allclose(output, output_tf)
