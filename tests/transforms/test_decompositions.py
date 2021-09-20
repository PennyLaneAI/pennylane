# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Tests for the QubitUnitary decomposition transforms.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np

from pennylane.wires import Wires
from pennylane.transforms.decompositions import zyz_decomposition

from test_optimization.utils import check_matrix_equivalence
from gate_data import I, Z, S, T, H, X

single_qubit_decomps = [
    # First set of gates are diagonal and converted to RZ
    (I, qml.RZ, [0.0]),
    (Z, qml.RZ, [np.pi]),
    (S, qml.RZ, [np.pi / 2]),
    (T, qml.RZ, [np.pi / 4]),
    (qml.RZ(0.3, wires=0).matrix, qml.RZ, [0.3]),
    (qml.RZ(-0.5, wires=0).matrix, qml.RZ, [-0.5]),
    # This is an off-diagonal gate that gets converted to Rot, but one RZ angle is 0
    (
        np.array([[0, -0.98310193 + 0.18305901j], [0.98310193 + 0.18305901j, 0]]),
        qml.Rot,
        [0, -np.pi, -5.914991017809059],
    ),
    # Next set of gates are non-diagonal and decomposed as Rots
    (H, qml.Rot, [np.pi, np.pi / 2, 0.0]),
    (X, qml.Rot, [0.0, -np.pi, -np.pi]),
    (qml.Rot(0.2, 0.5, -0.3, wires=0).matrix, qml.Rot, [0.2, 0.5, -0.3]),
    (np.exp(1j * 0.02) * qml.Rot(-1.0, 2.0, -3.0, wires=0).matrix, qml.Rot, [-1.0, 2.0, -3.0]),
]


class TestQubitUnitaryZYZDecomposition:
    """Test that the decompositions are correct."""

    def test_zyz_decomposition_invalid_input(self):
        """Test that non-unitary operations throw errors when we try to decompose."""
        with pytest.raises(ValueError, match="Operator must be unitary"):
            zyz_decomposition(I + H, Wires("a"))

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition(self, U, expected_gate, expected_params):
        """Test that a one-qubit matrix in isolation is correctly decomposed."""
        obtained_gates = zyz_decomposition(U, Wires("a"))

        assert len(obtained_gates) == 1
        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")
        assert qml.math.allclose(obtained_gates[0].parameters, expected_params)

        if obtained_gates[0].num_params == 1:
            obtained_mat = qml.RZ(obtained_gates[0].parameters[0], wires=0).matrix
        else:
            obtained_mat = qml.Rot(*obtained_gates[0].parameters, wires=0).matrix

        assert check_matrix_equivalence(obtained_mat, U)

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition_torch(self, U, expected_gate, expected_params):
        """Test that a one-qubit operation in Torch is correctly decomposed."""
        torch = pytest.importorskip("torch")

        U = torch.tensor(U, dtype=torch.complex128)

        obtained_gates = zyz_decomposition(U, wire="a")

        assert len(obtained_gates) == 1
        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")
        assert qml.math.allclose(qml.math.unwrap(obtained_gates[0].parameters), expected_params)

        if obtained_gates[0].num_params == 1:
            obtained_mat = qml.RZ(obtained_gates[0].parameters[0], wires=0).matrix
        else:
            obtained_mat = qml.Rot(*obtained_gates[0].parameters, wires=0).matrix

        assert check_matrix_equivalence(obtained_mat, qml.math.unwrap(U))

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition_tf(self, U, expected_gate, expected_params):
        """Test that a one-qubit operation in Tensorflow is correctly decomposed."""
        tf = pytest.importorskip("tensorflow")

        U = tf.Variable(U, dtype=tf.complex128)

        obtained_gates = zyz_decomposition(U, wire="a")

        assert len(obtained_gates) == 1
        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")
        assert qml.math.allclose(qml.math.unwrap(obtained_gates[0].parameters), expected_params)

        print(qml.math.unwrap(obtained_gates[0].parameters))
        print(expected_params)

        if obtained_gates[0].num_params == 1:
            # With TF and RZ, need to cast since can't just unwrap
            obtained_mat = qml.RZ(obtained_gates[0].parameters[0].numpy(), wires=0).matrix
        else:
            obtained_mat = qml.Rot(*qml.math.unwrap(obtained_gates[0].parameters), wires=0).matrix

        assert check_matrix_equivalence(obtained_mat, U)

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition_jax(self, U, expected_gate, expected_params):
        """Test that a one-qubit operation in JAX is correctly decomposed."""
        jax = pytest.importorskip("jax")

        # Enable float64 support
        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        U = jax.numpy.array(U, dtype=jax.numpy.complex128)

        obtained_gates = zyz_decomposition(U, wire="a")

        assert len(obtained_gates) == 1
        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")
        assert qml.math.allclose(qml.math.unwrap(obtained_gates[0].parameters), expected_params)

        if obtained_gates[0].num_params == 1:
            obtained_mat = qml.RZ(obtained_gates[0].parameters[0], wires=0).matrix
        else:
            obtained_mat = qml.Rot(*obtained_gates[0].parameters, wires=0).matrix

        assert check_matrix_equivalence(obtained_mat, U)
