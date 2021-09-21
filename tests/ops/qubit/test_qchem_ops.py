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
Unit tests for the available qubit operations for quantum chemistry purposes.
"""
import pytest
import numpy as np
from scipy.linalg import expm

import pennylane as qml

from gate_data import (
    X,
    StateZeroProjector,
    StateOneProjector,
    ControlledPhaseShift,
    SingleExcitation,
    SingleExcitationPlus,
    SingleExcitationMinus,
    DoubleExcitation,
    DoubleExcitationPlus,
    DoubleExcitationMinus,
)


class TestDecomposition:
    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    def test_single_excitation_plus_decomp(self, phi):
        """Tests that the SingleExcitationPlus operation calculates the correct decomposition.

        Need to consider the matrix of CRY separately, as the control is wire 1
        and the target is wire 0 in the decomposition. (Not applicable for
        ControlledPhase as it has the same matrix representation regardless of the
        control and target wires.)"""
        decomp = qml.SingleExcitationPlus.decomposition(phi, wires=[0, 1])

        mats = []
        for i in reversed(decomp):
            if i.wires.tolist() == [0]:
                mats.append(np.kron(i.matrix, np.eye(2)))
            elif i.wires.tolist() == [1]:
                mats.append(np.kron(np.eye(2), i.matrix))
            elif i.wires.tolist() == [1, 0] and isinstance(i, qml.CRY):
                new_mat = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(phi / 2), 0, -np.sin(phi / 2)],
                        [0, 0, 1, 0],
                        [0, np.sin(phi / 2), 0, np.cos(phi / 2)],
                    ]
                )

                mats.append(new_mat)
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)
        exp = SingleExcitationPlus(phi)

        assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    def test_single_excitation_minus_decomp(self, phi):
        """Tests that the SingleExcitationMinus operation calculates the correct decomposition.

        Need to consider the matrix of CRY separately, as the control is wire 1
        and the target is wire 0 in the decomposition. (Not applicable for
        ControlledPhase as it has the same matrix representation regardless of the
        control and target wires.)"""
        decomp = qml.SingleExcitationMinus.decomposition(phi, wires=[0, 1])

        mats = []
        for i in reversed(decomp):
            if i.wires.tolist() == [0]:
                mats.append(np.kron(i.matrix, np.eye(2)))
            elif i.wires.tolist() == [1]:
                mats.append(np.kron(np.eye(2), i.matrix))
            elif i.wires.tolist() == [1, 0] and isinstance(i, qml.CRY):
                new_mat = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(phi / 2), 0, -np.sin(phi / 2)],
                        [0, 0, 1, 0],
                        [0, np.sin(phi / 2), 0, np.cos(phi / 2)],
                    ]
                )

                mats.append(new_mat)
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)
        exp = SingleExcitationMinus(phi)

        assert np.allclose(decomposed_matrix, exp)


class TestSingleExcitation:
    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_matrix(self, phi):
        """Tests that the SingleExcitation operation calculates the correct matrix"""
        op = qml.SingleExcitation(phi, wires=[0, 1])
        res = op.matrix
        exp = SingleExcitation(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_decomp(self, phi):
        """Tests that the SingleExcitation operation calculates the correct decomposition.

        Need to consider the matrix of CRY separately, as the control is wire 1
        and the target is wire 0 in the decomposition."""
        decomp = qml.SingleExcitation.decomposition(phi, wires=[0, 1])

        mats = []
        for i in reversed(decomp):
            if i.wires.tolist() == [1, 0] and isinstance(i, qml.CRY):
                new_mat = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(phi / 2), 0, -np.sin(phi / 2)],
                        [0, 0, 1, 0],
                        [0, np.sin(phi / 2), 0, np.cos(phi / 2)],
                    ]
                )
                mats.append(new_mat)
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)
        exp = SingleExcitation(phi)

        assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_generator(self, phi):
        """Tests that the SingleExcitation operation calculates the correct generator"""
        op = qml.SingleExcitation(phi, wires=[0, 1])
        g, a = op.generator
        res = expm(1j * a * g * phi)
        exp = SingleExcitation(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_plus_matrix(self, phi):
        """Tests that the SingleExcitationPlus operation calculates the correct matrix"""
        op = qml.SingleExcitationPlus(phi, wires=[0, 1])
        res = op.matrix
        exp = SingleExcitationPlus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_plus_generator(self, phi):
        """Tests that the SingleExcitationPlus operation calculates the correct generator"""
        op = qml.SingleExcitationPlus(phi, wires=[0, 1])
        g, a = op.generator
        res = expm(1j * a * g * phi)
        exp = SingleExcitationPlus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_minus_matrix(self, phi):
        """Tests that the SingleExcitationMinus operation calculates the correct matrix"""
        op = qml.SingleExcitationMinus(phi, wires=[0, 1])
        res = op.matrix
        exp = SingleExcitationMinus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_minus_generator(self, phi):
        """Tests that the SingleExcitationMinus operation calculates the correct generator"""
        op = qml.SingleExcitationMinus(phi, wires=[0, 1])
        g, a = op.generator
        res = expm(1j * a * g * phi)
        exp = SingleExcitationMinus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize(
        "excitation", [qml.SingleExcitation, qml.SingleExcitationPlus, qml.SingleExcitationMinus]
    )
    def test_autograd(self, excitation):
        """Tests that operations are computed correctly using the
        autograd interface"""

        pytest.importorskip("autograd")
        dev = qml.device("default.qubit.autograd", wires=2)
        state = np.array([0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0])

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            excitation(phi, wires=[0, 1])
            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.SingleExcitation, -0.1),
            (qml.SingleExcitationPlus, 0.2),
            (qml.SingleExcitationMinus, np.pi / 4),
        ],
    )
    def test_autograd_grad(self, diff_method, excitation, phi):
        """Tests that gradients are computed correctly using the
        autograd interface"""

        pytest.importorskip("autograd")
        dev = qml.device("default.qubit.autograd", wires=2)

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            excitation(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(qml.grad(circuit)(phi), np.sin(phi))

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.SingleExcitation, -0.1),
            (qml.SingleExcitationPlus, 0.2),
            (qml.SingleExcitationMinus, np.pi / 4),
        ],
    )
    def test_tf(self, excitation, phi, diff_method):
        """Tests that gradients and operations are computed correctly using the
        tensorflow interface"""

        tf = pytest.importorskip("tensorflow")
        dev = qml.device("default.qubit.tf", wires=2)

        @qml.qnode(dev, interface="tf", diff_method=diff_method)
        def circuit(phi):
            qml.PauliX(wires=0)
            excitation(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi_t = tf.Variable(phi, dtype=tf.float64)
        with tf.GradientTape() as tape:
            res = circuit(phi_t)

        grad = tape.gradient(res, phi_t)
        assert np.allclose(grad, np.sin(phi))

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.SingleExcitation, -0.1),
            (qml.SingleExcitationPlus, 0.2),
            (qml.SingleExcitationMinus, np.pi / 4),
        ],
    )
    def test_jax(self, excitation, phi, diff_method):
        """Tests that gradients and operations are computed correctly using the
        jax interface"""

        if diff_method == "parameter-shift":
            pytest.skip("JAX support for the parameter-shift method is still TBD")

        jax = pytest.importorskip("jax")

        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(phi):
            qml.PauliX(wires=0)
            excitation(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(jax.grad(circuit)(phi), np.sin(phi))


class TestDoubleExcitation:
    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_matrix(self, phi):
        """Tests that the DoubleExcitation operation calculates the correct matrix"""
        op = qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
        res = op.matrix
        exp = DoubleExcitation(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_decomp(self, phi):
        """Tests that the DoubleExcitation operation calculates the correct decomposition"""
        op = qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
        decomp = op.decomposition(phi, wires=[0, 1, 2, 3])

        mats = [m.matrix for m in decomp]
        decomposed_matrix = mats[0] @ mats[1]
        exp = DoubleExcitation(phi)

        assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_generator(self, phi):
        """Tests that the DoubleExcitation operation calculates the correct generator"""
        op = qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
        g, a = op.generator

        res = expm(1j * a * g * phi)
        exp = DoubleExcitation(phi)

        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    def test_double_excitation_decomp(self, phi):
        """Tests that the DoubleExcitation operation calculates the correct decomposition.

        The decomposition has already been expressed in terms of single-qubit rotations
        and CNOTs. For each term in the decomposition we need to construct the appropriate
        four-qubit tensor product matrix and then multiply them together.
        """
        decomp = qml.DoubleExcitation.decomposition(phi, wires=[0, 1, 2, 3])

        from functools import reduce

        # To compute the matrix for CX on an arbitrary number of qubits, use the fact that
        # CU  = |0><0| \otimes I + |1><1| \otimes U
        def cnot_four_qubits(wires):
            proj_0_term = [StateZeroProjector if idx == wires[0] else np.eye(2) for idx in range(4)]

            proj_1_term = [np.eye(2) for idx in range(4)]
            proj_1_term[wires[0]] = StateOneProjector
            proj_1_term[wires[1]] = X

            proj_0_kron = reduce(np.kron, proj_0_term)
            proj_1_kron = reduce(np.kron, proj_1_term)

            return proj_0_kron + proj_1_kron

        # Inserts a single-qubit matrix into a four-qubit matrix at the right place
        def single_mat_four_qubits(mat, wire):
            individual_mats = [mat if idx == wire else np.eye(2) for idx in range(4)]
            return reduce(np.kron, individual_mats)

        mats = []
        for i in reversed(decomp):
            # Single-qubit gate
            if len(i.wires.tolist()) == 1:
                mat = single_mat_four_qubits(i.matrix, i.wires.tolist()[0])
                mats.append(mat)
            # Two-qubit gate
            else:
                mat = cnot_four_qubits(i.wires.tolist())
                mats.append(mat)

        decomposed_matrix = np.linalg.multi_dot(mats)
        exp = DoubleExcitation(phi)

        assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_plus_matrix(self, phi):
        """Tests that the DoubleExcitationPlus operation calculates the correct matrix"""
        op = qml.DoubleExcitationPlus(phi, wires=[0, 1, 2, 3])
        res = op.matrix
        exp = DoubleExcitationPlus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_plus_generator(self, phi):
        """Tests that the DoubleExcitationPlus operation calculates the correct generator"""
        op = qml.DoubleExcitationPlus(phi, wires=[0, 1, 2, 3])
        g, a = op.generator

        res = expm(1j * a * g * phi)
        exp = DoubleExcitationPlus(phi)

        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_minus_matrix(self, phi):
        """Tests that the DoubleExcitationMinus operation calculates the correct matrix"""
        op = qml.DoubleExcitationMinus(phi, wires=[0, 1, 2, 3])
        res = op.matrix
        exp = DoubleExcitationMinus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_minus_generator(self, phi):
        """Tests that the DoubleExcitationMinus operation calculates the correct generator"""
        op = qml.DoubleExcitationMinus(phi, wires=[0, 1, 2, 3])
        g, a = op.generator

        res = expm(1j * a * g * phi)
        exp = DoubleExcitationMinus(phi)

        assert np.allclose(res, exp)

    @pytest.mark.parametrize(
        "excitation", [qml.DoubleExcitation, qml.DoubleExcitationPlus, qml.DoubleExcitationMinus]
    )
    def test_autograd(self, excitation):
        """Tests that operations are computed correctly using the
        autograd interface"""

        pytest.importorskip("autograd")

        dev = qml.device("default.qubit.autograd", wires=4)
        state = np.array(
            [0, 0, 0, -1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]
        )

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.parametrize(
        "excitation", [qml.DoubleExcitation, qml.DoubleExcitationPlus, qml.DoubleExcitationMinus]
    )
    def test_tf(self, excitation):
        """Tests that operations are computed correctly using the
        tensorflow interface"""

        pytest.importorskip("tensorflow")

        dev = qml.device("default.qubit.tf", wires=4)
        state = np.array(
            [0, 0, 0, -1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]
        )

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.parametrize(
        "excitation", [qml.DoubleExcitation, qml.DoubleExcitationPlus, qml.DoubleExcitationMinus]
    )
    def test_jax(self, excitation):
        """Tests that operations are computed correctly using the
        jax interface"""

        pytest.importorskip("jax")

        dev = qml.device("default.qubit.jax", wires=4)
        state = np.array(
            [0, 0, 0, -1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]
        )

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.DoubleExcitation, -0.1),
            (qml.DoubleExcitationPlus, 0.2),
            (qml.DoubleExcitationMinus, np.pi / 4),
        ],
    )
    def test_autograd_grad(self, excitation, phi):
        """Tests that gradients are computed correctly using the
        autograd interface"""

        pytest.importorskip("autograd")

        dev = qml.device("default.qubit.autograd", wires=4)

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.expval(qml.PauliZ(0))

        assert np.allclose(qml.grad(circuit)(phi), np.sin(phi))

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.DoubleExcitation, -0.1),
            (qml.DoubleExcitationPlus, 0.2),
            (qml.DoubleExcitationMinus, np.pi / 4),
        ],
    )
    def test_tf_grad(self, excitation, phi, diff_method):
        """Tests that gradients are computed correctly using the
        tensorflow interface"""

        tf = pytest.importorskip("tensorflow")
        dev = qml.device("default.qubit.tf", wires=4)

        @qml.qnode(dev, interface="tf", diff_method=diff_method)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0))

        phi_t = tf.Variable(phi, dtype=tf.float64)
        with tf.GradientTape() as tape:
            res = circuit(phi_t)

        grad = tape.gradient(res, phi_t)
        assert np.allclose(grad, np.sin(phi))

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.DoubleExcitation, -0.1),
            (qml.DoubleExcitationPlus, 0.2),
            (qml.DoubleExcitationMinus, np.pi / 4),
        ],
    )
    def test_jax_grad(self, excitation, phi, diff_method):
        """Tests that gradients and operations are computed correctly using the
        jax interface"""

        if diff_method == "parameter-shift":
            pytest.skip("JAX support for the parameter-shift method is still TBD")

        jax = pytest.importorskip("jax")

        dev = qml.device("default.qubit.jax", wires=4)

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(jax.grad(circuit)(phi), np.sin(phi))
