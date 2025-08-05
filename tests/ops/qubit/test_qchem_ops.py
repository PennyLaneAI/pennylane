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
import numpy as np

# pylint: disable=too-few-public-methods, unnecessary-lambda-assignment
import pytest
from gate_data import (
    DoubleExcitation,
    DoubleExcitationMinus,
    DoubleExcitationPlus,
    FermionicSWAP,
    OrbitalRotation,
    SingleExcitation,
    SingleExcitationMinus,
    SingleExcitationPlus,
)
from scipy.linalg import expm, fractional_matrix_power

import pennylane as qml
from pennylane import numpy as pnp

PARAMETRIZED_QCHEM_OPERATIONS = [
    qml.SingleExcitation(0.14, wires=[0, 1]),
    qml.SingleExcitationMinus(0.14, wires=[0, 1]),
    qml.SingleExcitationPlus(0.14, wires=[0, 1]),
    qml.DoubleExcitation(0.14, wires=[0, 1, 2, 3]),
    qml.DoubleExcitationMinus(0.14, wires=[0, 1, 2, 3]),
    qml.DoubleExcitationPlus(0.14, wires=[0, 1, 2, 3]),
    qml.OrbitalRotation(0.14, wires=[0, 1, 2, 3]),
    qml.FermionicSWAP(0.14, wires=[0, 1]),
]


class TestParameterFrequencies:

    @pytest.mark.parametrize("op", PARAMETRIZED_QCHEM_OPERATIONS)
    def test_parameter_frequencies_match_generator(self, op, tol):
        if not op.has_generator:
            pytest.skip(f"Operation {op.name} does not have a generator defined to test against.")

        gen = op.generator()

        try:
            mat = gen.matrix()
        except (AttributeError, qml.operation.MatrixUndefinedError):
            if isinstance(gen, (qml.Hamiltonian, qml.SparseHamiltonian)):
                mat = gen.sparse_matrix().toarray()
            else:
                pytest.skip(f"Operation {op.name}'s generator does not define a matrix.")

        gen_eigvals = np.round(np.linalg.eigvalsh(mat), 8)
        freqs_from_gen = qml.gradients.eigvals_to_frequencies(tuple(gen_eigvals))

        freqs = op.parameter_frequencies
        assert np.allclose(freqs, np.sort(freqs_from_gen), atol=tol)


class TestDecomposition:
    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    def test_single_excitation_plus_decomp(self, phi):
        """Tests that the SingleExcitationPlus operation calculates the correct decomposition.

        Need to consider the matrix of CRY separately, as the control is wire 1
        and the target is wire 0 in the decomposition. (Not applicable for
        ControlledPhase as it has the same matrix representation regardless of the
        control and target wires.)"""
        decomp1 = qml.SingleExcitationPlus(phi, wires=[0, 1]).decomposition()
        decomp2 = qml.SingleExcitationPlus.compute_decomposition(phi, wires=[0, 1])
        exp = SingleExcitationPlus(phi)
        for decomp in [decomp1, decomp2]:
            decomp_mat = qml.matrix(qml.tape.QuantumScript(decomp), wire_order=[0, 1])

            assert np.allclose(decomp_mat, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    def test_single_excitation_minus_decomp(self, phi):
        """Tests that the SingleExcitationMinus operation calculates the correct decomposition.

        Need to consider the matrix of CRY separately, as the control is wire 1
        and the target is wire 0 in the decomposition. (Not applicable for
        ControlledPhase as it has the same matrix representation regardless of the
        control and target wires.)"""
        decomp1 = qml.SingleExcitationMinus(phi, wires=[0, 1]).decomposition()
        decomp2 = qml.SingleExcitationMinus.compute_decomposition(phi, wires=[0, 1])
        exp = SingleExcitationMinus(phi)
        for decomp in [decomp1, decomp2]:
            decomp_mat = qml.matrix(qml.tape.QuantumScript(decomp), wire_order=[0, 1])

            assert np.allclose(decomp_mat, exp)


class TestSingleExcitation:
    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_matrix(self, phi):
        """Tests that the SingleExcitation operation calculates the correct matrix"""
        op = qml.SingleExcitation(phi, wires=[0, 1])
        res_dynamic = op.matrix()
        res_static = qml.SingleExcitation.compute_matrix(phi)
        exp = SingleExcitation(phi)
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    def test_single_excitation_matrix_broadcasted(self):
        """Tests that the SingleExcitation operation calculates the correct broadcasted matrix"""
        phi = np.array([-0.1, 0.2, np.pi / 4])
        op = qml.SingleExcitation(phi, wires=[0, 1])
        res_dynamic = op.matrix()
        res_static = qml.SingleExcitation.compute_matrix(phi)
        exp = np.stack([SingleExcitation(_phi) for _phi in phi])
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_decomp(self, phi):
        """Tests that the SingleExcitation operation calculates the correct decomposition."""
        exp = SingleExcitation(phi)

        decomp1 = qml.SingleExcitation(phi, wires=[0, 1]).decomposition()
        decomp2 = qml.SingleExcitation.compute_decomposition(phi, wires=[0, 1])

        for decomp in [decomp1, decomp2]:
            with qml.tape.QuantumTape() as tape:
                for op in decomp:
                    qml.apply(op)

            decomposed_matrix = qml.matrix(tape, wire_order=[0, 1])
            assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_generator(self, phi):
        """Tests that the SingleExcitation operation calculates the correct generator"""
        op = qml.SingleExcitation(phi, wires=[0, 1])
        g = qml.matrix(qml.generator(op, format="observable"))
        res = expm(1j * g * phi)
        exp = SingleExcitation(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("n", (2, -2, 1.3, -0.6))
    def test_single_excitatation_pow(self, n):
        op = qml.SingleExcitation(1.234, wires=(0, 1))

        pow_ops = op.pow(n)
        assert len(pow_ops) == 1
        assert pow_ops[0].__class__ is qml.SingleExcitation

        mat = qml.matrix(op)
        pow_mat = qml.matrix(op.pow, wire_order=op.wires)(n)

        mat_then_pow = fractional_matrix_power(mat, n)
        assert qml.math.allclose(pow_mat, mat_then_pow)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_plus_matrix(self, phi):
        """Tests that the SingleExcitationPlus operation calculates the correct matrix"""
        op = qml.SingleExcitationPlus(phi, wires=[0, 1])
        res_dynamic = op.matrix()
        res_static = qml.SingleExcitationPlus.compute_matrix(phi)
        exp = SingleExcitationPlus(phi)
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    def test_single_excitation_plus_matrix_broadcasted(self):
        """Tests that the SingleExcitationPlus operation
        calculates the correct broadcasted matrix"""
        phi = np.array([-0.1, 0.2, np.pi / 4])
        op = qml.SingleExcitationPlus(phi, wires=[0, 1])
        res_dynamic = op.matrix()
        res_static = qml.SingleExcitationPlus.compute_matrix(phi)
        exp = np.stack([SingleExcitationPlus(_phi) for _phi in phi])
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_plus_generator(self, phi):
        """Tests that the SingleExcitationPlus operation calculates the correct generator"""
        op = qml.SingleExcitationPlus(phi, wires=[0, 1])
        g = qml.matrix(qml.generator(op, format="observable"))
        res = expm(1j * g * phi)
        exp = SingleExcitationPlus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_minus_matrix(self, phi):
        """Tests that the SingleExcitationMinus operation calculates the correct matrix"""
        op = qml.SingleExcitationMinus(phi, wires=[0, 1])
        res_dynamic = op.matrix()
        res_static = qml.SingleExcitationMinus.compute_matrix(phi)
        exp = SingleExcitationMinus(phi)
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    def test_single_excitation_minus_matrix_broadcasted(self):
        """Tests that the SingleExcitationMinus operation
        calculates the correct broadcasted matrix"""
        phi = np.array([-0.1, 0.2, np.pi / 4])
        op = qml.SingleExcitationMinus(phi, wires=[0, 1])
        res_dynamic = op.matrix()
        res_static = qml.SingleExcitationMinus.compute_matrix(phi)
        exp = np.stack([SingleExcitationMinus(_phi) for _phi in phi])
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_minus_generator(self, phi):
        """Tests that the SingleExcitationMinus operation calculates the correct generator"""
        op = qml.SingleExcitationMinus(phi, wires=[0, 1])
        g = qml.matrix(qml.generator(op, format="observable"))
        res = expm(1j * g * phi)
        exp = SingleExcitationMinus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "excitation", [qml.SingleExcitation, qml.SingleExcitationPlus, qml.SingleExcitationMinus]
    )
    def test_autograd(self, excitation):
        """Tests that operations are computed correctly using the
        autograd interface"""

        dev = qml.device("default.qubit")
        state = np.array([0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0])

        @qml.qnode(dev, interface="autograd")
        def circuit(phi):
            qml.PauliX(wires=0)
            excitation(phi, wires=[0, 1])
            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.SingleExcitation, pnp.array(-0.1, requires_grad=True)),
            (qml.SingleExcitationPlus, pnp.array(0.2, requires_grad=True)),
            (qml.SingleExcitationMinus, pnp.array(np.pi / 4, requires_grad=True)),
        ],
    )
    def test_autograd_grad(self, diff_method, excitation, phi):
        """Tests that gradients are computed correctly using the
        autograd interface"""

        dev = qml.device("default.qubit")

        @qml.qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(phi):
            qml.PauliX(wires=0)
            excitation(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(qml.grad(circuit)(phi), np.sin(phi))

    @pytest.mark.tf
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

        import tensorflow as tf

        dev = qml.device("default.qubit")

        @qml.qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(phi):
            qml.PauliX(wires=0)
            excitation(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi_t = tf.Variable(phi, dtype=tf.float64)
        with tf.GradientTape() as tape:
            res = circuit(phi_t)

        grad = tape.gradient(res, phi_t)
        assert np.allclose(grad, np.sin(phi))

    @pytest.mark.jax
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

        import jax

        dev = qml.device("default.qubit")

        @qml.qnode(dev, diff_method=diff_method, interface="jax")
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
        res_dynamic = op.matrix()
        res_static = qml.DoubleExcitation.compute_matrix(phi)
        exp = DoubleExcitation(phi)
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    def test_double_excitation_matrix_broadcasted(self):
        """Tests that the DoubleExcitation operation
        calculates the correct broadcasted matrix"""
        phi = np.array([-0.1, 0.2, np.pi / 4])
        op = qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
        res_dynamic = op.matrix()
        res_static = qml.DoubleExcitation.compute_matrix(phi)
        exp = np.stack([DoubleExcitation(_phi) for _phi in phi])
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_generator(self, phi):
        """Tests that the DoubleExcitation operation calculates the correct generator"""
        op = qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
        g = qml.matrix(qml.generator(op, format="observable"))

        res = expm(1j * g * phi)
        exp = DoubleExcitation(phi)

        assert np.allclose(res, exp)

    @pytest.mark.parametrize("n", (2, -2, 1.3, -0.6))
    def test_double_excitatation_pow(self, n):
        """Test the double excitation pow method."""
        op = qml.DoubleExcitation(1.234, wires=(0, 1, 2, 3))

        pow_ops = op.pow(n)
        assert len(pow_ops) == 1
        assert pow_ops[0].__class__ is qml.DoubleExcitation

        mat = qml.matrix(op)
        pow_mat = qml.matrix(op.pow, wire_order=op.wires)(n)

        mat_then_pow = fractional_matrix_power(mat, n)
        assert qml.math.allclose(pow_mat, mat_then_pow)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    def test_double_excitation_decomp(self, phi):
        """Tests that the DoubleExcitation operation calculates the correct decomposition.

        The decomposition has already been expressed in terms of single-qubit rotations
        and CNOTs. For each term in the decomposition we need to construct the appropriate
        four-qubit tensor product matrix and then multiply them together.
        """
        decomp1 = qml.DoubleExcitation(phi, wires=[0, 1, 2, 3]).decomposition()
        decomp2 = qml.DoubleExcitation.compute_decomposition(phi, wires=[0, 1, 2, 3])

        for decomp in [decomp1, decomp2]:
            mats = [op.matrix(wire_order=list(range(4))) for op in decomp]

            decomposed_matrix = np.linalg.multi_dot(mats)
            exp = DoubleExcitation(phi)

            assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_plus_matrix(self, phi):
        """Tests that the DoubleExcitationPlus operation calculates the correct matrix"""
        op = qml.DoubleExcitationPlus(phi, wires=[0, 1, 2, 3])
        res_dynamic = op.matrix()
        res_static = qml.DoubleExcitationPlus.compute_matrix(phi)
        exp = DoubleExcitationPlus(phi)
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    def test_double_excitation_plus_matrix_broadcasted(self):
        """Tests that the DoubleExcitationPlus operation
        calculates the correct broadcasted matrix"""
        phi = np.array([-0.1, 0.2, np.pi / 4])
        op = qml.DoubleExcitationPlus(phi, wires=[0, 1, 2, 3])
        res_dynamic = op.matrix()
        res_static = qml.DoubleExcitationPlus.compute_matrix(phi)
        exp = np.stack([DoubleExcitationPlus(_phi) for _phi in phi])
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_plus_generator(self, phi):
        """Tests that the DoubleExcitationPlus operation calculates the correct generator"""
        op = qml.DoubleExcitationPlus(phi, wires=[0, 1, 2, 3])
        g = qml.matrix(qml.generator(op, format="observable"))

        res = expm(1j * g * phi)
        exp = DoubleExcitationPlus(phi)

        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_minus_matrix(self, phi):
        """Tests that the DoubleExcitationMinus operation calculates the correct matrix"""
        op = qml.DoubleExcitationMinus(phi, wires=[0, 1, 2, 3])
        res_dynamic = op.matrix()
        res_static = qml.DoubleExcitationMinus.compute_matrix(phi)
        exp = DoubleExcitationMinus(phi)
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    def test_double_excitation_minus_matrix_broadcasted(self):
        """Tests that the DoubleExcitationMinus operation
        calculates the correct broadcasted matrix"""
        phi = np.array([-0.1, 0.2, np.pi / 4])
        op = qml.DoubleExcitationMinus(phi, wires=[0, 1, 2, 3])
        res_dynamic = op.matrix()
        res_static = qml.DoubleExcitationMinus.compute_matrix(phi)
        exp = np.stack([DoubleExcitationMinus(_phi) for _phi in phi])
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_minus_generator(self, phi):
        """Tests that the DoubleExcitationMinus operation calculates the correct generator"""
        op = qml.DoubleExcitationMinus(phi, wires=[0, 1, 2, 3])
        g = qml.matrix(qml.generator(op, format="observable"))

        res = expm(1j * g * phi)
        exp = DoubleExcitationMinus(phi)

        assert np.allclose(res, exp)

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "excitation", [qml.DoubleExcitation, qml.DoubleExcitationPlus, qml.DoubleExcitationMinus]
    )
    def test_autograd(self, excitation):
        """Tests that operations are computed correctly using the
        autograd interface"""

        dev = qml.device("default.qubit")
        state = np.array(
            [0, 0, 0, -1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]
        )

        @qml.qnode(dev, interface="autograd")
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "excitation", [qml.DoubleExcitation, qml.DoubleExcitationPlus, qml.DoubleExcitationMinus]
    )
    def test_tf(self, excitation):
        """Tests that operations are computed correctly using the
        tensorflow interface"""

        dev = qml.device("default.qubit")
        state = np.array(
            [0, 0, 0, -1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]
        )

        @qml.qnode(dev, interface="tf")
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "excitation", [qml.DoubleExcitation, qml.DoubleExcitationPlus, qml.DoubleExcitationMinus]
    )
    def test_jax(self, excitation):
        """Tests that operations are computed correctly using the
        jax interface"""

        dev = qml.device("default.qubit")
        state = np.array(
            [0, 0, 0, -1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]
        )

        @qml.qnode(dev, interface="jax")
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.DoubleExcitation, pnp.array(-0.1, requires_grad=True)),
            (qml.DoubleExcitationPlus, pnp.array(0.2, requires_grad=True)),
            (qml.DoubleExcitationMinus, pnp.array(np.pi / 4, requires_grad=True)),
        ],
    )
    def test_autograd_grad(self, excitation, phi):
        """Tests that gradients are computed correctly using the
        autograd interface"""

        dev = qml.device("default.qubit")

        @qml.qnode(dev, interface="autograd")
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.expval(qml.PauliZ(0))

        assert np.allclose(qml.grad(circuit)(phi), np.sin(phi))

    @pytest.mark.tf
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

        import tensorflow as tf

        dev = qml.device("default.qubit")

        @qml.qnode(dev, diff_method=diff_method, interface="tf")
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

    @pytest.mark.jax
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

        import jax

        dev = qml.device("default.qubit")

        @qml.qnode(dev, diff_method=diff_method, interface="jax")
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(jax.grad(circuit)(phi), np.sin(phi))


class TestOrbitalRotation:
    """Test OrbitalRotation gate operation"""

    def grad_circuit_0(self, phi):
        qml.PauliX(1)
        qml.Hadamard(2)
        qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
        return qml.expval(qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(3))

    def grad_circuit_1(self, phi):
        qml.PauliX(0)
        qml.PauliX(1)
        qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(3))

    def expected_grad_fn(self, phi):
        # The following expression is obtained using the eight-parameter shift rule mentioned in arXiv:2107.12390.
        return 1.1 * (3 * np.sin(3 * phi / 2) - np.sin(phi / 2)) * 0.25 - 0.7 * np.sin(phi)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_orbital_rotation_matrix(self, phi):
        """Tests that the OrbitalRotation operation calculates the correct matrix"""
        op = qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
        res_dynamic = op.matrix()
        res_static = qml.OrbitalRotation.compute_matrix(phi)
        exp = OrbitalRotation(phi)
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    def test_orbital_rotation_matrix_broadcasted(self):
        """Tests that the OrbitalRotation operation
        calculates the correct broadcasted matrix"""
        phi = np.array([-0.1, 0.2, np.pi / 4])
        op = qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
        res_dynamic = op.matrix()
        res_static = qml.OrbitalRotation.compute_matrix(phi)
        exp = np.stack([OrbitalRotation(_phi) for _phi in phi])
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_orbital_rotation_generator(self, phi):
        """Tests that the OrbitalRotation operation calculates the correct generator"""
        op = qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
        g = qml.matrix(qml.generator(op, format="observable"))

        res = expm(1j * g * phi)
        exp = OrbitalRotation(phi)

        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    def test_orbital_rotation_decomp(self, phi):
        """Tests that the OrbitalRotation operation calculates the correct decomposition.

        The decomposition is expressed in terms of two SingleExcitation gates sandwiched between FermionicExcitations gates.
        """
        op = qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
        decomposed_matrix = qml.matrix(
            qml.FermionicSWAP(np.pi, wires=[1, 2])
            @ qml.SingleExcitation(phi, [0, 1])
            @ qml.SingleExcitation(phi, [2, 3])
            @ qml.FermionicSWAP(np.pi, wires=[1, 2]),
            wire_order=[0, 1, 2, 3],
        )
        assert np.allclose(decomposed_matrix, op.matrix())

    def test_adjoint(self):
        """Test adjoint method for adjoint op decomposition."""

        phi = 1.234
        wires = (0, 1, 2, 3)
        op = qml.OrbitalRotation(phi, wires=wires)
        adj_op = qml.adjoint(op, lazy=False)
        qml.assert_equal(adj_op, qml.OrbitalRotation(-phi, wires=wires))

    def test_adjoint_integration(self):
        """Test that the adjoint correctly inverts the orbital rotation operation"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
            qml.adjoint(qml.OrbitalRotation)(phi, wires=[0, 1, 2, 3])
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            return qml.state()

        res = circuit(0.1)

        expected = np.zeros([2**4])
        expected[0] = 1.0

        assert np.allclose(res, expected)

    @pytest.mark.parametrize("ref_state", [np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0])])
    @pytest.mark.parametrize("op", [qml.qchem.particle_number(4), qml.qchem.spin2(2, 4)])
    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_spin_particle_conservation(self, ref_state, op, phi):
        """Test that the total spin and particle are conserved after orbital rotation operation"""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit1():
            qml.BasisState(ref_state, wires=[0, 1, 2, 3])
            return qml.expval(op)

        @qml.qnode(dev)
        def circuit2(phi):
            qml.BasisState(ref_state, wires=[0, 1, 2, 3])
            qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
            return qml.expval(op)

        assert np.allclose(circuit1(), circuit2(phi))

    @pytest.mark.autograd
    def test_autograd(self):
        """Tests that operations are computed correctly using the
        autograd interface"""

        dev = qml.device("default.qubit")
        state = np.array(
            [
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                -0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ]
        )

        @qml.qnode(dev, interface="autograd")
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.tf
    def test_tf(self):
        """Tests that operations are computed correctly using the
        tensorflow interface"""

        dev = qml.device("default.qubit")
        state = np.array(
            [
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                -0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ]
        )

        @qml.qnode(dev, interface="tf")
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.jax
    def test_jax(self):
        """Tests that operations are computed correctly using the
        jax interface"""

        dev = qml.device("default.qubit")
        state = np.array(
            [
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                -0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ]
        )

        @qml.qnode(dev, interface="jax")
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.torch
    def test_torch(self):
        """Tests that operations are computed correctly using the
        torch interface"""

        dev = qml.device("default.qubit")
        state = np.array(
            [
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                -0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ]
        )

        @qml.qnode(dev, interface="torch")
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        "phi",
        [
            pnp.array(-0.1, requires_grad=True),
            pnp.array(0.1421, requires_grad=True),
        ],
    )
    def test_autograd_grad(self, phi, diff_method):
        """Tests that gradients are computed correctly using the
        autograd interface"""

        dev = qml.device("default.qubit")

        circuit_0 = qml.QNode(
            self.grad_circuit_0, dev, interface="autograd", diff_method=diff_method
        )
        circuit_1 = qml.QNode(
            self.grad_circuit_1, dev, interface="autograd", diff_method=diff_method
        )
        total = lambda phi: 1.1 * circuit_0(phi) + 0.7 * circuit_1(phi)

        assert np.allclose(qml.grad(total)(phi), self.expected_grad_fn(phi))

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("phi"),
        [-0.1, 0.1421],
    )
    def test_tf_grad(self, phi, diff_method):
        """Tests that gradients are computed correctly using the
        tensorflow interface"""

        import tensorflow as tf

        dev = qml.device("default.qubit")

        circuit_0 = qml.QNode(self.grad_circuit_0, dev, interface="tf", diff_method=diff_method)
        circuit_1 = qml.QNode(self.grad_circuit_1, dev, interface="tf", diff_method=diff_method)
        total = lambda phi: 1.1 * circuit_0(phi) + 0.7 * circuit_1(phi)

        phi_t = tf.Variable(phi, dtype=tf.float64)
        with tf.GradientTape() as tape:
            res = total(phi_t)

        grad = tape.gradient(res, phi_t)

        assert np.allclose(grad, self.expected_grad_fn(phi))

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("phi"),
        [-0.1, 0.1421],
    )
    def test_jax_grad(self, phi, diff_method):
        """Tests that gradients and operations are computed correctly using the
        jax interface"""

        import jax

        dev = qml.device("default.qubit")

        circuit_0 = qml.QNode(self.grad_circuit_0, dev, interface="jax", diff_method=diff_method)
        circuit_1 = qml.QNode(self.grad_circuit_1, dev, interface="jax", diff_method=diff_method)
        total = lambda phi: 1.1 * circuit_0(phi) + 0.7 * circuit_1(phi)

        phi_j = jax.numpy.array(phi)

        assert np.allclose(jax.grad(total)(phi_j), self.expected_grad_fn(phi))

    @pytest.mark.torch
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("phi"),
        [-0.1, 0.1421],
    )
    def test_torch_grad(self, phi, diff_method):
        """Tests that gradients and operations are computed correctly using the
        torch interface"""

        import torch

        dev = qml.device("default.qubit")

        circuit_0 = qml.QNode(self.grad_circuit_0, dev, interface="torch", diff_method=diff_method)
        circuit_1 = qml.QNode(self.grad_circuit_1, dev, interface="torch", diff_method=diff_method)
        total = lambda phi: 1.1 * circuit_0(phi) + 0.7 * circuit_1(phi)

        phi_t = torch.tensor(phi, dtype=torch.complex128, requires_grad=True)
        result = total(phi_t)
        result.backward()

        assert np.allclose(phi_t.grad, self.expected_grad_fn(phi))


class TestFermionicSWAP:
    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_fermionic_swap_matrix(self, phi):
        """Tests that the FermionicSWAP operation calculates the correct matrix"""
        op = qml.FermionicSWAP(phi, wires=[0, 1])
        res_dynamic = op.matrix()
        res_static = qml.FermionicSWAP.compute_matrix(phi)
        exp = FermionicSWAP(phi)
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    def test_fermionic_swap_broadcasted(self):
        """Tests that the FermionicSWAP operation calculates the correct broadcasted matrix"""
        phi = np.array([-0.1, 0.2, np.pi / 4])
        op = qml.FermionicSWAP(phi, wires=[0, 1])
        res_dynamic = op.matrix()
        res_static = qml.FermionicSWAP.compute_matrix(phi)
        exp = np.stack([FermionicSWAP(_phi) for _phi in phi])
        assert np.allclose(res_dynamic, exp)
        assert np.allclose(res_static, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_fermionic_swap_decomp(self, phi):
        """Tests that the FermionicSWAP operation calculates the correct decomposition"""
        decomp1 = qml.FermionicSWAP(phi, wires=[0, 1]).decomposition()
        decomp2 = qml.FermionicSWAP.compute_decomposition(phi, wires=[0, 1])

        for decomp in [decomp1, decomp2]:
            mats = []
            for i in reversed(decomp):
                mats.append(i.matrix(wire_order=[0, 1]))

            decomposed_matrix = np.linalg.multi_dot(mats)
            exp = FermionicSWAP(phi)

            assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_fermionic_swap_generator(self, phi):
        """Tests that the FermionicSWAP operation calculates the correct generator"""
        op = qml.FermionicSWAP(phi, wires=[0, 1])
        g = qml.matrix(qml.generator(op, format="observable"))
        res = expm(1j * g * phi)
        exp = FermionicSWAP(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("n", (2, -2, 1.3, -0.6))
    def test_fermionic_swap_pow(self, n):
        op = qml.FermionicSWAP(1.234, wires=(0, 1))

        pow_ops = op.pow(n)
        assert len(pow_ops) == 1
        assert pow_ops[0].__class__ is qml.FermionicSWAP

        mat = qml.matrix(op)
        pow_mat = qml.matrix(op.pow, wire_order=op.wires)(n)

        mat_then_pow = fractional_matrix_power(mat, n)
        assert qml.math.allclose(pow_mat, mat_then_pow)

    def test_adjoint(self):
        """Test adjoint method for adjoint op decomposition"""

        phi = 1.234
        wires = (0, 1)
        op = qml.FermionicSWAP(phi, wires=wires)
        adj_op = qml.adjoint(op, lazy=False)
        qml.assert_equal(adj_op, qml.FermionicSWAP(-phi, wires=wires))

    def test_adjoint_integration(self):
        """Test that the adjoint correctly inverts the fermionic swap operation"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.FermionicSWAP(phi, wires=[0, 1])
            qml.adjoint(qml.FermionicSWAP)(phi, wires=[0, 1])
            qml.PauliX(wires=0)
            return qml.state()

        res = circuit(0.1)

        expected = np.zeros(4)
        expected[0] = 1.0

        assert np.allclose(res, expected)

    @pytest.mark.autograd
    def test_autograd(self):
        """Tests that operations are computed correctly using the
        autograd interface"""

        dev = qml.device("default.qubit")
        state = np.array(
            [
                0,
                -1j * np.exp(1j * np.pi / 4) * np.sin(np.pi / 4),
                np.exp(1j * np.pi / 4) * np.cos(np.pi / 4),
                0,
            ]
        )

        @qml.qnode(dev, interface="autograd")
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.FermionicSWAP(phi, wires=[0, 1])
            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("phi"),
        [
            pnp.array(-0.1, requires_grad=True),
            pnp.array(0.2, requires_grad=True),
            pnp.array(np.pi / 4, requires_grad=True),
        ],
    )
    def test_autograd_grad(self, diff_method, phi):
        """Tests that gradients are computed correctly using the
        autograd interface"""

        dev = qml.device("default.qubit")

        @qml.qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.FermionicSWAP(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(qml.grad(circuit)(phi), np.sin(phi))

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("phi"),
        [
            -0.1,
            0.2,
            np.pi / 4,
        ],
    )
    def test_tf(self, phi, diff_method):
        """Tests that gradients and operations are computed correctly using the
        tensorflow interface"""

        import tensorflow as tf

        dev = qml.device("default.qubit")

        @qml.qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.FermionicSWAP(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi_t = tf.Variable(phi, dtype=tf.float64)
        with tf.GradientTape() as tape:
            res = circuit(phi_t)

        grad = tape.gradient(res, phi_t)
        assert np.allclose(grad, np.sin(phi))

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("phi"),
        [
            -0.1,
            0.2,
            np.pi / 4,
        ],
    )
    def test_jax(self, phi, diff_method):
        """Tests that gradients and operations are computed correctly using the
        jax interface"""

        import jax

        dev = qml.device("default.qubit")

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.FermionicSWAP(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(jax.grad(circuit)(phi), np.sin(phi))


label_data = [
    (qml.SingleExcitation(1.2345, wires=(0, 1)), "G", "G\n(1.23)", "G\n(1)"),
    (qml.SingleExcitationMinus(1.2345, wires=(0, 1)), "G₋", "G₋\n(1.23)", "G₋\n(1)"),
    (qml.SingleExcitationPlus(1.2345, wires=(0, 1)), "G₊", "G₊\n(1.23)", "G₊\n(1)"),
    (qml.DoubleExcitation(2.3456, wires=(0, 1, 2, 3)), "G²", "G²\n(2.35)", "G²\n(2)"),
    (qml.DoubleExcitationPlus(2.3456, wires=(0, 1, 2, 3)), "G²₊", "G²₊\n(2.35)", "G²₊\n(2)"),
    (qml.DoubleExcitationMinus(2.345, wires=(0, 1, 2, 3)), "G²₋", "G²₋\n(2.35)", "G²₋\n(2)"),
    (
        qml.OrbitalRotation(2.3456, wires=(0, 1, 2, 3)),
        "OrbitalRotation",
        "OrbitalRotation\n(2.35)",
        "OrbitalRotation\n(2)",
    ),
    (qml.FermionicSWAP(1.2345, wires=(0, 1)), "fSWAP", "fSWAP\n(1.23)", "fSWAP\n(1)"),
]


@pytest.mark.parametrize("op, label1, label2, label3", label_data)
def test_label_method(op, label1, label2, label3):
    """Test the label method for qchem operations."""
    assert op.label() == label1
    assert op.label(decimals=2) == label2
    assert op.label(decimals=0) == label3


@pytest.mark.parametrize("op", PARAMETRIZED_QCHEM_OPERATIONS)
def test_generators(op):
    """Check that the type of the generator returned by the qchem ops is
    the same as the type pointed to by LinearCombiantion"""
    if isinstance(op, (qml.ops.DoubleExcitationPlus, qml.ops.DoubleExcitationMinus)):
        pytest.skip(reason="Operator has SparseHamiltonian generator instead")

    assert isinstance(op.generator(), qml.Hamiltonian)
