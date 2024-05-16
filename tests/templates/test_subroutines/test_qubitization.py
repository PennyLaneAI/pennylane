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
Tests for the Qubitization template.
"""

import copy

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines.qubitization import _positive_coeffs_hamiltonian


@pytest.mark.parametrize(
    "hamiltonian, expected_unitaries",
    (
        (
            qml.ops.LinearCombination(
                np.array([1, -1, 2]), [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
            ),
            [
                qml.PauliX(0) @ qml.GlobalPhase(np.array([0.0]), wires=0),
                qml.PauliY(0) @ qml.GlobalPhase(np.array(np.pi), wires=0),
                qml.PauliZ(0) @ qml.GlobalPhase(np.array([0.0]), wires=0),
            ],
        ),
        (
            qml.ops.LinearCombination(
                np.array([1.0, 1.0, 2.0]), [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
            ),
            [
                qml.PauliX(0) @ qml.GlobalPhase(np.array([0.0]), wires=0),
                qml.PauliY(0) @ qml.GlobalPhase(np.array([0.0]), wires=0),
                qml.PauliZ(0) @ qml.GlobalPhase(np.array([0.0]), wires=0),
            ],
        ),
        (
            qml.ops.LinearCombination(
                np.array([-0.2, -0.6, 2.1]), [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
            ),
            [
                qml.PauliX(0) @ qml.GlobalPhase(np.array(np.pi), wires=0),
                qml.PauliY(0) @ qml.GlobalPhase(np.array(np.pi), wires=0),
                qml.PauliZ(0) @ qml.GlobalPhase(np.array(0), wires=0),
            ],
        ),
    ),
)
def test_positive_coeffs_hamiltonian(hamiltonian, expected_unitaries):
    """Tests that the function _positive_coeffs_hamiltonian correctly transforms the Hamiltonian"""

    new_coeffs, new_unitaries = _positive_coeffs_hamiltonian(hamiltonian)

    assert np.allclose(new_coeffs, np.abs(hamiltonian.terms()[0]))

    for i, unitary in enumerate(new_unitaries):
        assert qml.equal(expected_unitaries[i], unitary)


@pytest.mark.parametrize(
    "hamiltonian",
    [
        qml.dot([0.2, -0.5, 0.3], [qml.Y(0) @ qml.X(1), qml.Z(1), qml.X(0) @ qml.Z(2)]),
        qml.dot([0.3, -0.5, 0.3], [qml.Z(0) @ qml.X(1), qml.X(1), qml.X(0) @ qml.Y(2)]),
        qml.dot([0.4, -0.5, -0.3], [qml.Z(0) @ qml.X(2), qml.Y(0), qml.X(1) @ qml.Z(2)]),
    ],
)
def test_operator_definition_qpe(hamiltonian):
    """Tests that Qubitization can be used in QPE to obtain the eigenvalues of a Hamiltonian."""

    from scipy.signal import find_peaks

    @qml.qnode(qml.device("default.qubit"))
    def circuit(theta):

        # initial state
        qml.RX(theta[2], wires=0)
        qml.CRY(theta[3], wires=[0, 2])
        qml.RY(theta[0], wires=2)
        qml.CRY(theta[4], wires=[1, 2])
        qml.RX(theta[1], wires=1)
        qml.CRX(theta[2], wires=[2, 0])

        # apply QPE (used iterative qpe here)
        measurements = qml.iterative_qpe(
            qml.Qubitization(hamiltonian, control=[3, 4]), ancilla=5, iters=8
        )

        return qml.probs(op=measurements)

    theta = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 100

    peaks, _ = find_peaks(circuit(theta))

    # Calculates the eigenvalues from the obtained output
    lamb = sum([abs(c) for c in hamiltonian.terms()[0]])
    estimated_eigenvalues = lamb * np.cos(2 * np.pi * peaks / 2**8)

    assert np.allclose(np.sort(estimated_eigenvalues), qml.eigvals(hamiltonian), atol=0.1)


def test_standard_validity():
    """Check the operation using the assert_valid function."""
    H = qml.dot([0.1, -0.3, -0.3], [qml.X(0), qml.Z(1), qml.Y(0) @ qml.Z(2)])
    op = qml.Qubitization(H, control=[3, 4])
    qml.ops.functions.assert_valid(op)


@pytest.mark.usefixtures("use_legacy_and_new_opmath")
def test_legacy_new_opmath():
    coeffs, ops = [0.1, -0.3, -0.3], [qml.X(0), qml.Z(1), qml.Y(0) @ qml.Z(2)]

    H1 = qml.dot(coeffs, ops)
    matrix_H1 = qml.matrix(qml.Qubitization(H1, control=[3, 4]), wire_order=[3, 4, 0, 1, 2])

    H2 = qml.Hamiltonian(coeffs, ops)
    matrix_H2 = qml.matrix(qml.Qubitization(H2, control=[3, 4]), wire_order=[3, 4, 0, 1, 2])

    assert np.allclose(matrix_H1, matrix_H2)


@pytest.mark.parametrize(
    "hamiltonian, expected_decomposition",
    (
        (
            qml.ops.LinearCombination(np.array([1.0, 1.0]), [qml.PauliX(0), qml.PauliZ(0)]),
            [
                qml.AmplitudeEmbedding(np.array([1.0, 1.0]) / np.sqrt(2), wires=[1]),
                qml.Select(
                    ops=(
                        qml.PauliX(0) @ qml.GlobalPhase(np.array(0.0), wires=0),
                        qml.PauliZ(0) @ qml.GlobalPhase(np.array(0.0), wires=0),
                    ),
                    control=[1],
                ),
                qml.adjoint(qml.AmplitudeEmbedding(np.array([1.0, 1.0]) / np.sqrt(2), wires=[1])),
                qml.Reflection(qml.Identity(wires=[1])),
            ],
        ),
        (
            qml.ops.LinearCombination(np.array([-1.0, 1.0]), [qml.PauliX(0), qml.PauliZ(0)]),
            [
                qml.AmplitudeEmbedding(np.array([1.0, 1.0]) / np.sqrt(2), wires=[1]),
                qml.Select(
                    ops=(
                        qml.PauliX(0) @ qml.GlobalPhase(np.array(np.pi), wires=0),
                        qml.PauliZ(0) @ qml.GlobalPhase(np.array(0.0), wires=0),
                    ),
                    control=[1],
                ),
                qml.adjoint(qml.AmplitudeEmbedding(np.array([1.0, 1.0]) / np.sqrt(2), wires=[1])),
                qml.Reflection(qml.Identity(wires=[1])),
            ],
        ),
    ),
)
def test_decomposition(hamiltonian, expected_decomposition):
    """Tests that the Qubitization template is correctly decomposed."""

    decomposition = qml.Qubitization.compute_decomposition(hamiltonian=hamiltonian, control=[1])

    for i, op in enumerate(decomposition):
        assert qml.equal(op, expected_decomposition[i])


def test_lightning_qubit():
    H = qml.ops.LinearCombination([0.1, 0.3, -0.3], [qml.Z(0), qml.Z(1), qml.Z(0) @ qml.Z(2)])

    @qml.qnode(qml.device("lightning.qubit", wires=5))
    def circuit_lightning():
        qml.Hadamard(wires=0)
        qml.Qubitization(H, control=[3, 4])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(4))

    @qml.qnode(qml.device("default.qubit", wires=5))
    def circuit_default():
        qml.Hadamard(wires=0)
        qml.Qubitization(H, control=[3, 4])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(4))

    assert np.allclose(circuit_lightning(), circuit_default())


class TestDifferentiability:
    """Test that Qubitization is differentiable"""

    @staticmethod
    def circuit(coeffs):
        H = qml.ops.LinearCombination(
            coeffs, [qml.Y(0), qml.Y(1) @ qml.Y(2), qml.X(0), qml.X(1) @ qml.X(2)]
        )
        qml.Qubitization(H, control=(3, 4))
        return qml.expval(qml.PauliZ(3) @ qml.PauliZ(4))

    # calculated numerically with finite diff method (h = 1e-4)
    exp_grad = np.array([0.41177729, -0.21262358, 1.64370464, -0.74256522])

    params = np.array([0.4, 0.5, 0.1, 0.3])

    @pytest.mark.autograd
    def test_qnode_autograd(self):
        """Test that the QNode executes with Autograd."""

        dev = qml.device("default.qubit")
        qnode = qml.QNode(self.circuit, dev, interface="autograd")

        params = qml.numpy.array(self.params, requires_grad=True)
        res = qml.grad(qnode)(params)
        assert qml.math.shape(res) == (4,)
        assert np.allclose(res, self.exp_grad, atol=1e-5)

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "use_jit , shots",
        ((False, None), (True, None), (False, 50000)),
    )  # TODO: (True, 50000) fails because jax.jit on jax.grad does not work with AmplitudeEmbedding
    def test_qnode_jax(self, shots, use_jit):
        """ "Test that the QNode executes and is differentiable with JAX. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", shots=shots, seed=10)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.QNode(self.circuit, dev, interface="jax", diff_method=diff_method)
        if use_jit:
            qnode = jax.jit(qnode)

        params = jax.numpy.array(self.params)

        jac_fn = jax.jacobian(qnode)
        if use_jit:
            jac_fn = jax.jit(jac_fn)

        jac = jac_fn(params)
        assert jac.shape == (4,)
        assert np.allclose(jac, self.exp_grad, atol=0.01)

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "shots", [None]
    )  # TODO: finite shots fails because Prod is not currently differentiable.
    def test_qnode_torch(self, shots):
        """ "Test that the QNode executes and is differentiable with Torch. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import torch

        dev = qml.device("default.qubit", shots=shots, seed=10)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.QNode(self.circuit, dev, interface="torch", diff_method=diff_method)

        params = torch.tensor(self.params, requires_grad=True)
        jac = torch.autograd.functional.jacobian(qnode, params)
        assert qml.math.shape(jac) == (4,)
        assert qml.math.allclose(jac, self.exp_grad, atol=0.01)

    @pytest.mark.tf
    @pytest.mark.parametrize("shots", [None, 50000])
    @pytest.mark.xfail(reason="tf gradient doesn't seem to be working, returns ()")
    def test_qnode_tf(self, shots):
        """ "Test that the QNode executes and is differentiable with TensorFlow. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import tensorflow as tf

        dev = qml.device("default.qubit", shots=shots, seed=10)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.QNode(self.circuit, dev, interface="tf", diff_method=diff_method)

        params = tf.Variable(self.params)
        with tf.GradientTape() as tape:
            res = qnode(params)

        jac = tape.gradient(res, params)
        assert qml.math.shape(jac) == (4,)
        assert qml.math.allclose(res, self.exp_grad, atol=0.001)

    @pytest.mark.xfail(reason="see https://github.com/PennyLaneAI/pennylane/issues/5507")
    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    def test_legacy_new_opmath_diff(self):
        coeffs, ops = np.array([0.1, -0.3, -0.3]), [qml.X(0), qml.Z(1), qml.Y(0) @ qml.Z(2)]

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit_dot(coeffs):
            H = qml.dot(coeffs, ops)
            qml.Qubitization(H, control=[3, 4])
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def circuit_Hamiltonian(coeffs):
            H = qml.Hamiltonian(coeffs, ops)
            qml.Qubitization(H, control=[3, 4])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(qml.grad(circuit_dot)(coeffs), qml.grad(circuit_Hamiltonian)(coeffs))


def test_copy():
    """Test that a Qubitization operator can be copied."""

    H = qml.dot([1.0, 2.0], [qml.PauliX(0), qml.PauliZ(1)])

    orig_op = qml.Qubitization(H, control=[2, 3])
    copy_op = copy.copy(orig_op)
    assert qml.equal(orig_op, copy_op)

    # Ensure the (nested) operations are copied instead of aliased.
    assert orig_op is not copy_op
    assert orig_op.hyperparameters["hamiltonian"] is not copy_op.hyperparameters["hamiltonian"]
    assert orig_op.hyperparameters["control"] is not copy_op.hyperparameters["control"]


def test_map_wires():
    """Test that a Qubitization operator can be mapped to a different wire mapping."""

    H = qml.dot([1.0, 2.0], [qml.PauliX(0), qml.PauliZ(1)])

    op = qml.Qubitization(H, control=[2, 3])
    op2 = op.map_wires({0: 5, 1: 6, 2: 7, 3: 8})

    assert op2.wires == qml.wires.Wires([5, 6, 7, 8])
