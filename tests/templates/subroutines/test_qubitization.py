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

import pennylane as qp
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.parametrize(
    "hamiltonian",
    [
        qp.dot([0.2, -0.5, 0.3], [qp.Y(0) @ qp.X(1), qp.Z(1), qp.X(0) @ qp.Z(2)]),
        qp.dot([0.3, -0.5, 0.3], [qp.Z(0) @ qp.X(1), qp.X(1), qp.X(0) @ qp.Y(2)]),
        qp.dot([0.4, -0.5, -0.3], [qp.Z(0) @ qp.X(2), qp.Y(0), qp.X(1) @ qp.Z(2)]),
    ],
)
def test_operator_definition_qpe(hamiltonian):
    """Tests that Qubitization can be used in QPE to obtain the eigenvalues of a Hamiltonian."""

    from scipy.signal import find_peaks

    @qp.qnode(qp.device("default.qubit"))
    def circuit(theta):

        # initial state
        qp.RX(theta[2], wires=0)
        qp.CRY(theta[3], wires=[0, 2])
        qp.RY(theta[0], wires=2)
        qp.CRY(theta[4], wires=[1, 2])
        qp.RX(theta[1], wires=1)
        qp.CRX(theta[2], wires=[2, 0])

        # apply QPE (used iterative qpe here)
        measurements = qp.iterative_qpe(
            qp.Qubitization(hamiltonian, control=[3, 4]), aux_wire=5, iters=8
        )

        return qp.probs(op=measurements)

    theta = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 100

    peaks, _ = find_peaks(circuit(theta))

    # Calculates the eigenvalues from the obtained output
    lamb = sum(abs(c) for c in hamiltonian.terms()[0])
    estimated_eigenvalues = lamb * np.cos(2 * np.pi * peaks / 2**8)

    assert np.allclose(np.sort(estimated_eigenvalues), qp.eigvals(hamiltonian), atol=0.1)


@pytest.mark.jax
@pytest.mark.parametrize(
    ("lcu", "control"),
    [
        (qp.X(1) @ qp.Z(2), [0]),
        (qp.X(0) @ qp.Z(1), [2]),
        (qp.X(1) @ qp.Z(2) @ qp.Y(3), [0]),
        (qp.X(0) @ qp.Z(1) @ qp.Y(2), [3]),
        (qp.PauliX("a") @ qp.PauliZ(1), [0]),
        (qp.PauliX("a") @ qp.PauliZ(1) @ qp.PauliY(2), [0]),
    ],
)
def test_standard_validity(lcu, control):
    """Check the operation using the assert_valid function."""
    op = qp.Qubitization(lcu, control)
    # Skip differentiation for test cases that raise NaNs in gradients (known limitation of ``MottonenStatePreparation``).
    qp.ops.functions.assert_valid(op)


@pytest.mark.parametrize(
    "hamiltonian, expected_decomposition",
    (
        (
            qp.ops.LinearCombination(np.array([1.0, 1.0]), [qp.PauliX(0), qp.PauliZ(0)]),
            [
                qp.Reflection(qp.I([1]), 3.141592653589793),
                qp.PrepSelPrep(
                    qp.ops.LinearCombination(np.array([1.0, 1.0]), [qp.PauliX(0), qp.PauliZ(0)]),
                    control=[1],
                ),
            ],
        ),
        (
            qp.ops.LinearCombination(np.array([-1.0, 1.0]), [qp.PauliX(0), qp.PauliZ(0)]),
            [
                qp.Reflection(qp.I(1), 3.141592653589793),
                qp.PrepSelPrep(
                    qp.ops.LinearCombination(
                        np.array([-1.0, 1.0]), [qp.PauliX(0), qp.PauliZ(0)]
                    ),
                    control=[1],
                ),
            ],
        ),
    ),
)
def test_decomposition(hamiltonian, expected_decomposition):
    """Tests that the Qubitization template is correctly decomposed."""

    decomposition = qp.Qubitization.compute_decomposition(hamiltonian=hamiltonian, control=[1])
    for i, op in enumerate(decomposition):
        qp.assert_equal(op, expected_decomposition[i])


@pytest.mark.parametrize(
    "hamiltonian, control",
    [
        (qp.X(1) @ qp.Z(2), [0]),
        (qp.X(0) @ qp.Z(1), [2]),
        (qp.X(1) @ qp.Z(2) @ qp.Y(3), [0]),
        (qp.X(0) @ qp.Z(1) @ qp.Y(2), [3]),
        (qp.PauliX("a") @ qp.PauliZ(1), [0]),
        (qp.PauliX("a") @ qp.PauliZ(1) @ qp.PauliY(2), [0]),
    ],
)
def test_decomposition_new(hamiltonian, control):  # pylint: disable=unused-argument
    """Tests the decomposition rule implemented with the new system."""
    op = qp.Qubitization(hamiltonian, control=control)
    for rule in qp.list_decomps(qp.Qubitization):
        _test_decomposition_rule(op, rule)


def test_lightning_qubit():
    H = qp.ops.LinearCombination([0.1, 0.3, -0.3], [qp.Z(0), qp.Z(1), qp.Z(0) @ qp.Z(2)])

    @qp.qnode(qp.device("lightning.qubit", wires=5))
    def circuit_lightning():
        qp.Hadamard(wires=0)
        qp.Qubitization(H, control=[3, 4])
        return qp.expval(qp.PauliZ(0) @ qp.PauliZ(4))

    @qp.qnode(qp.device("default.qubit", wires=5))
    def circuit_default():
        qp.Hadamard(wires=0)
        qp.Qubitization(H, control=[3, 4])
        return qp.expval(qp.PauliZ(0) @ qp.PauliZ(4))

    assert np.allclose(circuit_lightning(), circuit_default())


class TestDifferentiability:
    """Test that Qubitization is differentiable"""

    @staticmethod
    def circuit(coeffs):
        H = qp.ops.LinearCombination(
            coeffs, [qp.Y(0), qp.Y(1) @ qp.Y(2), qp.X(0), qp.X(1) @ qp.X(2)]
        )
        qp.Qubitization(H, control=(3, 4))
        return qp.expval(qp.PauliZ(3) @ qp.PauliZ(4))

    # calculated numerically with finite diff method (h = 1e-4)
    exp_grad = np.array([0.41177729, -0.21262358, 1.64370464, -0.74256522])

    params = np.array([0.4, 0.5, 0.1, 0.3])

    @pytest.mark.autograd
    def test_qnode_autograd(self):
        """Test that the QNode executes with Autograd."""

        dev = qp.device("default.qubit")
        qnode = qp.QNode(self.circuit, dev, interface="autograd")

        params = qp.numpy.array(self.params, requires_grad=True)
        res = qp.grad(qnode)(params)
        assert qp.math.shape(res) == (4,)
        assert np.allclose(res, self.exp_grad, atol=1e-5)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (False, True))
    @pytest.mark.parametrize("shots", (None, 50000))
    def test_qnode_jax(self, shots, use_jit, seed):
        """Test that the QNode executes and it's differentiable with JAX. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qp.device("default.qubit", seed=seed)

        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qp.set_shots(
            qp.QNode(self.circuit, dev, interface="jax", diff_method=diff_method), shots=shots
        )
        if use_jit:
            qnode = jax.jit(qnode)

        params = jax.numpy.array(self.params)

        jac_fn = jax.jacobian(qnode)
        if use_jit:
            jac_fn = jax.jit(jac_fn)

        jac = jac_fn(params)
        assert jac.shape == (4,)

        atol = 1e-5 if shots is None else 0.05
        assert np.allclose(jac, self.exp_grad, atol=atol)

    @pytest.mark.torch
    @pytest.mark.parametrize("shots", [None, 50000])
    def test_qnode_torch(self, shots, seed):
        """ "Test that the QNode executes and is differentiable with Torch. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import torch

        dev = qp.device("default.qubit", seed=seed)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qp.set_shots(
            qp.QNode(self.circuit, dev, interface="torch", diff_method=diff_method), shots=shots
        )

        params = torch.tensor(self.params, requires_grad=True)
        jac = torch.autograd.functional.jacobian(qnode, params)
        assert qp.math.shape(jac) == (4,)
        atol = 1e-5 if shots is None else 0.05
        assert qp.math.allclose(jac, self.exp_grad, atol=atol)

    @pytest.mark.tf
    @pytest.mark.parametrize("shots", [None, 50000])
    @pytest.mark.xfail(reason="tf gradient doesn't seem to be working, returns ()")
    def test_qnode_tf(self, shots, seed):
        """ "Test that the QNode executes and is differentiable with TensorFlow. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import tensorflow as tf

        dev = qp.device("default.qubit", seed=seed)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qp.set_shots(
            qp.QNode(self.circuit, dev, interface="tf", diff_method=diff_method), shots=shots
        )

        params = tf.Variable(self.params)
        with tf.GradientTape() as tape:
            res = qnode(params)

        jac = tape.gradient(res, params)
        assert qp.math.shape(jac) == (4,)
        assert qp.math.allclose(res, self.exp_grad, atol=0.001)


def test_copy():
    """Test that a Qubitization operator can be copied."""

    H = qp.dot([1.0, 2.0], [qp.PauliX(0), qp.PauliZ(1)])

    orig_op = qp.Qubitization(H, control=[2, 3])
    copy_op = copy.copy(orig_op)
    qp.assert_equal(orig_op, copy_op)

    # Ensure the (nested) operations are copied instead of aliased.
    assert orig_op is not copy_op
    assert orig_op.hyperparameters["hamiltonian"] is not copy_op.hyperparameters["hamiltonian"]
    assert orig_op.hyperparameters["control"] is not copy_op.hyperparameters["control"]


def test_map_wires():
    """Test that a Qubitization operator can be mapped to a different wire mapping."""

    H = qp.dot([1.0, 2.0], [qp.PauliX(0), qp.PauliZ(1)])

    op = qp.Qubitization(H, control=[2, 3])
    op2 = op.map_wires({0: 5, 1: 6, 2: 7, 3: 8})

    assert op2.wires == qp.wires.Wires([7, 8, 5, 6])


@pytest.mark.parametrize(
    "hamiltonian, control",
    [
        (qp.dot([1.0, 2.0], [qp.PauliX("a"), qp.PauliZ(1)]), [0]),
        (qp.dot([1.0, -2.0], [qp.PauliX("a"), qp.PauliZ(1)]), [0]),
        (
            qp.dot(
                [1.0, 2.0, 1.0, 1.0],
                [qp.PauliZ("a"), qp.PauliX("a") @ qp.PauliZ(4), qp.PauliX("a"), qp.PauliZ(4)],
            ),
            [0, 1],
        ),
    ],
)
def test_order_wires(hamiltonian, control):
    """Test that the Qubitization operator orders the wires according to other templates."""

    op1 = qp.Qubitization(hamiltonian, control=control)
    op2 = qp.PrepSelPrep(hamiltonian, control=control)
    op3 = qp.Select(hamiltonian.terms()[1], control=control)

    assert op1.wires == op2.wires == op3.wires
