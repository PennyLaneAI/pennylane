# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the available built-in parametric qutrit operations.
"""
# pylint: disable=unnecessary-lambda-assignment, too-few-public-methods, too-many-arguments

import copy
from functools import reduce

import numpy as np
import pytest
from gate_data import TCLOCK, TSHIFT

import pennylane as qml
from pennylane import numpy as npp
from pennylane.ops.qutrit import validate_subspace
from pennylane.wires import Wires

PARAMETRIZED_OPERATIONS = [
    qml.TRX(0.123, wires=0, subspace=(0, 1)),
    qml.TRX(0.123, wires=0, subspace=(0, 2)),
    qml.TRX(0.123, wires=0, subspace=(1, 2)),
    qml.TRY(0.123, wires=0, subspace=(0, 1)),
    qml.TRY(0.123, wires=0, subspace=(0, 2)),
    qml.TRY(0.123, wires=0, subspace=(1, 2)),
    qml.TRZ(0.123, wires=0, subspace=(0, 1)),
    qml.TRZ(0.123, wires=0, subspace=(0, 2)),
    qml.TRZ(0.123, wires=0, subspace=(1, 2)),
    qml.QutritUnitary(TSHIFT, wires=0),
    qml.ControlledQutritUnitary(TCLOCK, wires=[0], control_wires=[2]),
]

BROADCASTED_OPERATIONS = [
    qml.TRX(np.array([0.142, -0.61, 2.3]), wires=0, subspace=(1, 2)),
    qml.TRY(np.array([0.142, -0.61, 2.3]), wires=0, subspace=(0, 2)),
    qml.TRZ(np.array([0.142, -0.61, 2.3]), wires=0, subspace=(0, 1)),
    qml.QutritUnitary(np.array([TSHIFT, TCLOCK]), wires=0),
    qml.ControlledQutritUnitary(np.array([TSHIFT, TCLOCK]), wires=[0], control_wires=[2]),
]

NON_PARAMETRIZED_OPERATIONS = [
    qml.TShift(wires=0),
    qml.TClock(wires=0),
    qml.TAdd(wires=[0, 1]),
    qml.TSWAP(wires=[0, 1]),
]


ALL_OPERATIONS = NON_PARAMETRIZED_OPERATIONS + PARAMETRIZED_OPERATIONS

dot_broadcasted = lambda a, b: np.einsum("...ij,...jk->...ik", a, b)
multi_dot_broadcasted = lambda matrices: reduce(dot_broadcasted, matrices)


class TestOperations:
    @pytest.mark.parametrize("op", ALL_OPERATIONS + BROADCASTED_OPERATIONS)
    def test_parametrized_op_copy(self, op, tol):
        """Tests that copied parametrized ops function as expected"""
        copied_op = copy.copy(op)
        assert np.allclose(op.matrix(), copied_op.matrix(), atol=tol)

        op = qml.adjoint(op)
        copied_op2 = copy.copy(op)
        assert np.allclose(op.matrix(), copied_op2.matrix(), atol=tol)

    @pytest.mark.parametrize("op", PARAMETRIZED_OPERATIONS)
    def test_adjoint_unitaries(self, op, tol):
        """Test that matrices of adjoint operations behave correctly"""
        op_d = op.adjoint()
        res1 = np.dot(op.matrix(), op_d.matrix())
        res2 = np.dot(op_d.matrix(), op.matrix())
        assert np.allclose(res1, np.eye(3 ** len(op.wires)), atol=tol)
        assert np.allclose(res2, np.eye(3 ** len(op.wires)), atol=tol)
        assert op.wires == op_d.wires

    @pytest.mark.parametrize("op", BROADCASTED_OPERATIONS)
    def test_adjoint_unitaries_broadcasted(self, op, tol):
        """Test that matrices of adjoint operations with broadcasting behave correctly"""
        op_d = op.adjoint()
        res1 = dot_broadcasted(op.matrix(), op_d.matrix())
        res2 = dot_broadcasted(op_d.matrix(), op.matrix())
        I = [np.eye(3 ** len(op.wires))] * op.batch_size
        assert np.allclose(res1, I, atol=tol)
        assert np.allclose(res2, I, atol=tol)
        assert op.wires == op_d.wires


class TestParameterFrequencies:
    @pytest.mark.parametrize("op", PARAMETRIZED_OPERATIONS)
    def test_parameter_frequencies_match_generator(self, op, tol):
        """Check that parameter frequencies of parametrized operations are defined correctly."""
        if not op.has_generator:
            pytest.skip(f"Operation {op.name} does not have a generator defined to test against.")

        gen = op.generator()
        mat = gen.matrix()

        gen_eigvals = np.round(np.linalg.eigvalsh(mat), 8)
        freqs_from_gen = qml.gradients.eigvals_to_frequencies(tuple(gen_eigvals))

        freqs = op.parameter_frequencies
        assert np.allclose(freqs, freqs_from_gen, atol=tol)


# TODO: Add tests for decompositions


matrix_data = [
    (qml.TRX, 0, (0, 1), np.eye(3)),
    (qml.TRX, 0, (1, 2), np.eye(3)),
    (qml.TRX, 0, (0, 2), np.eye(3)),
    (qml.TRY, 0, (0, 1), np.eye(3)),
    (qml.TRY, 0, (1, 2), np.eye(3)),
    (qml.TRY, 0, (0, 2), np.eye(3)),
    (qml.TRZ, 0, (0, 1), np.eye(3)),
    (qml.TRZ, 0, (1, 2), np.eye(3)),
    (qml.TRZ, 0, (0, 2), np.eye(3)),
    (
        qml.TRX,
        np.pi / 2,
        (0, 1),
        np.array([[1, -1j, 0], [-1j, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2),
    ),
    (
        qml.TRX,
        np.pi / 2,
        (1, 2),
        np.array([[np.sqrt(2), 0, 0], [0, 1, -1j], [0, -1j, 1]]) / np.sqrt(2),
    ),
    (
        qml.TRX,
        np.pi / 2,
        (0, 2),
        np.array([[1, 0, -1j], [0, np.sqrt(2), 0], [-1j, 0, 1]]) / np.sqrt(2),
    ),
    (
        qml.TRY,
        np.pi / 2,
        (0, 1),
        np.array([[1, -1, 0], [1, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2),
    ),
    (
        qml.TRY,
        np.pi / 2,
        (1, 2),
        np.array([[np.sqrt(2), 0, 0], [0, 1, -1], [0, 1, 1]]) / np.sqrt(2),
    ),
    (
        qml.TRY,
        np.pi / 2,
        (0, 2),
        np.array([[1, 0, -1], [0, np.sqrt(2), 0], [1, 0, 1]]) / np.sqrt(2),
    ),
    (
        qml.TRZ,
        np.pi / 2,
        (0, 1),
        np.diag(np.exp([-1j * np.pi / 4, 1j * np.pi / 4, 0])),
    ),
    (
        qml.TRZ,
        np.pi / 2,
        (1, 2),
        np.diag(np.exp([0, -1j * np.pi / 4, 1j * np.pi / 4])),
    ),
    (
        qml.TRZ,
        np.pi / 2,
        (0, 2),
        np.diag(np.exp([-1j * np.pi / 4, 0, 1j * np.pi / 4])),
    ),
    (qml.TRX, np.pi, (0, 1), -1j * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1j]])),
    (qml.TRX, np.pi, (1, 2), -1j * np.array([[1j, 0, 0], [0, 0, 1], [0, 1, 0]])),
    (qml.TRX, np.pi, (0, 2), -1j * np.array([[0, 0, 1], [0, 1j, 0], [1, 0, 0]])),
    (qml.TRY, np.pi, (0, 1), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
    (qml.TRY, np.pi, (1, 2), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])),
    (qml.TRY, np.pi, (0, 2), np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])),
    (qml.TRZ, np.pi, (0, 1), -1j * np.diag([1, -1, 1j])),
    (qml.TRZ, np.pi, (1, 2), -1j * np.diag([1j, 1, -1])),
    (qml.TRZ, np.pi, (0, 2), -1j * np.diag([1, 1j, -1])),
    (
        qml.TRX,
        np.array([np.pi / 2] * 2),
        (0, 1),
        np.tensordot(
            [1, 1], np.array([[1, -1j, 0], [-1j, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2), axes=0
        ),
    ),
    (
        qml.TRX,
        np.array([np.pi / 2] * 2),
        (1, 2),
        np.tensordot(
            [1, 1], np.array([[np.sqrt(2), 0, 0], [0, 1, -1j], [0, -1j, 1]]) / np.sqrt(2), axes=0
        ),
    ),
    (
        qml.TRX,
        np.array([np.pi / 2] * 2),
        (0, 2),
        np.tensordot(
            [1, 1], np.array([[1, 0, -1j], [0, np.sqrt(2), 0], [-1j, 0, 1]]) / np.sqrt(2), axes=0
        ),
    ),
    (
        qml.TRY,
        np.array([np.pi / 2] * 2),
        (0, 1),
        np.tensordot(
            [1, 1], np.array([[1, -1, 0], [1, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2), axes=0
        ),
    ),
    (
        qml.TRY,
        np.array([np.pi / 2] * 2),
        (1, 2),
        np.tensordot(
            [1, 1], np.array([[np.sqrt(2), 0, 0], [0, 1, -1], [0, 1, 1]]) / np.sqrt(2), axes=0
        ),
    ),
    (
        qml.TRY,
        np.array([np.pi / 2] * 2),
        (0, 2),
        np.tensordot(
            [1, 1], np.array([[1, 0, -1], [0, np.sqrt(2), 0], [1, 0, 1]]) / np.sqrt(2), axes=0
        ),
    ),
    (
        qml.TRZ,
        np.array([np.pi / 2] * 2),
        (0, 1),
        np.tensordot([1, 1], np.diag(np.exp([-1j * np.pi / 4, 1j * np.pi / 4, 0])), axes=0),
    ),
    (
        qml.TRZ,
        np.array([np.pi / 2] * 2),
        (1, 2),
        np.tensordot([1, 1], np.diag(np.exp([0, -1j * np.pi / 4, 1j * np.pi / 4])), axes=0),
    ),
    (
        qml.TRZ,
        np.array([np.pi / 2] * 2),
        (0, 2),
        np.tensordot([1, 1], np.diag(np.exp([-1j * np.pi / 4, 0, 1j * np.pi / 4])), axes=0),
    ),
]


@pytest.mark.parametrize("op, theta, subspace, expected", matrix_data)
class TestMatrix:
    """Tests for the matrix of parametrized qutrit operations."""

    def test_matrix(self, op, theta, subspace, expected, tol):
        """Test that matrices of parametric qutrit operations are correct"""
        assert np.allclose(op.compute_matrix(theta, subspace=subspace), expected, atol=tol, rtol=0)
        assert np.allclose(
            op(theta, wires=0, subspace=subspace).matrix(), expected, atol=tol, rtol=0
        )

    @pytest.mark.tf
    def test_matrix_tf(self, op, theta, subspace, expected, tol):
        """Test that compute_matrix works with tensorflow variables"""
        import tensorflow as tf

        theta = tf.Variable(theta, dtype="float64")
        expected = tf.convert_to_tensor(expected, dtype="complex128")
        assert qml.math.allclose(
            op.compute_matrix(theta, subspace=subspace), expected, atol=tol, rtol=0
        )
        assert qml.math.allclose(
            op(theta, wires=0, subspace=subspace).matrix(), expected, atol=tol, rtol=0
        )


label_data = [
    (qml.TRX(1.23456, wires=0), "TRX", "TRX\n(1.23)", "TRX\n(1)", "TRX\n(1)†"),
    (qml.TRY(1.23456, wires=0), "TRY", "TRY\n(1.23)", "TRY\n(1)", "TRY\n(1)†"),
    (qml.TRZ(1.23456, wires=0), "TRZ", "TRZ\n(1.23)", "TRZ\n(1)", "TRZ\n(1)†"),
]

label_data_broadcasted = [
    (qml.TRX(np.array([1.23, 4.56]), wires=0), "TRX", "TRX", "TRX", "TRX†"),
    (qml.TRY(np.array([1.23, 4.56]), wires=0), "TRY", "TRY", "TRY", "TRY†"),
    (qml.TRZ(np.array([1.23, 4.56]), wires=0), "TRZ", "TRZ", "TRZ", "TRZ†"),
]


class TestLabel:
    """Test the label method on parametric ops"""

    @pytest.mark.parametrize("op, label1, label2, label3, label4", label_data)
    def test_label_method(self, op, label1, label2, label3, label4):
        """Test label method with plain scalars."""

        assert op.label() == label1
        assert op.label(decimals=2) == label2
        assert op.label(decimals=0) == label3

        op = qml.adjoint(op)
        assert op.label(decimals=0) == label4

    @pytest.mark.parametrize("op, label1, label2, label3, label4", label_data_broadcasted)
    def test_label_method_broadcasted(self, op, label1, label2, label3, label4):
        """Test broadcasted label method with plain scalars."""

        assert op.label() == label1
        assert op.label(decimals=2) == label2
        assert op.label(decimals=0) == label3

        op = qml.adjoint(op)
        assert op.label(decimals=0) == label4

    @pytest.mark.tf
    def test_label_tf(self):
        """Test label methods work with tensorflow variables"""
        import tensorflow as tf

        op1 = qml.TRX(tf.Variable(0.123456), wires=0)
        assert op1.label(decimals=2) == "TRX\n(0.12)"

    @pytest.mark.torch
    def test_label_torch(self):
        """Test label methods work with torch tensors"""
        import torch

        op1 = qml.TRX(torch.tensor(1.23456), wires=0)
        assert op1.label(decimals=2) == "TRX\n(1.23)"

    @pytest.mark.jax
    def test_label_jax(self):
        """Test the label method works with jax"""
        import jax

        op1 = qml.TRX(jax.numpy.array(1.23456), wires=0)
        assert op1.label(decimals=2) == "TRX\n(1.23)"

    def test_string_parameter(self):
        """Test labelling works if variable is a string instead of a float."""

        op1 = qml.TRX("x", wires=0)
        assert op1.label() == "TRX"
        assert op1.label(decimals=0) == "TRX\n(x)"

    def test_string_parameter_broadcasted(self):
        """Test labelling works (i.e. does not raise an Error) if variable is a
        string instead of a float."""

        op1 = qml.TRX(np.array(["x0", "x1", "x2"]), wires=0)
        assert op1.label() == "TRX"
        assert op1.label(decimals=0) == "TRX"


pow_parametric_ops = (
    qml.TRX(1.234, wires=0),
    qml.TRY(1.234, wires=0),
    qml.TRZ(1.234, wires=0),
)


class TestParametricPow:
    """Test that the `pow` method works for parametric qutrit operations."""

    @pytest.mark.parametrize("op", pow_parametric_ops)
    @pytest.mark.parametrize("n", (2, -1, 0.2631, -0.987))
    def test_pow_method_parametric_ops(self, op, n):
        """Assert that a matrix raised to a power is the same as
        multiplying the data by n for relevant ops."""
        pow_op = op.pow(n)

        assert len(pow_op) == 1
        assert pow_op[0].__class__ is op.__class__
        assert all((qml.math.allclose(d1, d2 * n) for d1, d2 in zip(pow_op[0].data, op.data)))

    @pytest.mark.parametrize("op", pow_parametric_ops)
    @pytest.mark.parametrize("n", (3, -2))
    def test_pow_matrix(self, op, n):
        """Test that the matrix of an op first raised to a power is the same as the
        matrix raised to the power.  This test only can work for integer powers."""
        op_mat = qml.matrix(op)
        # Can't use qml.matrix(op.pow)(n) because qml.matrix is hardcoded to work with qubits
        # TODO: update this test once qml.matrix is updated
        pow_mat = op.pow(n)[0].matrix()

        assert qml.math.allclose(qml.math.linalg.matrix_power(op_mat, n), pow_mat)


control_data = [
    (qml.TRX(1.234, wires=0), Wires([])),
    (qml.TRY(1.234, wires=0), Wires([])),
    (qml.TRZ(1.234, wires=0), Wires([])),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test the ``control_wires`` attribute for parametrized operations."""
    assert op.control_wires == control_wires


qutrit_subspace_error_data = [
    ([1, 1], "Elements of subspace list must be unique."),
    ([1, 2, 3], "The subspace must be a sequence with"),
    ([3, 1], "Elements of the subspace must be 0, 1, or 2."),
    ([3, 3], "Elements of the subspace must be 0, 1, or 2."),
    ([1], "The subspace must be a sequence with"),
    (0, "The subspace must be a sequence with two unique"),
]


@pytest.mark.parametrize("subspace, err_msg", qutrit_subspace_error_data)
def test_qutrit_subspace_op_errors(subspace, err_msg):
    """Test that the correct errors are raised when subspace is incorrectly defined"""

    with pytest.raises(ValueError, match=err_msg):
        _ = validate_subspace(subspace)


@pytest.mark.parametrize(
    "op, obs, grad_fn",
    [
        (qml.TRX, qml.GellMann(0, 3), lambda phi: -np.sin(phi)),
        (qml.TRY, qml.GellMann(0, 1), np.cos),
        (qml.TRZ, qml.GellMann(0, 1), lambda phi: -np.sin(phi)),
    ],
)
class TestGrad:
    """Test that the gradients for qutrit parametrized operations are correct"""

    # ``default.qutrit`` doesn't currently support device, adjoint, or backprop diff methods
    diff_methods = ["parameter-shift", "finite-diff", "best", "backprop"]

    @pytest.mark.autograd
    @pytest.mark.parametrize("phi", npp.linspace(0, 2 * np.pi, 7, requires_grad=True))
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_differentiability(self, op, obs, grad_fn, phi, diff_method, tol):
        """Test that parametrized rotations are differentiable and the gradient is correct"""
        dev = qml.device("default.qutrit", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            if op is qml.TRZ:
                # Without Hadamard the derivative is always 0
                qml.THadamard(wires=0, subspace=(0, 1))
            op(phi, wires=0)
            return qml.expval(obs)

        grad = np.squeeze(qml.grad(circuit)(phi))

        assert np.isclose(grad, grad_fn(phi), atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_differentiability_broadcasted(self, op, obs, grad_fn, diff_method, tol):
        """Test that differentiation of parametrized operations with broadcasting works."""
        if diff_method in ("finite-diff", "parameter-shift"):
            pytest.xfail()

        phi = npp.linspace(0, 2 * np.pi, 7, requires_grad=True)

        dev = qml.device("default.qutrit", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            if op is qml.TRZ:
                # Without Hadamard the derivative is always 0
                qml.THadamard(wires=0, subspace=(0, 1))
            op(phi, wires=0)
            return qml.expval(obs)

        jac = qml.jacobian(circuit)(phi)

        assert np.allclose(jac, np.diag(grad_fn(phi)), atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("phi", npp.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_differentiability_jax(self, op, obs, grad_fn, phi, diff_method, tol):
        """Test that parametrized operations are differentiable with JAX and the gradient is correct"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qutrit", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            if op is qml.TRZ:
                # Without Hadamard the derivative is always 0
                qml.THadamard(wires=0, subspace=(0, 1))
            op(phi, wires=0)
            return qml.expval(obs)

        phi = jnp.array(phi)
        grad = np.squeeze(jax.grad(circuit)(phi))

        assert np.isclose(grad, grad_fn(phi), atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_differentiability_jax_broadcasted(self, op, obs, grad_fn, diff_method, tol):
        """Test that differentiation of parametrized operations in JAX with broadcasting works."""
        if diff_method in ("finite-diff", "parameter-shift"):
            pytest.xfail()

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qutrit", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            if op is qml.TRZ:
                # Without Hadamard the derivative is always 0
                qml.THadamard(wires=0, subspace=(0, 1))
            op(phi, wires=0)
            return qml.expval(obs)

        phi = jnp.linspace(0, 2 * np.pi, 7)
        jac = jax.jacobian(circuit)(phi)

        assert np.allclose(jac, np.diag(grad_fn(phi)), atol=tol, rtol=0)

    @pytest.mark.torch
    @pytest.mark.parametrize("phi", npp.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_differentiability_torch(self, op, obs, grad_fn, phi, diff_method, tol):
        """Test that parametrized operations are differentiable with Torch and the gradient is correct"""
        import torch

        dev = qml.device("default.qutrit", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            if op is qml.TRZ:
                # Without Hadamard the derivative is always 0
                qml.THadamard(wires=0, subspace=(0, 1))
            op(phi, wires=0)
            return qml.expval(obs)

        phi_torch = torch.tensor(phi, requires_grad=True, dtype=torch.float64)
        grad = torch.autograd.grad(circuit(phi_torch), phi_torch)

        assert qml.math.isclose(grad, grad_fn(phi), atol=tol, rtol=0)

    @pytest.mark.torch
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_differentiability_torch_broadcasted(self, op, obs, grad_fn, diff_method, tol):
        """Test that differentiation of parametrized operations in Torch with broadcasting works."""
        if diff_method in ("finite-diff", "parameter-shift"):
            pytest.xfail()

        import torch

        dev = qml.device("default.qutrit", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            if op is qml.TRZ:
                # Without Hadamard the derivative is always 0
                qml.THadamard(wires=0, subspace=(0, 1))
            op(phi, wires=0)
            return qml.expval(obs)

        phi_torch = torch.linspace(0, 2 * np.pi, 7, requires_grad=True, dtype=torch.float64)
        jac = torch.autograd.functional.jacobian(circuit, phi_torch)
        phi = phi_torch.detach().numpy()

        assert qml.math.allclose(jac, np.diag(grad_fn(phi)), atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("phi", npp.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_differentiability_tf(self, op, obs, grad_fn, phi, diff_method, tol):
        """Test that parametrized operations are differentiable with TensorFlow and the gradient is correct"""
        import tensorflow as tf

        dev = qml.device("default.qutrit", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            if op is qml.TRZ:
                # Without Hadamard the derivative is always 0
                qml.THadamard(wires=0, subspace=(0, 1))
            op(phi, wires=0)
            return qml.expval(obs)

        phi_tf = tf.Variable(phi)

        with tf.GradientTape() as tape:
            result = circuit(phi_tf)
        res = tape.gradient(result, phi_tf)

        assert qml.math.isclose(res, grad_fn(phi), atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_differentiability_tf_broadcasted(self, op, obs, grad_fn, diff_method, tol):
        """Test that differentiation of parametrized operations in TensorFlow with broadcasting works."""
        if diff_method in ("finite-diff", "parameter-shift"):
            pytest.xfail()

        import tensorflow as tf

        dev = qml.device("default.qutrit", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            if op is qml.TRZ:
                # Without Hadamard the derivative is always 0
                qml.THadamard(wires=0, subspace=(0, 1))
            op(phi, wires=0)
            return qml.expval(obs)

        phi = np.linspace(0, 2 * np.pi, 7)
        phi_tf = tf.Variable(phi)
        with tf.GradientTape() as tape:
            result = circuit(phi_tf)
        res = tape.jacobian(result, phi_tf)
        expected = tf.Variable(np.diag(grad_fn(phi)))

        assert qml.math.allclose(res, expected, atol=tol, rtol=0)
