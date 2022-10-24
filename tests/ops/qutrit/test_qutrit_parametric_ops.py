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
from functools import reduce
import pytest
import copy
import numpy as np
from pennylane import numpy as npp

import pennylane as qml
from pennylane.wires import Wires
from tests.ops.qubit.test_parametric_ops import NON_PARAMETRIZED_OPERATIONS

from gate_data import TSHIFT, TCLOCK

PARAMETRIZED_OPERATIONS = [
    qml.TRX(0.123, wires=0, subspace=[1, 2]),
    qml.QutritUnitary(TSHIFT, wires=0),
    qml.ControlledQutritUnitary(TCLOCK, wires=[0], control_wires=[2]),
]

BROADCASTED_OPERATIONS = [
    qml.TRX(np.array([0.142, -0.61, 2.3]), wires=0, subspace=[1, 2]),
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

        op.inv()
        copied_op2 = copy.copy(op)
        assert np.allclose(op.matrix(), copied_op2.matrix(), atol=tol)
        op.inv()

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
        if not qml.operation.has_gen(op):
            pytest.skip(f"Operation {op.name} does not have a generator defined to test against.")

        gen = op.generator()

        try:
            mat = gen.matrix()
        except (AttributeError, qml.operation.MatrixUndefinedError):

            if isinstance(gen, qml.Hamiltonian):
                mat = qml.utils.sparse_hamiltonian(gen, level=3).toarray()
            elif isinstance(gen, qml.SparseHamiltonian):
                mat = gen.sparse_matrix().toarray()
            else:
                pytest.skip(f"Operation {op.name}'s generator does not define a matrix.")

        gen_eigvals = np.round(np.linalg.eigvalsh(mat), 8)
        freqs_from_gen = qml.gradients.eigvals_to_frequencies(tuple(gen_eigvals))

        freqs = op.parameter_frequencies
        assert np.allclose(freqs, freqs_from_gen, atol=tol)


# TODO: Add tests for decompositions


class TestMatrix:
    def test_trx(self, tol):
        """Test that qutrit TRX rotation is correct"""

        # test identity for theta = 0
        expected = np.eye(3)
        assert np.allclose(qml.TRX.compute_matrix(0, subspace=[0, 1]), expected, atol=tol, rtol=0)
        assert np.allclose(qml.TRX.compute_matrix(0, subspace=[1, 2]), expected, atol=tol, rtol=0)
        assert np.allclose(qml.TRX.compute_matrix(0, subspace=[0, 2]), expected, atol=tol, rtol=0)
        assert np.allclose(
            qml.TRX(0, wires=0, subspace=[0, 1]).matrix(), expected, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.TRX(0, wires=0, subspace=[1, 2]).matrix(), expected, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.TRX(0, wires=0, subspace=[0, 2]).matrix(), expected, atol=tol, rtol=0
        )

        # test identity for theta=pi/2
        expected = np.array([[1, -1j, 0], [-1j, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi / 2, subspace=[0, 1]), expected, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.TRX(np.pi / 2, wires=0, subspace=[0, 1]).matrix(), expected, atol=tol, rtol=0
        )

        expected = np.array([[np.sqrt(2), 0, 0], [0, 1, -1j], [0, -1j, 1]]) / np.sqrt(2)
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi / 2, subspace=[1, 2]), expected, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.TRX(np.pi / 2, wires=0, subspace=[1, 2]).matrix(), expected, atol=tol, rtol=0
        )

        expected = np.array([[1, 0, -1j], [0, np.sqrt(2), 0], [-1j, 0, 1]]) / np.sqrt(2)
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi / 2, subspace=[0, 2]), expected, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.TRX(np.pi / 2, wires=0, subspace=[0, 2]).matrix(), expected, atol=tol, rtol=0
        )

        # test identity for broadcasted theta=pi/2
        pi_half = np.array([np.pi / 2] * 2)
        expected = np.tensordot(
            [1, 1], np.array([[1, -1j, 0], [-1j, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2), axes=0
        )
        assert np.allclose(
            qml.TRX.compute_matrix(pi_half, subspace=[0, 1]), expected, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.TRX(pi_half, wires=0, subspace=[0, 1]).matrix(), expected, atol=tol, rtol=0
        )

        expected = np.tensordot(
            [1, 1], np.array([[1, 0, -1j], [0, np.sqrt(2), 0], [-1j, 0, 1]]) / np.sqrt(2), axes=0
        )
        assert np.allclose(
            qml.TRX.compute_matrix(pi_half, subspace=[0, 2]), expected, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.TRX(pi_half, wires=0, subspace=[0, 2]).matrix(), expected, atol=tol, rtol=0
        )

        expected = np.tensordot(
            [1, 1], np.array([[np.sqrt(2), 0, 0], [0, 1, -1j], [0, -1j, 1]]) / np.sqrt(2), axes=0
        )
        assert np.allclose(
            qml.TRX.compute_matrix(pi_half, subspace=[1, 2]), expected, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.TRX(pi_half, wires=0, subspace=[1, 2]).matrix(), expected, atol=tol, rtol=0
        )

        # test identity for theta=pi
        expected = -1j * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1j]])
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi, subspace=[0, 1]), expected, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.TRX(np.pi, wires=0, subspace=[0, 1]).matrix(), expected, atol=tol, rtol=0
        )

        expected = -1j * np.array([[1j, 0, 0], [0, 0, 1], [0, 1, 0]])
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi, subspace=[1, 2]), expected, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.TRX(np.pi, wires=0, subspace=[1, 2]).matrix(), expected, atol=tol, rtol=0
        )

        expected = -1j * np.array([[0, 0, 1], [0, 1j, 0], [1, 0, 0]])
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi, subspace=[0, 2]), expected, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.TRX(np.pi, wires=0, subspace=[0, 2]).matrix(), expected, atol=tol, rtol=0
        )


# TODO: Add tests for grad

label_data = [
    (qml.TRX(1.23456, wires=0), "TRX", "TRX\n(1.23)", "TRX\n(1)", "TRX⁻¹\n(1)"),
]

label_data_broadcasted = [
    (qml.TRX(np.array([1.23, 4.56]), wires=0), "TRX", "TRX", "TRX", "TRX⁻¹"),
]


class TestLabel:
    """Test the label method on parametric ops"""

    @pytest.mark.parametrize("op, label1, label2, label3, label4", label_data)
    def test_label_method(self, op, label1, label2, label3, label4):
        """Test label method with plain scalars."""

        assert op.label() == label1
        assert op.label(decimals=2) == label2
        assert op.label(decimals=0) == label3

        op.inv()
        assert op.label(decimals=0) == label4
        op.inv()

    @pytest.mark.parametrize("op, label1, label2, label3, label4", label_data_broadcasted)
    def test_label_method_broadcasted(self, op, label1, label2, label3, label4):
        """Test broadcasted label method with plain scalars."""

        assert op.label() == label1
        assert op.label(decimals=2) == label2
        assert op.label(decimals=0) == label3

        op.inv()
        assert op.label(decimals=0) == label4
        op.inv()

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


pow_parametric_ops = (qml.TRX(1.234, wires=0),)


class TestParametricPow:
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
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test the ``control_wires`` attribute for parametrized operations."""
    assert op.control_wires == control_wires
