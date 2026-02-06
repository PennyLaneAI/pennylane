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
"""Tests for the SProd class representing the product of an operator by a scalar"""

from copy import copy

import gate_data as gd  # a file containing matrix rep of each gate
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import pennylane as qp
import pennylane.numpy as qnp
from pennylane import math
from pennylane.exceptions import DecompositionUndefinedError, MatrixUndefinedError
from pennylane.ops.op_math import Prod, SProd, Sum, s_prod
from pennylane.wires import Wires

scalars = (1, 1.23, 0.0, 1 + 2j)  # int, float, zero, and complex cases accounted for

no_mat_ops = (
    qp.Barrier,
    qp.WireCut,
)

non_param_ops = (
    (qp.Identity, gd.I),
    (qp.Hadamard, gd.H),
    (qp.PauliX, gd.X),
    (qp.PauliY, gd.Y),
    (qp.PauliZ, gd.Z),
    (qp.S, gd.S),
    (qp.T, gd.T),
    (qp.SX, gd.SX),
    (qp.CNOT, gd.CNOT),
    (qp.CZ, gd.CZ),
    (qp.CY, gd.CY),
    (qp.SWAP, gd.SWAP),
    (qp.ISWAP, gd.ISWAP),
    (qp.SISWAP, gd.SISWAP),
    (qp.CSWAP, gd.CSWAP),
    (qp.Toffoli, gd.Toffoli),
)

param_ops = (
    (qp.RX, gd.Rotx),
    (qp.RY, gd.Roty),
    (qp.RZ, gd.Rotz),
    (qp.PhaseShift, gd.Rphi),
    (qp.Rot, gd.Rot3),
    (qp.U1, gd.U1),
    (qp.U2, gd.U2),
    (qp.U3, gd.U3),
    (qp.CRX, gd.CRotx),
    (qp.CRY, gd.CRoty),
    (qp.CRZ, gd.CRotz),
    (qp.CRot, gd.CRot3),
    (qp.IsingXX, gd.IsingXX),
    (qp.IsingYY, gd.IsingYY),
    (qp.IsingZZ, gd.IsingZZ),
)

ops = (
    (1.0, qp.PauliX(wires=0)),
    (0.0, qp.PauliZ(wires=0)),
    (1j, qp.Hadamard(wires=0)),
    (1.23, qp.CNOT(wires=[0, 1])),
    (4.56, qp.RX(1.23, wires=1)),
    (1.0 + 2.0j, qp.Identity(wires=0)),
    (10, qp.IsingXX(4.56, wires=[2, 3])),
    (0j, qp.Toffoli(wires=[1, 2, 3])),
    (42, qp.Rot(0.34, 1.0, 0, wires=0)),
)

ops_rep = (
    "1.0 * X(0)",
    "0.0 * Z(0)",
    "1j * H(0)",
    "1.23 * CNOT(wires=[0, 1])",
    "4.56 * RX(1.23, wires=[1])",
    "(1+2j) * I(0)",
    "10 * IsingXX(4.56, wires=[2, 3])",
    "0j * Toffoli(wires=[1, 2, 3])",
    "42 * Rot(0.34, 1.0, 0, wires=[0])",
)


class TestInitialization:
    """Test initialization of the SProd Class."""

    @pytest.mark.parametrize("test_id", ("foo", "bar"))
    def test_init_sprod_op(self, test_id):
        sprod_op = s_prod(3.14, qp.RX(0.23, wires="a"), id=test_id)

        # no need to test if op.base == RX since this is covered in SymbolicOp tests
        assert sprod_op.scalar == 3.14
        assert sprod_op.wires == Wires(("a",))
        assert sprod_op.num_wires == 1
        assert sprod_op.name == "SProd"
        assert sprod_op.id == test_id

        assert sprod_op.data == (3.14, 0.23)
        assert sprod_op.parameters == [3.14, 0.23]
        assert sprod_op.num_params == 2

    def test_parameters(self):
        sprod_op = s_prod(9.87, qp.Rot(1.23, 4.0, 5.67, wires=1))
        assert sprod_op.parameters == [9.87, 1.23, 4.0, 5.67]

    def test_data(self):
        sprod_op = s_prod(9.87, qp.Rot(1.23, 4.0, 5.67, wires=1))
        assert sprod_op.data == (9.87, 1.23, 4.0, 5.67)

    def test_data_setter(self):
        """Test the setter method for data"""
        scalar, angles = (9.87, (1.23, 4.0, 5.67))
        old_data = (9.87, 1.23, 4.0, 5.67)

        sprod_op = s_prod(scalar, qp.Rot(*angles, wires=1))
        assert sprod_op.data == old_data

        new_data = (1.23, 0.0, -1.0, -2.0)
        sprod_op.data = new_data
        assert sprod_op.data == new_data
        assert sprod_op.scalar == new_data[0]
        assert sprod_op.base.data == new_data[1:]

    def test_data_setter_shallow(self):
        """Test the setter method for data with a non-parametric base op."""
        op = s_prod(0.1, qp.PauliX(0))
        op.data = (0.2,)
        assert op.data == (0.2,) == (op.scalar,)

    def test_data_setter_deep(self):
        """Test the setter method for data with a deep base operator."""
        op = s_prod(0.1, qp.sum(qp.PauliX(0), qp.prod(qp.PauliY(0), qp.RX(0.2, 1))))
        assert op.data == (0.1, 0.2)

        new_data = (0.3, 0.4)
        op.data = new_data
        assert op.data == new_data
        assert op.scalar == 0.3
        assert op.base[1].data == (0.4,)
        assert op.base[1][1].data == (0.4,)

    @pytest.mark.parametrize("scalar, op", ops)
    def test_terms(self, op, scalar):
        sprod_op = SProd(scalar, op)
        coeff, op2 = sprod_op.terms()

        assert coeff == [scalar]
        for op1, op2 in zip(op2, [op]):
            qp.assert_equal(op1, op2)

    @pytest.mark.parametrize(
        "sprod_op, coeffs_exp, ops_exp",
        [
            (qp.s_prod(1.23, qp.sum(qp.X(0), qp.Y(0))), [1.23, 1.23], [qp.X(0), qp.Y(0)]),
            (
                qp.s_prod(1.23, qp.Hamiltonian([0.1, 0.2], [qp.X(0), qp.Y(0)])),
                [0.123, 0.246],
                [qp.X(0), qp.Y(0)],
            ),
        ],
    )
    def test_terms_nested(self, sprod_op, coeffs_exp, ops_exp):
        """Tests that SProd.terms() flattens a nested structure."""
        coeffs, ops_actual = sprod_op.terms()
        assert coeffs == coeffs_exp
        for op1, op2 in zip(ops_actual, ops_exp):
            qp.assert_equal(op1, op2)

    def test_diagonalizing_gates(self):
        """Test that the diagonalizing gates are correct."""
        diag_sprod_op = SProd(1.23, qp.PauliX(wires=0))
        diagonalizing_gates = diag_sprod_op.diagonalizing_gates()[0].matrix()
        true_diagonalizing_gates = (
            qp.PauliX(wires=0).diagonalizing_gates()[0].matrix()
        )  # scaling doesn't change diagonalizing gates

        assert np.allclose(diagonalizing_gates, true_diagonalizing_gates)

    def test_base_gets_cast_to_new_type(self):
        """Test that Tensor and Hamiltonian instances get cast to new types."""
        base_H = qp.Hamiltonian([1.1, 2.2], [qp.PauliZ(0), qp.PauliZ(1)])
        op_H = qp.s_prod(2j, base_H)
        assert isinstance(op_H.base, Sum)

        base_T = qp.PauliZ(0) @ qp.PauliZ(1)
        op_T = qp.s_prod(2j, base_T)
        assert isinstance(op_T.base, Prod)


class TestMscMethods:
    """Test miscellaneous methods of the SProd class."""

    def test_string_with_single_pauli(self):
        """Test the string representation with single pauli"""
        res = qp.s_prod(0.5, qp.PauliX("a"))
        true_res = "0.5 * X('a')"
        assert repr(res) == true_res

        res = qp.s_prod(0.5, qp.PauliX(0))
        true_res = "0.5 * X(0)"
        assert repr(res) == true_res

    def test_string_with_sum_of_pauli(self):
        """Test the string representation with single pauli"""
        res = qp.s_prod(0.5, qp.sum(qp.PauliX("a"), qp.PauliX("b")))
        true_res = "0.5 * (X('a') + X('b'))"
        assert repr(res) == true_res

        res = qp.s_prod(0.5, qp.sum(qp.PauliX(0), qp.PauliX(1)))
        true_res = "0.5 * (X(0) + X(1))"
        assert repr(res) == true_res

        res = qp.s_prod(0.5, qp.sum(qp.PauliX("a"), qp.PauliX(1)))
        true_res = "0.5 * (X('a') + X(1))"
        assert repr(res) == true_res

    @pytest.mark.parametrize("op_scalar_tup, op_rep", tuple((i, j) for i, j in zip(ops, ops_rep)))
    def test_repr(self, op_scalar_tup, op_rep):
        """Test the repr dunder method."""
        scalar, op = op_scalar_tup
        sprod_op = SProd(scalar, op)
        assert op_rep == repr(sprod_op)

    # pylint: disable=protected-access
    @pytest.mark.parametrize("op_scalar_tup", ops)
    def test_flatten_unflatten(self, op_scalar_tup):
        scalar, op = op_scalar_tup
        sprod_op = SProd(scalar, op)

        data, metadata = sprod_op._flatten()

        assert len(data) == 2
        assert data[0] == scalar
        assert data[1] is op

        assert metadata == tuple()

        new_op = type(sprod_op)._unflatten(*sprod_op._flatten())
        qp.assert_equal(new_op, sprod_op)
        assert new_op is not sprod_op

    @pytest.mark.parametrize("op_scalar_tup", ops)
    def test_copy(self, op_scalar_tup):
        """Test the copy dunder method properly copies the operator."""
        scalar, op = op_scalar_tup
        sprod_op = SProd(scalar, op, id="something")
        copied_op = copy(sprod_op)

        assert sprod_op.scalar == copied_op.scalar

        assert sprod_op.id == copied_op.id
        assert sprod_op.data == copied_op.data
        assert sprod_op.wires == copied_op.wires

        assert sprod_op.base.name == copied_op.base.name
        assert sprod_op.base.wires == copied_op.base.wires
        assert sprod_op.base.data == copied_op.base.data

    def test_has_matrix_true_via_factor_has_matrix(self):
        """Test that a scalar product with an operator that has `has_matrix=True`
        has `has_matrix=True` as well."""

        sprod_op = SProd(0.7, qp.RZ(0.23, wires="a"))
        assert sprod_op.has_matrix is True

    def test_has_matrix_true_via_factor_has_no_matrix_but_is_hamiltonian(self):
        """Test that a scalar product with an operator that has `has_matrix=False`
        but is a Hamiltonian has `has_matrix=True`."""

        H = qp.Hamiltonian([0.5], [qp.PauliX(wires=1)])
        sprod_op = SProd(0.6, H)
        assert sprod_op.has_matrix is True

    def test_has_matrix_false_via_factor_has_no_matrix(self):
        """Test that a scalar product with an operator that has `has_matrix=False`
        has `has_matrix=True` as well."""

        # pylint: disable=too-few-public-methods
        class MyOp(qp.RX):
            """Variant of qp.RX that claims to not have `adjoint` or a matrix defined."""

            has_matrix = False

        sprod_op = SProd(0.4, MyOp(0.23, wires="a"))
        assert sprod_op.has_matrix is False

    @pytest.mark.parametrize("value", (True, False))
    def test_has_diagonalizing_gates(self, value):
        """Test that SProd defers has_diagonalizing_gates to base operator."""

        # pylint: disable=too-few-public-methods
        class DummyOp(qp.operation.Operator):
            num_wires = 1
            has_diagonalizing_gates = value

        op = SProd(0.21319, DummyOp(1))
        assert op.has_diagonalizing_gates is value


class TestDecomposition:

    def test_decomposition_coeff_norm_1(self):
        """Test that a decomposition exists when the coefficient is of norm 1."""

        op = qp.s_prod(1j, qp.X(0))
        assert op.has_decomposition
        decomp = op.decomposition()

        assert len(decomp) == 2
        qp.assert_equal(decomp[0], qp.GlobalPhase(-np.pi / 2))
        qp.assert_equal(decomp[1], qp.X(0))

        decomp_mat = qp.matrix(op.decomposition, wire_order=[0])()
        assert qp.math.allclose(decomp_mat, 1j * qp.X.compute_matrix())

        with qp.queuing.AnnotatedQueue() as q:
            op.decomposition()

        assert len(q) == 2
        qp.assert_equal(q.queue[0], qp.GlobalPhase(-np.pi / 2))
        qp.assert_equal(q.queue[1], qp.X(0))

    @pytest.mark.jax
    def test_no_decomposition_abstract_coeff(self):
        """Test that no decomposition exists when the coeff is abstract."""

        import jax

        def f(coeff):
            op = qp.s_prod(coeff, qp.X(0))
            assert not op.has_decomposition
            with pytest.raises(DecompositionUndefinedError):
                op.decomposition()

        jax.jit(f)(1.0)

    def test_no_decomposition_norm_not_one(self):
        """Test that no decomposition exists if the norm is not 1."""

        op = qp.s_prod(2, qp.X(0))
        assert not op.has_decomposition
        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()


class TestMatrix:
    """Tests of the matrix of a SProd class."""

    def test_base_batching_support(self):
        """Test that SProd matrix has base batching support."""
        x = np.array([-1, -2, -3])
        op = qp.s_prod(3, qp.RX(x, 0))
        mat = op.matrix()
        true_mat = qp.math.stack([qp.s_prod(3, qp.RX(i, 0)).matrix() for i in x])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    def test_coeff_batching_support(self):
        """Test that SProd matrix has coeff batching support."""
        x = np.array([-1, -2, -3])
        op = qp.s_prod(x, qp.PauliX(0))
        mat = op.matrix()
        true_mat = qp.math.stack([qp.s_prod(i, qp.PauliX(0)).matrix() for i in x])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    def test_base_and_coeff_batching_support(self):
        """Test that SProd matrix has base and coeff batching support."""
        x = np.array([-1, -2, -3])
        y = np.array([1, 2, 3])
        op = qp.s_prod(y, qp.RX(x, 0))
        mat = op.matrix()
        true_mat = qp.math.stack([qp.s_prod(j, qp.RX(i, 0)).matrix() for i, j in zip(x, y)])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    @pytest.mark.jax
    def test_batching_jax(self):
        """Test that SProd matrix has batching support with the jax interface."""
        import jax.numpy as jnp

        x = jnp.array([-1, -2, -3])
        y = jnp.array([1, 2, 3])
        op = qp.s_prod(y, qp.RX(x, 0))
        mat = op.matrix()
        true_mat = qp.math.stack([qp.s_prod(j, qp.RX(i, 0)).matrix() for i, j in zip(x, y)])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)
        assert isinstance(mat, jnp.ndarray)

    @pytest.mark.torch
    def test_batching_torch(self):
        """Test that SProd matrix has batching support with the torch interface."""
        import torch

        x = torch.tensor([-1, -2, -3])
        y = torch.tensor([1, 2, 3])
        op = qp.s_prod(y, qp.RX(x, 0))
        mat = op.matrix()
        true_mat = qp.math.stack([qp.s_prod(j, qp.RX(i, 0)).matrix() for i, j in zip(x, y)])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)
        assert isinstance(mat, torch.Tensor)

    @pytest.mark.tf
    def test_batching_tf(self):
        """Test that SProd matrix has batching support with the tensorflow interface."""
        import tensorflow as tf

        x = tf.constant([-1.0, -2.0, -3.0])
        y = tf.constant([1.0, 2.0, 3.0])
        op = qp.s_prod(y, qp.RX(x, 0))
        mat = op.matrix()
        true_mat = qp.math.stack([qp.s_prod(j, qp.RX(i, 0)).matrix() for i, j in zip(x, y)])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)
        assert isinstance(mat, tf.Tensor)

    @pytest.mark.parametrize("scalar", scalars)
    @pytest.mark.parametrize("op, mat", param_ops + non_param_ops)
    def test_various_ops(self, scalar, op, mat):
        """Test matrix method for a scalar product of parametric ops"""
        params = range(op.num_params)

        sprod_op = SProd(
            scalar, op(*params, wires=0 if op.num_wires is None else range(op.num_wires))
        )
        sprod_mat = sprod_op.matrix()

        true_mat = scalar * mat(*params) if op.num_params > 0 else scalar * mat
        assert np.allclose(sprod_mat, true_mat)

    @pytest.mark.parametrize("op", no_mat_ops)
    def test_error_no_mat(self, op):
        """Test that an error is raised if the operator doesn't
        have its matrix method defined."""
        sprod_op = SProd(1.23, op(wires=0))
        with pytest.raises(MatrixUndefinedError):
            sprod_op.matrix()

    def test_sprod_ops_wire_order(self):
        """Test correct matrix is returned when the wire_order arg is provided."""
        scalar = 1.23
        sprod_op = SProd(scalar, qp.Toffoli(wires=[2, 0, 1]))
        wire_order = [0, 1, 2]
        mat = sprod_op.matrix(wire_order=wire_order)

        ccnot = math.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )

        true_mat = scalar * ccnot
        assert np.allclose(mat, true_mat)

    templates_and_mats = (
        (qp.QFT(wires=[0, 1, 2]), qp.QFT(wires=[0, 1, 2]).compute_matrix(3)),
        (
            qp.GroverOperator(wires=[0, 1, 2]),
            qp.GroverOperator(wires=[0, 1, 2]).compute_matrix(3, range(3)),
        ),
    )

    @pytest.mark.parametrize("template, mat", templates_and_mats)
    def test_sprod_templates(self, template, mat):
        """Test that we can scale templates and the generated matrix is correct."""
        scalar = 3.14
        sprod_op = SProd(scalar, template)

        expected_mat = sprod_op.matrix()
        true_mat = scalar * mat
        assert np.allclose(expected_mat, true_mat)

    def test_sprod_qchem_ops(self):
        """Test that we can scale qchem operations and the generated matrix is correct."""
        wires = [0, 1, 2, 3]
        sprod_op1 = SProd(1.23, qp.OrbitalRotation(4.56, wires=wires))
        sprod_op2 = SProd(3.45, qp.SingleExcitation(1.23, wires=[0, 1]))
        mat1 = sprod_op1.matrix()
        mat2 = sprod_op2.matrix()

        or_mat = 1.23 * gd.OrbitalRotation(4.56)
        se_mat = 3.45 * gd.SingleExcitation(1.23)

        assert np.allclose(mat1, or_mat)
        assert np.allclose(mat2, se_mat)

    def test_sprod_observables(self):
        """Test that observable objects can also be scaled with correct matrix representation."""
        wires = [0, 1]
        sprod_op1 = SProd(1.23, qp.Projector(state=qnp.array([0, 1]), wires=wires))
        sprod_op2 = SProd(3.45, qp.Hermitian(qnp.array([[0.0, 1.0], [1.0, 0.0]]), wires=0))
        mat1 = sprod_op1.matrix()
        mat2 = sprod_op2.matrix()

        her_mat = 3.45 * qnp.array([[0.0, 1.0], [1.0, 0.0]])
        proj_mat = 1.23 * qnp.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )

        assert np.allclose(mat1, proj_mat)
        assert np.allclose(mat2, her_mat)

    def test_sprod_qubit_unitary(self):
        """Test that an arbitrary QubitUnitary can be scaled with correct matrix representation."""
        U = 1 / qnp.sqrt(2) * qnp.array([[1, 1], [1, -1]])  # Hadamard
        U_op = qp.QubitUnitary(U, wires=0)

        sprod_op = SProd(42, U_op)
        mat = sprod_op.matrix()

        true_mat = 42 * U
        assert np.allclose(mat, true_mat)

    def test_sprod_hamiltonian(self):
        """Test that a hamiltonian object can be scaled."""
        U = qp.Hamiltonian([0.5], [qp.PauliX(wires=0)])
        sprod_op = SProd(-4, U)
        mat = sprod_op.matrix()

        true_mat = [[0, -2], [-2, 0]]
        assert np.allclose(mat, true_mat)

    # Add interface tests for each interface !

    @pytest.mark.jax
    def test_sprod_jax(self):
        """Test matrix is cast correctly using jax parameters."""
        import jax.numpy as jnp

        coeff = jnp.array(1.23)
        rot_params = jnp.array([0.12, 3.45, 6.78])

        sprod_op = SProd(coeff, qp.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0))
        mat = sprod_op.matrix()

        true_mat = 1.23 * gd.Rot3(rot_params[0], rot_params[1], rot_params[2])
        true_mat = jnp.array(true_mat)

        assert jnp.allclose(mat, true_mat)

    @pytest.mark.torch
    def test_sprod_torch(self):
        """Test matrix is cast correctly using torch parameters."""
        import torch

        coeff = torch.tensor(1.23)
        rot_params = torch.tensor([0.12, 3.45, 6.78])

        sprod_op = SProd(coeff, qp.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0))
        mat = sprod_op.matrix()

        true_mat = 1.23 * gd.Rot3(rot_params[0], rot_params[1], rot_params[2])
        true_mat = torch.tensor(true_mat, dtype=torch.complex64)

        assert torch.allclose(mat, true_mat)

    @pytest.mark.tf
    def test_sprod_tf(self):
        """Test matrix is cast correctly using tf parameters."""
        import tensorflow as tf

        coeff = tf.Variable(1.23, dtype=tf.complex128)
        raw_rot_params = [0.12, 3.45, 6.78]
        rot_params = tf.Variable(raw_rot_params)

        sprod_op = SProd(coeff, qp.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0))
        mat = sprod_op.matrix()

        true_mat = 1.23 * gd.Rot3(raw_rot_params[0], raw_rot_params[1], raw_rot_params[2])
        true_mat = tf.Variable(true_mat, dtype=tf.complex128)

        assert isinstance(mat, tf.Tensor)
        assert mat.dtype == true_mat.dtype
        assert np.allclose(mat, true_mat)

    @pytest.mark.tf
    def test_tf_matrix_type_casting(self):
        """Test that types for the matrix are always converted to complex128 and parameters aren't truncated."""
        import tensorflow as tf

        coeff = tf.Variable(0.1)
        op = qp.PauliX(0)

        sprod_op = SProd(coeff, op)
        mat = sprod_op.matrix()

        assert mat.dtype == tf.complex128
        expected = np.array([[0, 0.1], [0.1, 0.0]], dtype="complex128")
        assert qp.math.allclose(mat, expected)
        assert sprod_op.data[0].dtype == coeff.dtype  # coeff not modified by calling the matrix

        op = qp.PauliY(0)

        sprod_op = SProd(coeff, op)
        mat = sprod_op.matrix()

        assert mat.dtype == tf.complex128
        expected = np.array([[0, -0.1j], [0.1j, 0.0]], dtype="complex128")
        assert qp.math.allclose(mat, expected)
        assert sprod_op.data[0].dtype == coeff.dtype  # coeff not modified by calling the matrix


class TestSparseMatrix:
    sparse_ops = (
        qp.Identity(wires=0),
        qp.PauliX(wires=0),
        qp.PauliY(wires=0),
        qp.PauliZ(wires=0),
        qp.Hadamard(wires=0),
    )

    @pytest.mark.parametrize("scalar", scalars)
    @pytest.mark.parametrize("op", sparse_ops)
    def test_sparse_matrix(self, scalar, op):
        """Test the sparse_matrix representation of scaled ops."""
        sprod_op = SProd(scalar, op)
        sparse_matrix = sprod_op.sparse_matrix()
        sparse_matrix.sort_indices()

        expected_sparse_matrix = csr_matrix(op.matrix()).multiply(scalar)
        expected_sparse_matrix.sort_indices()
        expected_sparse_matrix.eliminate_zeros()

        assert isinstance(sparse_matrix, type(expected_sparse_matrix))
        assert all(sparse_matrix.data == expected_sparse_matrix.data)
        assert all(sparse_matrix.indices == expected_sparse_matrix.indices)

    @pytest.mark.parametrize("scalar", scalars)
    @pytest.mark.parametrize("op", sparse_ops)
    def test_sparse_matrix_format(self, scalar, op):
        """Test that the sparse matrix accepts the format parameter."""
        from scipy.sparse import coo_matrix, csc_matrix, lil_matrix

        sprod_op = SProd(scalar, op)
        expected_sparse_matrix = csr_matrix(op.matrix()).multiply(scalar)
        expected_sparse_matrix.sort_indices()
        expected_sparse_matrix.eliminate_zeros()

        assert isinstance(sprod_op.sparse_matrix(), csr_matrix)
        sprod_op_csc = sprod_op.sparse_matrix(format="csc")
        sprod_op_lil = sprod_op.sparse_matrix(format="lil")
        sprod_op_coo = sprod_op.sparse_matrix(format="coo")
        assert isinstance(sprod_op_csc, csc_matrix)
        assert isinstance(sprod_op_lil, lil_matrix)
        assert isinstance(sprod_op_coo, coo_matrix)

    @pytest.mark.jax
    @pytest.mark.parametrize("scalar", scalars)
    @pytest.mark.parametrize("op", sparse_ops)
    def test_sparse_matrix_jax_scalar(self, scalar, op):
        """Test the sparse_matrix representation of scaled ops when scalar is a jax array."""
        import jax.numpy as jnp

        scalar = jnp.array(scalar)
        sprod_op = SProd(scalar, op)
        sparse_matrix = sprod_op.sparse_matrix()
        sparse_matrix.sort_indices()

        expected_sparse_matrix = csr_matrix(op.matrix()).multiply(scalar)
        expected_sparse_matrix.sort_indices()
        expected_sparse_matrix.eliminate_zeros()

        assert isinstance(sparse_matrix, type(expected_sparse_matrix))
        assert all(sparse_matrix.data == expected_sparse_matrix.data)
        assert all(sparse_matrix.indices == expected_sparse_matrix.indices)

    @pytest.mark.torch
    @pytest.mark.parametrize("scalar", scalars)
    @pytest.mark.parametrize("op", sparse_ops)
    def test_sparse_matrix_torch_scalar(self, scalar, op):
        """Test the sparse_matrix representation of scaled ops when scalar is a torch tensor."""
        import torch

        scalar = torch.tensor(scalar)
        sprod_op = SProd(scalar, op)
        sparse_matrix = sprod_op.sparse_matrix()
        sparse_matrix.sort_indices()

        expected_sparse_matrix = csr_matrix(op.matrix()).multiply(scalar)
        expected_sparse_matrix.sort_indices()
        expected_sparse_matrix.eliminate_zeros()

        assert isinstance(sparse_matrix, type(expected_sparse_matrix))
        assert all(sparse_matrix.data == expected_sparse_matrix.data)
        assert all(sparse_matrix.indices == expected_sparse_matrix.indices)

    @pytest.mark.tf
    @pytest.mark.parametrize("scalar", scalars)
    @pytest.mark.parametrize("op", sparse_ops)
    def test_sparse_matrix_tf_scalar(self, scalar, op):
        """Test the sparse_matrix representation of scaled ops when scalar is a tf Variable."""
        import tensorflow as tf

        scalar = tf.Variable(scalar)
        sprod_op = SProd(scalar, op)
        sparse_matrix = sprod_op.sparse_matrix()
        sparse_matrix.sort_indices()

        expected_sparse_matrix = csr_matrix(op.matrix()).multiply(scalar)
        expected_sparse_matrix.sort_indices()
        expected_sparse_matrix.eliminate_zeros()

        assert isinstance(sparse_matrix, type(expected_sparse_matrix))
        assert all(sparse_matrix.data == expected_sparse_matrix.data)
        assert all(sparse_matrix.indices == expected_sparse_matrix.indices)

    def test_sparse_matrix_sparse_hamiltonian(self):
        """Test the sparse_matrix representation of scaled ops."""
        scalar = 1.23
        op = qp.Hadamard(wires=0)
        sparse_ham = qp.SparseHamiltonian(csr_matrix(op.matrix()), wires=0)

        sprod_op = SProd(scalar, sparse_ham)
        sparse_matrix = sprod_op.sparse_matrix()

        expected_sparse_matrix = scalar * op.matrix()
        expected_sparse_matrix = csr_matrix(expected_sparse_matrix)
        expected_sparse_matrix.eliminate_zeros()

        assert np.allclose(sparse_matrix.todense(), expected_sparse_matrix.todense())


class TestProperties:
    @pytest.mark.parametrize("op_scalar_tup", ops)
    def test_queue_category(self, op_scalar_tup):
        """Test queue_category property is always None."""  # currently not supporting queuing SProd
        scalar, op = op_scalar_tup
        sprod_op = SProd(scalar, op)
        assert sprod_op._queue_category is None  # pylint: disable=protected-access

    def test_eigvals(self):
        """Test that the eigvals of the scalar product op are correct."""
        coeff, op = (1.0 + 2j, qp.PauliX(wires=0))
        sprod_op = SProd(coeff, op)
        sprod_op_eigvals = sprod_op.eigvals()

        x_eigvals = np.array([1.0, -1.0])
        true_eigvals = coeff * x_eigvals  # the true eigvals
        assert np.allclose(sprod_op_eigvals, true_eigvals)

    ops_are_hermitian = (
        (qp.PauliX(wires=0), 1.23 + 0.0j, True),  # Op is hermitian, scalar is real
        (qp.RX(1.23, wires=0), 1.0 + 0.0j, False),  # Op not hermitian
        (qp.PauliZ(wires=0), 2.0 + 1.0j, False),  # Scalar not real
    )

    @pytest.mark.parametrize("op, scalar, hermitian_status", ops_are_hermitian)
    def test_is_verified_hermitian(self, op, scalar, hermitian_status):
        """Test that scalar product ops are correctly classified as hermitian or not."""
        sprod_op = s_prod(scalar, op)
        assert sprod_op.is_verified_hermitian == hermitian_status

    @pytest.mark.tf
    def test_is_verified_hermitian_tf(self):
        """Test that is_verified_hermitian works when a tf type scalar is provided."""
        import tensorflow as tf

        coeffs = (tf.Variable(1.23), tf.Variable(1.23 + 1.2j))
        true_hermitian_states = (True, False)

        for scalar, hermitian_state in zip(coeffs, true_hermitian_states):
            op = s_prod(scalar, qp.PauliX(wires=0))
            assert op.is_verified_hermitian == hermitian_state

    @pytest.mark.tf
    def test_no_dtype_promotion(self):
        import tensorflow as tf

        op = qp.s_prod(tf.constant(0.5), qp.X(0))
        assert op.scalar.dtype == next(iter(op.pauli_rep.values())).dtype

    @pytest.mark.jax
    def test_is_verified_hermitian_jax(self):
        """Test that is_verified_hermitian works when a jax type scalar is provided."""
        import jax.numpy as jnp

        coeffs = (jnp.array(1.23), jnp.array(1.23 + 1.2j))
        true_hermitian_states = (True, False)

        for scalar, hermitian_state in zip(coeffs, true_hermitian_states):
            op = s_prod(scalar, qp.PauliX(wires=0))
            assert op.is_verified_hermitian == hermitian_state

    @pytest.mark.torch
    def test_is_verified_hermitian_torch(self):
        """Test that is_verified_hermitian works when a torch type scalar is provided."""
        import torch

        coeffs = (torch.tensor(1.23), torch.tensor(1.23 + 1.2j))
        true_hermitian_states = (True, False)

        for scalar, hermitian_state in zip(coeffs, true_hermitian_states):
            op = s_prod(scalar, qp.PauliX(wires=0))
            assert op.is_verified_hermitian == hermitian_state

    ops_labels = (
        (qp.PauliX(wires=0), 1.23, 2, "1.23*X"),
        (qp.RX(1.23, wires=0), 4.56, 1, "4.6*RX\n(1.2)"),
        (qp.RY(1.234, wires=0), 4.56, 3, "4.560*RY\n(1.234)"),
        (qp.Rot(1.0, 2.12, 3.1416, wires=0), 1, 2, "1.00*Rot\n(1.00,\n2.12,\n3.14)"),
    )

    @pytest.mark.parametrize("op, scalar, decimal, label", ops_labels)
    def test_label(self, op, scalar, decimal, label):
        """Testing that the label method works well with SProd objects."""
        sprod_op = s_prod(scalar, op)
        op_label = sprod_op.label(decimals=decimal)
        assert label == op_label

    def test_label_cache(self):
        """Test label method with cache keyword arg."""
        base = qp.QubitUnitary(np.eye(2), wires=0)
        op = s_prod(-1.2, base)

        cache = {"matrices": []}
        assert op.label(decimals=2, cache=cache) == "-1.20*U\n(M0)"
        assert len(cache["matrices"]) == 1

    op_pauli_reps = (
        (
            qp.s_prod(1.23, qp.PauliZ(wires=0)),
            qp.pauli.PauliSentence({qp.pauli.PauliWord({0: "Z"}): 1.23}),
        ),
        (
            qp.s_prod(-1j, qp.PauliX(wires=1)),
            qp.pauli.PauliSentence({qp.pauli.PauliWord({1: "X"}): -1j}),
        ),
        (
            qp.s_prod(1.23 - 4j, qp.PauliY(wires="a")),
            qp.pauli.PauliSentence({qp.pauli.PauliWord({"a": "Y"}): 1.23 - 4j}),
        ),
    )

    @pytest.mark.parametrize("op, rep", op_pauli_reps)
    def test_pauli_rep(self, op, rep):
        """Test the pauli rep is produced as expected."""
        assert op.pauli_rep == rep

    def test_pauli_rep_none_if_base_pauli_rep_none(self):
        """Test that None is produced if the base op does not have a pauli rep"""
        base = qp.RX(1.23, wires=0)
        op = qp.s_prod(2, base)
        assert op.pauli_rep is None

    def test_batching_properties(self):
        """Test the batching properties and methods."""

        # base is batched
        base = qp.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = qp.s_prod(0.5, base)
        assert op.batch_size == 3

        # coeff is batched
        base = qp.RX(1, 0)
        op = qp.s_prod(np.array([1.2, 2.3, 3.4]), base)
        assert op.batch_size == 3

        # both are batched
        base = qp.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = qp.s_prod(np.array([1.2, 2.3, 3.4]), base)
        assert op.batch_size == 3

    def test_different_batch_sizes_raises_error(self):
        """Test that using different batch sizes for base and scalar raises an error."""
        base = qp.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = qp.s_prod(np.array([0.1, 1.2, 2.3, 3.4]), base)
        with pytest.raises(
            ValueError, match="Broadcasting was attempted but the broadcasted dimensions"
        ):
            _ = op.batch_size


class TestSimplify:
    """Test SProd simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        sprod_op = s_prod(5, s_prod(3, s_prod(-1, qp.RZ(1.32, wires=0))))
        assert sprod_op.arithmetic_depth == 3

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        sprod_op = SProd(
            2, SProd(2, qp.RZ(1.32, wires=0)) + qp.Identity(wires=0) + qp.RX(1.9, wires=1)
        )
        final_op = qp.ops.Sum(  # pylint:disable=no-member
            SProd(4, qp.RZ(1.32, wires=0)),
            SProd(2, qp.Identity(wires=0)),
            SProd(2, qp.RX(1.9, wires=1)),
        )
        simplified_op = sprod_op.simplify()
        qp.assert_equal(simplified_op, final_op)

    def test_simplify_scalar_equal_to_1(self):
        """Test the simplify method when the scalar is 1."""
        sprod_op = s_prod(1, qp.PauliX(0))
        final_op = qp.PauliX(0)
        simplified_op = sprod_op.simplify()
        qp.assert_equal(simplified_op, final_op)

    def test_simplify_nested_sprod_scalar_equal_to_1(self):
        """Test the simplify method with nested SProd where the global scalar is 1."""
        sprod_op = s_prod(3, s_prod(1 / 3, qp.PauliX(0)))
        final_op = qp.PauliX(0)
        simplified_op = sprod_op.simplify()
        qp.assert_equal(simplified_op, final_op)

    def test_simplify_with_sum_operator(self):
        """Test the simplify method a scalar product of a Sum operator."""
        sprod_op = s_prod(0 - 3j, qp.sum(qp.PauliX(0), qp.PauliX(0)))
        final_op = s_prod(0 - 6j, qp.PauliX(0))
        simplified_op = sprod_op.simplify()
        qp.assert_equal(simplified_op, final_op)

    @pytest.mark.jax
    def test_simplify_pauli_rep_jax(self):
        """Test that simplifying operators with a valid pauli representation works with jax interface."""
        import jax.numpy as jnp

        c1, c2, c3 = jnp.array(1.23), jnp.array(2.0), jnp.array(2.46)

        op = s_prod(c1, s_prod(c2, qp.PauliX(0)))
        result = s_prod(c3, qp.PauliX(0))
        simplified_op = op.simplify()

        qp.assert_equal(simplified_op, result)

    @pytest.mark.tf
    def test_simplify_pauli_rep_tf(self):
        """Test that simplifying operators with a valid pauli representation works with tf interface."""
        import tensorflow as tf

        c1, c2, c3 = tf.Variable(1.23), tf.Variable(2.0), tf.Variable(2.46)

        op = s_prod(c1, s_prod(c2, qp.PauliX(0)))
        result = s_prod(c3, qp.PauliX(0))
        simplified_op = op.simplify()
        qp.assert_equal(simplified_op, result)

    @pytest.mark.torch
    def test_simplify_pauli_rep_torch(self):
        """Test that simplifying operators with a valid pauli representation works with torch interface."""
        import torch

        c1, c2, c3 = torch.tensor(1.23), torch.tensor(2.0), torch.tensor(2.46)

        op = s_prod(c1, s_prod(c2, qp.PauliX(0)))
        result = s_prod(c3, qp.PauliX(0))
        simplified_op = op.simplify()

        qp.assert_equal(simplified_op, result)


class TestWrapperFunc:
    @pytest.mark.parametrize("op_scalar_tup", ops)
    def test_s_prod_top_level(self, op_scalar_tup):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        coeff, op = op_scalar_tup

        op_id = "sprod_op"

        sprod_func_op = s_prod(coeff, op, id=op_id)
        sprod_class_op = SProd(coeff, op, id=op_id)
        qp.assert_equal(sprod_func_op, sprod_class_op)

    def test_lazy_mode(self):
        """Test that by default, the operator is simply wrapped in `SProd`, even if a simplification exists."""
        op = s_prod(3, s_prod(4, qp.PauliX(0)))

        assert isinstance(op, SProd)
        assert op.scalar == 3
        assert isinstance(op.base, SProd)

    def test_non_lazy_mode(self):
        """Test the lazy=False keyword."""
        op = s_prod(3, s_prod(4, qp.PauliX(0)), lazy=False)

        assert isinstance(op, SProd)
        assert op.scalar == 12
        qp.assert_equal(op.base, qp.PauliX(0))

    def test_non_lazy_mode_queueing(self):
        """Test that if a simpification is accomplished, the metadata for the original op
        and the new simplified op is updated."""
        with qp.queuing.AnnotatedQueue() as q:
            sprod1 = s_prod(4, qp.PauliX(0))
            sprod2 = s_prod(3, sprod1, lazy=False)

        assert len(q) == 1
        assert q.queue[0] is sprod2


class TestIntegration:
    """Integration tests for SProd with a QNode."""

    def test_measurement_process_expval(self):
        """Test SProd class instance in expval measurement process."""
        dev = qp.device("default.qubit", wires=2)
        sprod_op = SProd(1.23, qp.Hadamard(1))

        @qp.qnode(dev)
        def my_circ():
            qp.PauliX(0)
            return qp.expval(sprod_op)

        exp_val = my_circ()
        true_exp_val = qnp.array(1.23 / qnp.sqrt(2))
        assert qnp.allclose(exp_val, true_exp_val)

    def test_measurement_process_var(self):
        """Test SProd class instance in var measurement process."""
        dev = qp.device("default.qubit", wires=2)
        sprod_op = SProd(1.23, qp.Hadamard(1))

        @qp.qnode(dev)
        def my_circ():
            qp.PauliX(0)
            return qp.var(sprod_op)

        var = my_circ()
        true_var = qnp.array(1.23**2 / 2)
        assert qnp.allclose(var, true_var)

    def test_measurement_process_probs(self):
        """Test SProd class instance in probs measurement process raises error."""  # currently can't support due to bug
        dev = qp.device("default.qubit", wires=2)
        sprod_op = SProd(1.23, qp.Hadamard(1))

        @qp.qnode(dev)
        def my_circ():
            qp.PauliX(0)
            return qp.probs(op=sprod_op), qp.probs(op=qp.Hadamard(1))

        res1, res2 = my_circ()
        assert qp.math.allclose(res1, res2)

    def test_measurement_process_sample(self):
        """Test SProd class instance in sample measurement process."""
        dev = qp.device("default.qubit", wires=2)
        sprod_op = SProd(1.23, qp.PauliX(1))

        @qp.set_shots(20)
        @qp.qnode(dev)
        def my_circ():
            qp.Hadamard(1)
            return qp.sample(op=sprod_op)

        results = my_circ()

        assert len(results) == 20
        assert (results == 1.23).all()

    def test_measurement_process_count(self):
        """Test SProd class instance in counts measurement process."""
        dev = qp.device("default.qubit", wires=2)
        sprod_op = SProd(1.23, qp.PauliX(1))

        @qp.set_shots(20)
        @qp.qnode(dev)
        def my_circ():
            qp.Hadamard(1)
            return qp.counts(op=sprod_op)

        results = my_circ()

        assert sum(results.values()) == 20
        assert 1.23 in results  # pylint:disable=unsupported-membership-test
        assert -1.23 not in results  # pylint:disable=unsupported-membership-test

    def test_differentiable_scalar(self):
        """Test that the gradient can be computed of the scalar when a SProd op
        is used in the measurement process."""
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev, diff_method="best")
        def circuit(scalar):
            qp.PauliX(wires=0)
            return qp.expval(SProd(scalar, qp.Hadamard(wires=0)))

        scalar = qnp.array(1.23, requires_grad=True)
        grad = qp.grad(circuit)(scalar)

        true_grad = -1 / qnp.sqrt(2)
        assert qnp.allclose(grad, true_grad)

    def test_differentiable_measurement_process(self):
        """Test that the gradient can be computed with a SProd op in the measurement process."""
        sprod_op = SProd(100, qp.Hadamard(0))
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev, diff_method="best")
        def circuit(weights):
            qp.RX(weights[0], wires=0)
            return qp.expval(sprod_op)

        weights = qnp.array([0.1], requires_grad=True)
        grad = qp.grad(circuit)(weights)

        true_grad = 100 * -qnp.sqrt(2) * qnp.cos(weights[0] / 2) * qnp.sin(weights[0] / 2)
        assert qnp.allclose(grad, true_grad)

    @pytest.mark.torch
    @pytest.mark.parametrize("diff_method", ("parameter-shift", "backprop"))
    def test_torch(self, diff_method):
        """Test that interface parameters can be unwrapped to numpy. This will occur when parameter-shift
        is requested for a given interface."""

        import torch

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev, diff_method=diff_method)
        def circuit(s):
            return qp.expval(qp.s_prod(s, qp.PauliZ(0)))

        res = circuit(torch.tensor(2))

        assert qp.math.allclose(res, 2)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ("parameter-shift", "backprop"))
    def test_jax(self, diff_method):
        """Test that interface parameters can be unwrapped to numpy. This will occur when parameter-shift
        is requested for a given interface."""

        from jax import numpy as jnp

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev, diff_method=diff_method)
        def circuit(s):
            return qp.expval(qp.s_prod(s, qp.PauliZ(0)))

        res = circuit(jnp.array(2))

        assert qp.math.allclose(res, 2)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ("parameter-shift", "backprop"))
    def test_tensorflow_qnode(self, diff_method):
        import tensorflow as tf

        dev = qp.device("default.qubit", wires=5)

        @qp.qnode(dev, diff_method=diff_method)
        def circuit(s):
            return qp.expval(qp.s_prod(s, qp.PauliZ(0)))

        res = circuit(tf.Variable(2, dtype=tf.float64))

        assert qp.math.allclose(res, 2)


class TestArithmetic:
    """Test arithmetic decomposition methods."""

    def test_pow(self):
        """Test the pow method for SProd Operators."""

        sprod_op = SProd(3, qp.RX(1.23, wires=0))
        final_op = SProd(scalar=3**2, base=qp.pow(base=qp.RX(1.23, wires=0), z=2))
        pow_op = sprod_op.pow(z=2)[0]
        qp.assert_equal(pow_op, final_op)

    def test_adjoint(self):
        """Test the adjoint method for Sprod Operators."""

        sprod_op = SProd(3j, qp.RX(1.23, wires=0))
        final_op = SProd(scalar=-3j, base=qp.adjoint(qp.RX(1.23, wires=0)))
        adj_op = sprod_op.adjoint()
        qp.assert_equal(adj_op, final_op)
