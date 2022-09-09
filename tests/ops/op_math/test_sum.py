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
Unit tests for the Sum arithmetic class of qubit operations
"""
from copy import copy
from typing import Tuple

import gate_data as gd  # a file containing matrix rep of each gate
import numpy as np
import pytest

import pennylane as qml
import pennylane.numpy as qnp
from pennylane import QuantumFunctionError, math
from pennylane.operation import DecompositionUndefinedError, MatrixUndefinedError, Operator
from pennylane.ops.op_math import Sum, op_sum
from pennylane.ops.op_math.sum import _sum, _sum_sort  # pylint: disable=protected-access
from pennylane.wires import Wires

no_mat_ops = (
    qml.Barrier,
    qml.WireCut,
)

non_param_ops = (
    (qml.Identity, gd.I),
    (qml.Hadamard, gd.H),
    (qml.PauliX, gd.X),
    (qml.PauliY, gd.Y),
    (qml.PauliZ, gd.Z),
    (qml.S, gd.S),
    (qml.T, gd.T),
    (qml.SX, gd.SX),
    (qml.CNOT, gd.CNOT),
    (qml.CZ, gd.CZ),
    (qml.CY, gd.CY),
    (qml.SWAP, gd.SWAP),
    (qml.ISWAP, gd.ISWAP),
    (qml.SISWAP, gd.SISWAP),
    (qml.CSWAP, gd.CSWAP),
    (qml.Toffoli, gd.Toffoli),
)

param_ops = (
    (qml.RX, gd.Rotx),
    (qml.RY, gd.Roty),
    (qml.RZ, gd.Rotz),
    (qml.PhaseShift, gd.Rphi),
    (qml.Rot, gd.Rot3),
    (qml.U1, gd.U1),
    (qml.U2, gd.U2),
    (qml.U3, gd.U3),
    (qml.CRX, gd.CRotx),
    (qml.CRY, gd.CRoty),
    (qml.CRZ, gd.CRotz),
    (qml.CRot, gd.CRot3),
    (qml.IsingXX, gd.IsingXX),
    (qml.IsingYY, gd.IsingYY),
    (qml.IsingZZ, gd.IsingZZ),
)

ops = (
    (qml.PauliX(wires=0), qml.PauliZ(wires=0), qml.Hadamard(wires=0)),
    (qml.CNOT(wires=[0, 1]), qml.RX(1.23, wires=1), qml.Identity(wires=0)),
    (
        qml.IsingXX(4.56, wires=[2, 3]),
        qml.Toffoli(wires=[1, 2, 3]),
        qml.Rot(0.34, 1.0, 0, wires=0),
    ),
)

ops_rep = (
    "PauliX(wires=[0]) + PauliZ(wires=[0]) + Hadamard(wires=[0])",
    "CNOT(wires=[0, 1]) + RX(1.23, wires=[1]) + Identity(wires=[0])",
    "IsingXX(4.56, wires=[2, 3]) + Toffoli(wires=[1, 2, 3]) + Rot(0.34, 1.0, 0, wires=[0])",
)


def sum_using_dunder_method(*summands, do_queue=True, id=None):
    """Helper function which computes the sum of all the summands to invoke the
    __add__ dunder method."""
    return sum(summands)


def compare_and_expand_mat(mat1, mat2):
    """Helper function which takes two square matrices (of potentially different sizes)
    and expands the smaller matrix until their shapes match."""

    if mat1.size == mat2.size:
        return mat1, mat2

    (smaller_mat, larger_mat, flip_order) = (
        (mat1, mat2, 0) if mat1.size < mat2.size else (mat2, mat1, 1)
    )

    while smaller_mat.size < larger_mat.size:
        smaller_mat = math.cast_like(math.kron(smaller_mat, math.eye(2)), smaller_mat)

    if flip_order:
        return larger_mat, smaller_mat

    return smaller_mat, larger_mat


class TestInitialization:
    """Test the initialization."""

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, op_sum])
    @pytest.mark.parametrize("id", ("foo", "bar"))
    def test_init_sum_op(self, id, sum_method):
        """Test the initialization of a Sum operator."""
        sum_op = sum_method(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"), do_queue=True, id=id)

        assert sum_op.wires == Wires((0, "a"))
        assert sum_op.num_wires == 2
        assert sum_op.name == "Sum"
        if sum_method.__name__ == op_sum.__name__:
            assert sum_op.id == id

        assert sum_op.data == [[], [0.23]]
        assert sum_op.parameters == [[], [0.23]]
        assert sum_op.num_params == 1

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, op_sum])
    def test_init_sum_op_with_sum_summands(self, sum_method):
        """Test the initialization of a Sum operator which contains a summand that is another
        Sum operator."""
        sum_op = sum_method(
            Sum(qml.PauliX(wires=0), qml.RZ(0.23, wires="a")), qml.RX(9.87, wires=0)
        )
        assert sum_op.wires == Wires((0, "a"))
        assert sum_op.num_wires == 2
        assert sum_op.name == "Sum"
        assert sum_op.id is None

        assert sum_op.data == [[[], [0.23]], [9.87]]
        assert sum_op.parameters == [[[], [0.23]], [9.87]]
        assert sum_op.num_params == 2

    def test_raise_error_fewer_then_2_summands(self):
        """Test that initializing a Sum operator with less than 2 summands raises a ValueError."""
        with pytest.raises(ValueError, match="Require at least two operators to sum;"):
            op_sum(qml.PauliX(0))

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, op_sum])
    def test_queue_idx(self, sum_method):
        """Test that queue_idx is None."""
        sum_op = sum_method(qml.PauliX(0), qml.Identity(1))
        assert sum_op.queue_idx is None

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, op_sum])
    def test_parameters(self, sum_method):
        """Test that parameters are initialized correctly."""
        sum_op = sum_method(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert sum_op.parameters == [[9.87], [1.23, 4.0, 5.67]]

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, op_sum])
    def test_data(self, sum_method):
        """Test that data is initialized correctly."""
        sum_op = sum_method(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert sum_op.data == [[9.87], [1.23, 4.0, 5.67]]

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, op_sum])
    def test_data_setter(self, sum_method):
        """Test the setter method for data"""
        sum_op = sum_method(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert sum_op.data == [[9.87], [1.23, 4.0, 5.67]]

        new_data = [[1.23], [0.0, -1.0, -2.0]]
        sum_op.data = new_data
        assert sum_op.data == new_data

        for op, new_entry in zip(sum_op.summands, new_data):
            assert op.data == new_entry

    @pytest.mark.parametrize("ops_lst", ops)
    def test_terms(self, ops_lst):
        """Test that terms are initialized correctly."""
        sum_op = Sum(*ops_lst)
        coeff, sum_ops = sum_op.terms()

        assert coeff == [1.0, 1.0, 1.0]

        for op1, op2 in zip(sum_ops, ops_lst):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert op1.data == op2.data

    def test_ndim_params_raises_error(self):
        """Test that calling ndim_params raises a ValueError."""
        sum_op = Sum(qml.PauliX(0), qml.Identity(1))

        with pytest.raises(AttributeError):
            _ = sum_op.ndim_params

    def test_batch_size_raises_error(self):
        """Test that calling batch_size raises a ValueError."""
        sum_op = Sum(qml.PauliX(0), qml.Identity(1))

        with pytest.raises(AttributeError):
            _ = sum_op.batch_size

    def test_decomposition_raises_error(self):
        """Test that calling decomposition() raises a ValueError."""
        sum_op = Sum(qml.PauliX(0), qml.Identity(1))

        with pytest.raises(DecompositionUndefinedError):
            sum_op.decomposition()

    def test_eigen_caching(self):
        """Test that the eigendecomposition is stored in cache."""
        diag_sum_op = Sum(qml.PauliZ(wires=0), qml.Identity(wires=1))
        eig_decomp = diag_sum_op.eigendecomposition

        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        eigs_cache = diag_sum_op._eigs[diag_sum_op.hash]
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)

    def test_diagonalizing_gates(self):
        """Test that the diagonalizing gates are correct."""
        diag_sum_op = Sum(qml.PauliZ(wires=0), qml.Identity(wires=1))
        assert diag_sum_op.diagonalizing_gates() == []


class TestMscMethods:
    """Test dunder and other visualizing methods."""

    @pytest.mark.parametrize("ops_lst, ops_rep", tuple((i, j) for i, j in zip(ops, ops_rep)))
    def test_repr(self, ops_lst, ops_rep):
        """Test __repr__ method."""
        sum_op = Sum(*ops_lst)
        assert ops_rep == repr(sum_op)

    def test_nested_repr(self):
        """Test nested repr values while other nested features such as equality are not ready"""
        sum_op = qml.PauliX(0) + qml.RY(1, wires=1) @ qml.PauliX(0)
        assert "PauliX(wires=[0]) + (RY(1, wires=[1]) @ PauliX(wires=[0]))" == repr(sum_op)

    def test_label(self):
        """Test label method."""
        sum_op = qml.RY(1, wires=1) + qml.PauliX(1)
        assert "RY+X" == sum_op.label()
        with pytest.raises(ValueError):
            sum_op.label(base_label=["only_first"])

        nested_op = qml.PauliX(0) + sum_op
        assert "X+(RY+X)" == nested_op.label()
        assert "X+(RY\n(1.00)+X)" == nested_op.label(decimals=2)
        assert "x0+(ry+x1)" == nested_op.label(base_label=["x0", ["ry", "x1"]])

        U = np.array([[1, 0], [0, -1]])
        cache = {"matrices": []}
        prod_op = qml.PauliX(0) + (qml.PauliY(1) + qml.QubitUnitary(U, wires=0))
        assert "X+(Y+U(M0))" == prod_op.label(cache=cache)
        assert cache["matrices"] == [U]

    @pytest.mark.parametrize("ops_lst", ops)
    def test_copy(self, ops_lst):
        """Test __copy__ method."""
        sum_op = Sum(*ops_lst)
        copied_op = copy(sum_op)

        assert sum_op.id == copied_op.id
        assert sum_op.data == copied_op.data
        assert sum_op.wires == copied_op.wires

        for s1, s2 in zip(sum_op.summands, copied_op.summands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data


class TestMatrix:
    """Test matrix-related methods."""

    @pytest.mark.parametrize("op_and_mat1", non_param_ops)
    @pytest.mark.parametrize("op_and_mat2", non_param_ops)
    def test_non_parametric_ops_two_terms(
        self,
        op_and_mat1: Tuple[Operator, np.ndarray],
        op_and_mat2: Tuple[Operator, np.ndarray],
    ):
        """Test matrix method for a sum of non_parametric ops"""
        op1, mat1 = op_and_mat1
        op2, mat2 = op_and_mat2
        mat1, mat2 = compare_and_expand_mat(mat1, mat2)
        true_mat = mat1 + mat2

        sum_op = Sum(op1(wires=range(op1.num_wires)), op2(wires=range(op2.num_wires)))
        sum_mat = sum_op.matrix()

        assert np.allclose(sum_mat, true_mat)

    @pytest.mark.parametrize("op_mat1", param_ops)
    @pytest.mark.parametrize("op_mat2", param_ops)
    def test_parametric_ops_two_terms(
        self, op_mat1: Tuple[Operator, np.ndarray], op_mat2: Tuple[Operator, np.ndarray]
    ):
        """Test matrix method for a sum of parametric ops"""
        op1, mat1 = op_mat1
        op2, mat2 = op_mat2

        par1 = tuple(range(op1.num_params))
        par2 = tuple(range(op2.num_params))
        mat1, mat2 = compare_and_expand_mat(mat1(*par1), mat2(*par2))

        sum_op = Sum(op1(*par1, wires=range(op1.num_wires)), op2(*par2, wires=range(op2.num_wires)))
        sum_mat = sum_op.matrix()
        true_mat = mat1 + mat2
        assert np.allclose(sum_mat, true_mat)

    @pytest.mark.parametrize("op", no_mat_ops)
    def test_error_no_mat(self, op: Operator):
        """Test that an error is raised if one of the summands doesn't
        have its matrix method defined."""
        sum_op = Sum(op(wires=0), qml.PauliX(wires=2), qml.PauliZ(wires=1))
        with pytest.raises(MatrixUndefinedError):
            sum_op.matrix()

    def test_sum_ops_multi_terms(self):
        """Test matrix is correct for a sum of more than two terms."""
        sum_op = Sum(qml.PauliX(wires=0), qml.Hadamard(wires=0), qml.PauliZ(wires=0))
        mat = sum_op.matrix()

        true_mat = math.array(
            [
                [1 / math.sqrt(2) + 1, 1 / math.sqrt(2) + 1],
                [1 / math.sqrt(2) + 1, -1 / math.sqrt(2) - 1],
            ]
        )
        assert np.allclose(mat, true_mat)

    def test_sum_ops_multi_wires(self):
        """Test matrix is correct when multiple wires are used in the sum."""
        sum_op = Sum(qml.PauliX(wires=0), qml.Hadamard(wires=1), qml.PauliZ(wires=2))
        mat = sum_op.matrix()

        x = math.array([[0, 1], [1, 0]])
        h = 1 / math.sqrt(2) * math.array([[1, 1], [1, -1]])
        z = math.array([[1, 0], [0, -1]])

        true_mat = (
            math.kron(x, math.eye(4))
            + math.kron(math.kron(math.eye(2), h), math.eye(2))
            + math.kron(math.eye(4), z)
        )

        assert np.allclose(mat, true_mat)

    def test_sum_ops_wire_order(self):
        """Test correct matrix is returned when the wire_order arg is provided."""
        sum_op = Sum(qml.PauliZ(wires=2), qml.PauliX(wires=0), qml.Hadamard(wires=1))
        wire_order = [0, 1, 2]
        mat = sum_op.matrix(wire_order=wire_order)

        x = math.array([[0, 1], [1, 0]])
        h = 1 / math.sqrt(2) * math.array([[1, 1], [1, -1]])
        z = math.array([[1, 0], [0, -1]])

        true_mat = (
            math.kron(x, math.eye(4))
            + math.kron(math.kron(math.eye(2), h), math.eye(2))
            + math.kron(math.eye(4), z)
        )

        assert np.allclose(mat, true_mat)

    @staticmethod
    def get_qft_mat(num_wires):
        """Helper function to generate the matrix of a qft protocol."""
        omega = math.exp(np.pi * 1.0j / 2 ** (num_wires - 1))
        mat = math.zeros((2**num_wires, 2**num_wires), dtype="complex128")

        for m in range(2**num_wires):
            for n in range(2**num_wires):
                mat[m, n] = omega ** (m * n)

        return 1 / math.sqrt(2**num_wires) * mat

    def test_sum_templates(self):
        """Test that we can sum templates and generated matrix is correct."""
        wires = [0, 1, 2]
        sum_op = Sum(qml.QFT(wires=wires), qml.GroverOperator(wires=wires), qml.PauliX(wires=0))
        mat = sum_op.matrix()

        grov_mat = (1 / 4) * math.ones((8, 8), dtype="complex128") - math.eye(8, dtype="complex128")
        qft_mat = self.get_qft_mat(3)
        x = math.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]])
        x_mat = math.kron(x, math.eye(4, dtype="complex128"))

        true_mat = grov_mat + qft_mat + x_mat
        assert np.allclose(mat, true_mat)

    def test_sum_qchem_ops(self):
        """Test that qchem operations can be summed and the generated matrix is correct."""
        wires = [0, 1, 2, 3]
        sum_op = Sum(
            qml.OrbitalRotation(4.56, wires=wires),
            qml.SingleExcitation(1.23, wires=[0, 1]),
            qml.Identity(3),
        )
        mat = sum_op.matrix()

        or_mat = gd.OrbitalRotation(4.56)
        se_mat = math.kron(gd.SingleExcitation(1.23), math.eye(4, dtype="complex128"))
        i_mat = math.eye(16)

        true_mat = or_mat + se_mat + i_mat
        assert np.allclose(mat, true_mat)

    def test_sum_observables(self):
        """Test that observable objects can also be summed with correct matrix representation."""
        wires = [0, 1]
        sum_op = Sum(
            qml.Hermitian(qnp.array([[0.0, 1.0], [1.0, 0.0]]), wires=0),
            qml.Projector(basis_state=qnp.array([0, 1]), wires=wires),
        )
        mat = sum_op.matrix()

        her_mat = qnp.kron(qnp.array([[0.0, 1.0], [1.0, 0.0]]), qnp.eye(2))
        proj_mat = qnp.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )

        true_mat = her_mat + proj_mat
        assert np.allclose(mat, true_mat)

    def test_sum_qubit_unitary(self):
        """Test that an arbitrary QubitUnitary can be summed with correct matrix representation."""
        U = 1 / qnp.sqrt(2) * qnp.array([[1, 1], [1, -1]])  # Hadamard
        U_op = qml.QubitUnitary(U, wires=0)

        sum_op = Sum(U_op, qml.Identity(wires=1))
        mat = sum_op.matrix()

        true_mat = qnp.kron(U, qnp.eye(2)) + qnp.eye(4)
        assert np.allclose(mat, true_mat)

    def test_sum_hamiltonian(self):
        """Test that a hamiltonian object can be summed."""
        U = 0.5 * (qml.PauliX(wires=0) @ qml.PauliZ(wires=1))
        sum_op = Sum(U, qml.PauliX(wires=0))
        mat = sum_op.matrix()

        true_mat = [[0, 0, 1.5, 0], [0, 0, 0, 0.5], [1.5, 0, 0, 0], [0, 0.5, 0, 0]]

        assert np.allclose(mat, true_mat)

    # Add interface tests for each interface !

    @pytest.mark.jax
    def test_sum_jax(self):
        """Test matrix is cast correctly using jax parameters."""
        import jax.numpy as jnp

        theta = jnp.array(1.23)
        rot_params = jnp.array([0.12, 3.45, 6.78])

        sum_op = Sum(
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qml.RX(theta, wires=1),
            qml.Identity(wires=0),
        )
        mat = sum_op.matrix()

        true_mat = (
            jnp.kron(gd.Rot3(rot_params[0], rot_params[1], rot_params[2]), qnp.eye(2))
            + jnp.kron(qnp.eye(2), gd.Rotx(theta))
            + qnp.eye(4)
        )
        true_mat = jnp.array(true_mat)

        assert jnp.allclose(mat, true_mat)

    @pytest.mark.torch
    def test_sum_torch(self):
        """Test matrix is cast correctly using torch parameters."""
        import torch

        theta = torch.tensor(1.23)
        rot_params = torch.tensor([0.12, 3.45, 6.78])

        sum_op = Sum(
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qml.RX(theta, wires=1),
            qml.Identity(wires=0),
        )
        mat = sum_op.matrix()

        true_mat = (
            qnp.kron(gd.Rot3(rot_params[0], rot_params[1], rot_params[2]), qnp.eye(2))
            + qnp.kron(qnp.eye(2), gd.Rotx(theta))
            + qnp.eye(4)
        )
        true_mat = torch.tensor(true_mat)

        assert torch.allclose(mat, true_mat)

    @pytest.mark.tf
    def test_sum_tf(self):
        """Test matrix is cast correctly using tf parameters."""
        import tensorflow as tf

        theta = tf.Variable(1.23)
        rot_params = tf.Variable([0.12, 3.45, 6.78])

        sum_op = Sum(
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qml.RX(theta, wires=1),
            qml.Identity(wires=0),
        )
        mat = sum_op.matrix()

        true_mat = (
            qnp.kron(gd.Rot3(0.12, 3.45, 6.78), qnp.eye(2))
            + qnp.kron(qnp.eye(2), gd.Rotx(1.23))
            + qnp.eye(4)
        )
        true_mat = tf.Variable(true_mat)

        assert isinstance(mat, tf.Tensor)
        assert mat.dtype == true_mat.dtype
        assert np.allclose(mat, true_mat)

    # sparse matrix tests:

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix of a Prod op is defined and correct."""
        sum_op = op_sum(op1(wires=0), op2(wires=1))
        true_mat = math.kron(mat1, np.eye(2)) + math.kron(np.eye(2), mat2)
        sum_mat = sum_op.sparse_matrix().todense()

        assert np.allclose(true_mat, sum_mat)

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix_wire_order(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix of a Prod op is defined
        with wire order and correct."""
        true_mat = math.kron(mat2, np.eye(4)) + math.kron(np.eye(4), mat1)

        sum_op = op_sum(op1(wires=2), op2(wires=0))
        sum_mat = sum_op.sparse_matrix(wire_order=[0, 1, 2]).todense()

        assert np.allclose(true_mat, sum_mat)

    def test_sparse_matrix_undefined_error(self):
        """Test that an error is raised when the sparse matrix method
        is undefined for any of the factors."""

        class DummyOp(qml.operation.Operation):
            num_wires = 1

            def sparse_matrix(self, wire_order=None):
                raise qml.operation.SparseMatrixUndefinedError

        sum_op = op_sum(qml.PauliX(wires=0), DummyOp(wires=1))

        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            sum_op.sparse_matrix()


class TestProperties:
    """Test class properties."""

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_params(self, ops_lst):
        """Test num_params property updates correctly."""
        sum_op = Sum(*ops_lst)
        true_num_params = sum(op.num_params for op in ops_lst)

        assert sum_op.num_params == true_num_params

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_wires(self, ops_lst):
        """Test num_wires property updates correctly."""
        sum_op = Sum(*ops_lst)
        true_wires = set()

        for op in ops_lst:
            true_wires = true_wires.union(op.wires.toset())

        assert sum_op.num_wires == len(true_wires)

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, op_sum])
    @pytest.mark.parametrize("ops_lst", ops)
    def test_is_hermitian(self, ops_lst, sum_method):
        """Test is_hermitian property updates correctly."""
        sum_op = sum_method(*ops_lst)
        true_hermitian_state = True

        for op in ops_lst:
            true_hermitian_state = true_hermitian_state and op.is_hermitian

        assert sum_op.is_hermitian == true_hermitian_state

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, op_sum])
    @pytest.mark.parametrize("ops_lst", ops)
    def test_queue_catagory(self, ops_lst, sum_method):
        """Test queue_catagory property is always None."""  # currently not supporting queuing Sum
        sum_op = sum_method(*ops_lst)
        assert sum_op._queue_category is None

    def test_eigendecompostion(self):
        """Test that the computed Eigenvalues and Eigenvectors are correct."""
        diag_sum_op = Sum(qml.PauliZ(wires=0), qml.Identity(wires=1))
        eig_decomp = diag_sum_op.eigendecomposition
        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        true_eigvecs = qnp.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

        true_eigvals = qnp.tensor([0.0, 0.0, 2.0, 2.0])

        assert np.allclose(eig_vals, true_eigvals)
        assert np.allclose(eig_vecs, true_eigvecs)


class TestSimplify:
    """Test Sum simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        sum_op = (
            qml.RZ(1.32, wires=0) + qml.Identity(wires=0) + qml.RX(1.9, wires=1) + qml.PauliX(0)
        )
        assert sum_op.arithmetic_depth == 3

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        sum_op = qml.RZ(1.32, wires=0) + qml.Identity(wires=0) + qml.RX(1.9, wires=1)
        final_op = Sum(qml.RZ(1.32, wires=0), qml.Identity(wires=0), qml.RX(1.9, wires=1))
        simplified_op = sum_op.simplify()

        assert qml.equal(final_op, simplified_op)

    def test_simplify_grouping(self):
        """Test that the simplify method groups equal terms."""
        sum_op = op_sum(
            qml.prod(qml.RX(1, 0), qml.PauliX(0), qml.PauliZ(1)),
            qml.prod(qml.RX(1.0, 0), qml.PauliX(0), qml.PauliZ(1)),
            qml.adjoint(qml.op_sum(qml.RY(1, 0), qml.PauliZ(1))),
            qml.adjoint(qml.RY(1, 0)),
            qml.adjoint(qml.PauliZ(1)),
        )
        mod_angle = -1 % (4 * np.pi)
        final_op = op_sum(
            qml.s_prod(2, qml.prod(qml.RX(1.0, 0), qml.PauliX(0), qml.PauliZ(1))),
            qml.s_prod(2, qml.RY(mod_angle, 0)),
            qml.s_prod(2, qml.PauliZ(1)),
        )
        simplified_op = sum_op.simplify()

        assert qml.equal(final_op, simplified_op)

    def test_simplify_grouping_delete_terms(self):
        """Test that the simplify method deletes all terms with coefficient equal to 0."""
        sum_op = qml.op_sum(
            qml.PauliX(0),
            qml.s_prod(0.3, qml.PauliX(0)),
            qml.s_prod(0.8, qml.PauliX(0)),
            qml.s_prod(0.2, qml.PauliX(0)),
            qml.s_prod(0.4, qml.PauliX(0)),
            qml.s_prod(0.3, qml.PauliX(0)),
            qml.s_prod(-3, qml.PauliX(0)),
        )
        simplified_op = sum_op.simplify()
        final_op = qml.s_prod(0, qml.Identity(0))
        assert qml.equal(final_op, simplified_op)

    def test_simplify_grouping_with_tolerance(self):
        """Test the simplify method with a specific tolerance."""
        sum_op = qml.op_sum(-0.9 * qml.RX(1, 0), qml.RX(1, 0))
        final_op = qml.s_prod(0, qml.Identity(0))
        simplified_op = sum_op.simplify(cutoff=0.1)
        assert qml.equal(final_op, simplified_op)


class TestSortWires:
    """Tests for the wire sorting algorithm."""

    def test_sorting_operators_with_one_wire(self):
        """Test that the sorting alforithm works for operators that act on one wire."""
        op_list = [
            qml.PauliX(3),
            qml.PauliZ(2),
            qml.RX(1, 5),
            qml.PauliY(0),
            qml.PauliY(1),
            qml.PauliZ(3),
            qml.PauliX(5),
        ]
        sorted_list = _sum_sort(op_list)
        final_list = [
            qml.PauliY(0),
            qml.PauliY(1),
            qml.PauliZ(2),
            qml.PauliX(3),
            qml.PauliZ(3),
            qml.PauliX(5),
            qml.RX(1, 5),
        ]

        for op1, op2 in zip(final_list, sorted_list):
            assert qml.equal(op1, op2)

    def test_sorting_operators_with_multiple_wires(self):
        """Test that the sorting alforithm works for operators that act on multiple wires."""
        op_tuple = (
            qml.PauliX(3),
            qml.PauliX(5),
            qml.Toffoli([2, 3, 4]),
            qml.CNOT([2, 5]),
            qml.RX(1, 5),
            qml.PauliY(0),
            qml.CRX(1, [0, 2]),
            qml.PauliZ(3),
            qml.CRY(1, [1, 2]),
        )
        sorted_list = _sum_sort(op_tuple)
        final_list = [
            qml.CRX(1, [0, 2]),
            qml.PauliY(0),
            qml.CRY(1, [1, 2]),
            qml.CNOT([2, 5]),
            qml.Toffoli([2, 3, 4]),
            qml.PauliX(3),
            qml.PauliZ(3),
            qml.PauliX(5),
            qml.RX(1, 5),
        ]

        for op1, op2 in zip(final_list, sorted_list):
            assert qml.equal(op1, op2)

    def test_sorting_operators_with_wire_map(self):
        """Test that the sorting alforithm works using a wire map."""
        op_list = [
            qml.PauliX("three"),
            qml.PauliX(5),
            qml.Toffoli([2, "three", 4]),
            qml.CNOT([2, 5]),
            qml.RX(1, 5),
            qml.PauliY(0),
            qml.CRX(1, ["test", 2]),
            qml.PauliZ("three"),
            qml.CRY(1, ["test", 2]),
        ]
        sorted_list = _sum_sort(op_list, wire_map={0: 0, "test": 1, 2: 2, "three": 3, 4: 4, 5: 5})
        final_list = [
            qml.PauliY(0),
            qml.CRX(1, ["test", 2]),
            qml.CRY(1, ["test", 2]),
            qml.CNOT([2, 5]),
            qml.Toffoli([2, "three", 4]),
            qml.PauliX("three"),
            qml.PauliZ("three"),
            qml.PauliX(5),
            qml.RX(1, 5),
        ]

        for op1, op2 in zip(final_list, sorted_list):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert op1.data == op2.data


class TestWrapperFunc:
    """Test wrapper function."""

    def test_op_sum_top_level(self):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        summands = (qml.PauliX(wires=1), qml.RX(1.23, wires=0), qml.CNOT(wires=[0, 1]))
        op_id = "sum_op"
        do_queue = False

        sum_func_op = op_sum(*summands, id=op_id, do_queue=do_queue)
        sum_class_op = Sum(*summands, id=op_id, do_queue=do_queue)

        assert sum_class_op.summands == sum_func_op.summands
        assert np.allclose(sum_class_op.matrix(), sum_func_op.matrix())
        assert sum_class_op.id == sum_func_op.id
        assert sum_class_op.wires == sum_func_op.wires
        assert sum_class_op.parameters == sum_func_op.parameters


class TestPrivateSum:
    """Test private _sum() method."""

    def test_sum_private(self):
        """Test the sum private method generates expected matrices."""
        mats_gen = (qnp.eye(2) for _ in range(3))

        sum_mat = _sum(mats_gen)
        expected_sum_mat = 3 * qnp.eye(2)
        assert qnp.allclose(sum_mat, expected_sum_mat)

    def test_dtype(self):
        """Test dtype keyword arg casts matrix correctly"""
        dtype = "complex128"
        mats_gen = (qnp.eye(2) for _ in range(3))

        sum_mat = _sum(mats_gen, dtype=dtype)
        expected_sum_mat = 3 * qnp.eye(2, dtype=dtype)

        assert sum_mat.dtype == "complex128"
        assert qnp.allclose(sum_mat, expected_sum_mat)

    def test_cast_like(self):
        """Test cast_like keyword arg casts matrix correctly"""
        cast_like = qnp.array(2, dtype="complex128")
        mats_gen = (qnp.eye(2) for _ in range(3))

        sum_mat = _sum(mats_gen, cast_like=cast_like)
        expected_sum_mat = 3 * qnp.eye(2, dtype="complex128")

        assert sum_mat.dtype == "complex128"
        assert qnp.allclose(sum_mat, expected_sum_mat)


class TestIntegration:
    """Integration tests for the Sum class."""

    def test_measurement_process_expval(self):
        """Test Sum class instance in expval measurement process."""
        dev = qml.device("default.qubit", wires=2)
        sum_op = Sum(qml.PauliX(0), qml.Hadamard(1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.expval(sum_op)

        exp_val = my_circ()
        true_exp_val = qnp.array(1 / qnp.sqrt(2))
        assert qnp.allclose(exp_val, true_exp_val)

    def test_measurement_process_var(self):
        """Test Sum class instance in var measurement process."""
        dev = qml.device("default.qubit", wires=2)
        sum_op = Sum(qml.PauliX(0), qml.Hadamard(1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.var(sum_op)

        var = my_circ()
        true_var = qnp.array(3 / 2)
        assert qnp.allclose(var, true_var)

    # def test_measurement_process_probs(self):
    #     dev = qml.device("default.qubit", wires=2)
    #     sum_op = Sum(qml.PauliX(0), qml.Hadamard(1))
    #
    #     @qml.qnode(dev)
    #     def my_circ():
    #         qml.PauliX(0)
    #         return qml.probs(op=sum_op)
    #
    #     hand_computed_probs = qnp.array([0.573223935039, 0.073223277604, 0.573223935039, 0.073223277604])
    #     returned_probs = qnp.array([0.0732233, 0.43898224, 0.06101776, 0.4267767])
    #     # TODO[Jay]: which of these two is correct?
    #     assert qnp.allclose(my_circ(), returned_probs)

    def test_measurement_process_probs(self):
        """Test Sum class instance in probs measurement process raises error."""
        dev = qml.device("default.qubit", wires=2)
        sum_op = Sum(qml.PauliX(0), qml.Hadamard(1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.probs(op=sum_op)

        with pytest.raises(
            QuantumFunctionError,
            match="Symbolic Operations are not supported for " "rotating probabilities yet.",
        ):
            my_circ()

    def test_measurement_process_sample(self):
        """Test Sum class instance in sample measurement process."""
        dev = qml.device("default.qubit", wires=2, shots=20)
        sum_op = Sum(qml.PauliX(0), qml.PauliX(0))

        @qml.qnode(dev)
        def my_circ():
            qml.prod(qml.Hadamard(0), qml.Hadamard(1))
            return qml.sample(op=sum_op)

        results = my_circ()

        assert len(results) == 20
        assert (results == 2).all()

    def test_measurement_process_count(self):
        """Test Sum class instance in counts measurement process."""
        dev = qml.device("default.qubit", wires=2, shots=20)
        sum_op = Sum(qml.PauliX(0), qml.PauliX(0))

        @qml.qnode(dev)
        def my_circ():
            qml.prod(qml.Hadamard(0), qml.Hadamard(1))
            return qml.counts(op=sum_op)

        results = my_circ()

        assert sum(results.values()) == 20
        assert 2 in results
        assert -2 not in results

    def test_differentiable_measurement_process(self):
        """Test that the gradient can be computed with a Sum op in the measurement process."""
        sum_op = Sum(qml.PauliX(0), qml.PauliZ(1))
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, grad_method="best")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(weights[2], wires=1)
            return qml.expval(sum_op)

        weights = qnp.array([0.1, 0.2, 0.3], requires_grad=True)
        grad = qml.grad(circuit)(weights)

        true_grad = qnp.array([-0.09347337, -0.18884787, -0.28818254])
        assert qnp.allclose(grad, true_grad)

    def test_non_hermitian_op_in_measurement_process(self):
        """Test that non-hermitian ops in a measurement process will raise a warning."""
        wires = [0, 1]
        dev = qml.device("default.qubit", wires=wires)
        sum_op = Sum(qml.RX(1.23, wires=0), qml.Identity(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.expval(sum_op)

        with pytest.warns(UserWarning, match="Sum might not be hermitian."):
            my_circ()


class TestArithmetic:
    """Test arithmetic decomposition methods."""

    def test_adjoint(self):
        """Test the adjoint method for Sum Operators."""

        sum_op = Sum(qml.RX(1.23, wires=0), qml.Identity(wires=1))
        final_op = Sum(qml.adjoint(qml.RX(1.23, wires=0)), qml.adjoint(qml.Identity(wires=1)))
        adj_op = sum_op.adjoint()

        assert qml.equal(final_op, adj_op)
