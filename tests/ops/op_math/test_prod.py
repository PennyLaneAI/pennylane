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
Unit tests for the Prod arithmetic class of qubit operations
"""
from copy import copy

import gate_data as gd  # a file containing matrix rep of each gate
import numpy as np
import pytest

import pennylane as qml
import pennylane.numpy as qnp
from pennylane import QuantumFunctionError, math
from pennylane.operation import MatrixUndefinedError, Operator
from pennylane.ops.op_math import Prod, prod
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
    (qml.PauliZ(0), qml.PauliX(1)),
    (qml.PauliX(wires=0), qml.PauliZ(wires=0), qml.Hadamard(wires=0)),
    (qml.CNOT(wires=[0, 1]), qml.RX(1.23, wires=1), qml.Identity(wires=0)),
    (
        qml.IsingXX(4.56, wires=[2, 3]),
        qml.Toffoli(wires=[1, 2, 3]),
        qml.Rot(0.34, 1.0, 0, wires=0),
    ),
)

ops_rep = (
    "PauliZ(wires=[0]) @ PauliX(wires=[1])",
    "PauliX(wires=[0]) @ PauliZ(wires=[0]) @ Hadamard(wires=[0])",
    "CNOT(wires=[0, 1]) @ RX(1.23, wires=[1]) @ Identity(wires=[0])",
    "IsingXX(4.56, wires=[2, 3]) @ Toffoli(wires=[1, 2, 3]) @ Rot(0.34, 1.0, 0, wires=[0])",
)

ops_hermitian_status = (  # computed manually
    True,  # True
    False,  # True
    False,  # False
    False,  # False
)


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

    @pytest.mark.parametrize("id", ("foo", "bar"))
    def test_init_prod_op(self, id):
        """Test the initialization of a Prod operator."""
        prod_op = prod(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"), do_queue=True, id=id)

        assert prod_op.wires == Wires((0, "a"))
        assert prod_op.num_wires == 2
        assert prod_op.name == "Prod"
        assert prod_op.id == id
        assert prod_op.queue_idx is None

        assert prod_op.data == [[], [0.23]]
        assert prod_op.parameters == [[], [0.23]]
        assert prod_op.num_params == 1

    def test_raise_error_fewer_then_2_factors(self):
        """Test that initializing a Prod operator with less than 2 factors raises a ValueError."""
        with pytest.raises(ValueError, match="Require at least two operators to multiply;"):
            prod(qml.PauliX(wires=0))

    def test_parameters(self):
        """Test that parameters are initialized correctly."""
        prod_op = prod(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert prod_op.parameters == [[9.87], [1.23, 4.0, 5.67]]

    def test_data(self):
        """Test that data is initialized correctly."""
        prod_op = prod(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert prod_op.data == [[9.87], [1.23, 4.0, 5.67]]

    def test_data_setter(self):
        """Test the setter method for data"""
        prod_op = prod(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert prod_op.data == [[9.87], [1.23, 4.0, 5.67]]

        new_data = [[1.23], [0.0, -1.0, -2.0]]
        prod_op.data = new_data
        assert prod_op.data == new_data

        for op, new_entry in zip(prod_op.factors, new_data):
            assert op.data == new_entry

    @pytest.mark.parametrize("ops_lst", ops)
    def test_terms(self, ops_lst):
        """Test that terms are initialized correctly."""
        prod_op = prod(*ops_lst)
        coeff, prod_term_ops = prod_op.terms()  # not a fan of this behaviour

        assert coeff == [1.0]
        assert len(prod_term_ops) == 1
        assert prod_op.id == prod_term_ops[0].id
        assert prod_op.data == prod_term_ops[0].data
        assert prod_op.wires == prod_term_ops[0].wires

        for f1, f2 in zip(prod_op.factors, prod_term_ops[0].factors):
            assert qml.equal(f1, f2)

    def test_batch_size_is_None(self):
        """Test that calling batch_size returns None
        (i.e no batching with Prod)."""
        prod_op = prod(qml.PauliX(0), qml.Identity(1))
        assert prod_op.batch_size is None

    @pytest.mark.parametrize("ops_lst", ops)
    def test_decomposition(self, ops_lst):
        """Test the decomposition of a prod of operators is a list
        of the provided factors."""
        prod_op = prod(*ops_lst)
        decomposition = prod_op.decomposition()
        true_decomposition = list(ops_lst[::-1])  # reversed list of factors

        assert isinstance(decomposition, list)
        for op1, op2 in zip(decomposition, true_decomposition):
            assert qml.equal(op1, op2)

    def test_diagonalizing_gates(self):
        """Test that the diagonalizing gates are correct."""
        diag_prod_op = Prod(qml.PauliX(wires=0), qml.Identity(wires=1), qml.PauliX(wires=0))
        diagonalizing_gates = diag_prod_op.diagonalizing_gates()

        assert len(diagonalizing_gates) == 1
        diagonalizing_mat = diagonalizing_gates[0].matrix()

        true_mat = qnp.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )

        assert np.allclose(diagonalizing_mat, true_mat)

    def test_eigen_caching(self):
        """Test that the eigendecomposition is stored in cache."""
        prod_op = Prod(qml.PauliZ(wires=0), qml.Identity(wires=1))
        eig_decomp = prod_op.eigendecomposition

        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        eigs_cache = prod_op._eigs[
            (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        ]
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)


class TestMscMethods:
    """Test dunder methods."""

    @pytest.mark.parametrize("ops_lst, ops_rep", tuple((i, j) for i, j in zip(ops, ops_rep)))
    def test_repr(self, ops_lst, ops_rep):
        """Test __repr__ method."""
        prod_op = Prod(*ops_lst)
        assert ops_rep == repr(prod_op)

    @pytest.mark.parametrize("ops_lst", ops)
    def test_copy(self, ops_lst):
        """Test __copy__ method."""
        prod_op = Prod(*ops_lst)
        copied_op = copy(prod_op)

        assert prod_op.id == copied_op.id
        assert prod_op.data == copied_op.data
        assert prod_op.wires == copied_op.wires

        for f1, f2 in zip(prod_op.factors, copied_op.factors):
            assert qml.equal(f1, f2)
            assert not (f1 is f2)


class TestMatrix:
    """Test matrix-related methods."""

    @pytest.mark.parametrize("op1, mat1", non_param_ops)
    @pytest.mark.parametrize("op2, mat2", non_param_ops)
    def test_non_parametric_ops_two_terms(
        self,
        op1: Operator,
        mat1: np.ndarray,
        op2: Operator,
        mat2: np.ndarray,
    ):
        """Test matrix method for a product of non_parametric ops"""
        mat1, mat2 = compare_and_expand_mat(mat1, mat2)
        true_mat = mat1 @ mat2

        prod_op = Prod(op1(wires=range(op1.num_wires)), op2(wires=range(op2.num_wires)))
        prod_mat = prod_op.matrix()

        assert np.allclose(prod_mat, true_mat)

    @pytest.mark.parametrize("op1, mat1", param_ops)
    @pytest.mark.parametrize("op2, mat2", param_ops)
    def test_parametric_ops_two_terms(
        self,
        op1: Operator,
        mat1: np.ndarray,
        op2: Operator,
        mat2: np.ndarray,
    ):
        """Test matrix method for a product of parametric ops"""
        par1 = tuple(range(op1.num_params))
        par2 = tuple(range(op2.num_params))
        mat1, mat2 = compare_and_expand_mat(mat1(*par1), mat2(*par2))

        prod_op = Prod(
            op1(*par1, wires=range(op1.num_wires)), op2(*par2, wires=range(op2.num_wires))
        )
        prod_mat = prod_op.matrix()
        true_mat = mat1 @ mat2
        assert np.allclose(prod_mat, true_mat)

    @pytest.mark.parametrize("op", no_mat_ops)
    def test_error_no_mat(self, op: Operator):
        """Test that an error is raised if one of the factors doesn't
        have its matrix method defined."""
        prod_op = Prod(op(wires=0), qml.PauliX(wires=2), qml.PauliZ(wires=1))
        with pytest.raises(MatrixUndefinedError):
            prod_op.matrix()

    def test_prod_ops_multi_terms(self):
        """Test matrix is correct for a product of more than two terms."""
        prod_op = Prod(qml.PauliX(wires=0), qml.PauliY(wires=0), qml.PauliZ(wires=0))
        mat = prod_op.matrix()

        true_mat = math.array(
            [
                [1j, 0],
                [0, 1j],
            ]
        )
        assert np.allclose(mat, true_mat)

    def test_prod_ops_multi_wires(self):
        """Test matrix is correct when multiple wires are used in the product."""
        prod_op = Prod(qml.PauliX(wires=0), qml.Hadamard(wires=1), qml.PauliZ(wires=2))
        mat = prod_op.matrix()

        x = math.array([[0, 1], [1, 0]])
        h = 1 / math.sqrt(2) * math.array([[1, 1], [1, -1]])
        z = math.array([[1, 0], [0, -1]])

        true_mat = math.kron(x, math.kron(h, z))
        assert np.allclose(mat, true_mat)

    def test_prod_ops_wire_order(self):
        """Test correct matrix is returned when the wire_order arg is provided."""
        prod_op = Prod(qml.PauliZ(wires=2), qml.PauliX(wires=0), qml.Hadamard(wires=1))
        wire_order = [0, 1, 2]
        mat = prod_op.matrix(wire_order=wire_order)

        x = math.array([[0, 1], [1, 0]])
        h = 1 / math.sqrt(2) * math.array([[1, 1], [1, -1]])
        z = math.array([[1, 0], [0, -1]])

        true_mat = math.kron(x, math.kron(h, z))
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

    def test_prod_templates(self):
        """Test that we can compose templates and the generated matrix is correct."""
        wires = [0, 1, 2]
        prod_op = Prod(qml.QFT(wires=wires), qml.GroverOperator(wires=wires), qml.PauliX(wires=0))
        mat = prod_op.matrix()

        grov_mat = (1 / 4) * math.ones((8, 8), dtype="complex128") - math.eye(8, dtype="complex128")
        qft_mat = self.get_qft_mat(3)
        x = math.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]])
        x_mat = math.kron(x, math.eye(4, dtype="complex128"))

        true_mat = qft_mat @ grov_mat @ x_mat
        assert np.allclose(mat, true_mat)

    def test_prod_qchem_ops(self):
        """Test that qchem operations can be composed and the generated matrix is correct."""
        wires = [0, 1, 2, 3]
        prod_op = Prod(
            qml.OrbitalRotation(4.56, wires=wires),
            qml.SingleExcitation(1.23, wires=[0, 1]),
            qml.PauliX(4),
        )
        mat = prod_op.matrix()

        or_mat = math.kron(gd.OrbitalRotation(4.56), math.eye(2))
        se_mat = math.kron(gd.SingleExcitation(1.23), math.eye(8, dtype="complex128"))
        x = math.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]])
        x_mat = math.kron(math.eye(16, dtype="complex128"), x)

        true_mat = or_mat @ se_mat @ x_mat
        assert np.allclose(mat, true_mat)

    def test_prod_observables(self):
        """Test that observable objects can also be composed with correct matrix representation."""
        wires = [0, 1]
        prod_op = Prod(
            qml.Hermitian(qnp.array([[0.0, 1.0], [1.0, 0.0]]), wires=2),
            qml.Projector(basis_state=qnp.array([0, 1]), wires=wires),
        )
        mat = prod_op.matrix()

        hermitian_mat = qnp.array([[0.0, 1.0], [1.0, 0.0]])
        proj_mat = qnp.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )

        true_mat = qnp.kron(hermitian_mat, proj_mat)
        assert np.allclose(mat, true_mat)

    def test_prod_qubit_unitary(self):
        """Test that an arbitrary QubitUnitary can be composed with correct matrix representation."""
        U = 1 / qnp.sqrt(2) * qnp.array([[1, 1], [1, -1]])  # Hadamard
        U_op = qml.QubitUnitary(U, wires=0)

        prod_op = Prod(U_op, qml.Identity(wires=1))
        mat = prod_op.matrix()

        true_mat = qnp.kron(U, qnp.eye(2)) @ qnp.eye(4)
        assert np.allclose(mat, true_mat)

    # Add interface tests for each interface !

    @pytest.mark.jax
    def test_prod_jax(self):
        """Test matrix is cast correctly using jax parameters."""
        import jax.numpy as jnp

        theta = jnp.array(1.23)
        rot_params = jnp.array([0.12, 3.45, 6.78])

        prod_op = Prod(
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qml.RX(theta, wires=1),
            qml.Identity(wires=0),
        )
        mat = prod_op.matrix()

        true_mat = (
            jnp.kron(gd.Rot3(rot_params[0], rot_params[1], rot_params[2]), qnp.eye(2))
            @ jnp.kron(qnp.eye(2), gd.Rotx(theta))
            @ qnp.eye(4)
        )
        true_mat = jnp.array(true_mat)

        assert jnp.allclose(mat, true_mat)

    @pytest.mark.torch
    def test_prod_torch(self):
        """Test matrix is cast correctly using torch parameters."""
        import torch

        theta = torch.tensor(1.23)
        rot_params = torch.tensor([0.12, 3.45, 6.78])

        prod_op = Prod(
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qml.RX(theta, wires=1),
            qml.Identity(wires=0),
        )
        mat = prod_op.matrix()

        true_mat = (
            qnp.kron(gd.Rot3(rot_params[0], rot_params[1], rot_params[2]), qnp.eye(2))
            @ qnp.kron(qnp.eye(2), gd.Rotx(theta))
            @ qnp.eye(4)
        )
        true_mat = torch.tensor(true_mat)
        true_mat = torch.tensor(true_mat, dtype=torch.complex64)

        assert torch.allclose(mat, true_mat)

    @pytest.mark.tf
    def test_prod_tf(self):
        """Test matrix is cast correctly using tf parameters."""
        import tensorflow as tf

        theta = tf.Variable(1.23)
        rot_params = tf.Variable([0.12, 3.45, 6.78])

        prod_op = Prod(
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qml.RX(theta, wires=1),
            qml.Identity(wires=0),
        )
        mat = prod_op.matrix()

        true_mat = (
            qnp.kron(gd.Rot3(0.12, 3.45, 6.78), qnp.eye(2))
            @ qnp.kron(qnp.eye(2), gd.Rotx(1.23))
            @ qnp.eye(4)
        )
        true_mat = tf.Variable(true_mat)
        true_mat = tf.Variable(true_mat, dtype=tf.complex128)

        assert isinstance(mat, tf.Tensor)
        assert mat.dtype == true_mat.dtype
        assert np.allclose(mat, true_mat)


class TestProperties:
    """Test class properties."""

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_params(self, ops_lst):
        """Test num_params property updates correctly."""
        prod_op = Prod(*ops_lst)
        true_num_params = sum(op.num_params for op in ops_lst)

        assert prod_op.num_params == true_num_params

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_wires(self, ops_lst):
        """Test num_wires property updates correctly."""
        prod_op = Prod(*ops_lst)
        true_wires = set()

        for op in ops_lst:
            true_wires = true_wires.union(op.wires.toset())

        assert prod_op.num_wires == len(true_wires)

    @pytest.mark.parametrize(
        "ops_lst, hermitian_status",
        [(ops_tup, status) for ops_tup, status in zip(ops, ops_hermitian_status)],
    )
    def test_is_hermitian(self, ops_lst, hermitian_status):
        """Test is_hermitian property updates correctly."""
        prod_op = prod(*ops_lst)
        assert prod_op.is_hermitian == hermitian_status

    @pytest.mark.tf
    @pytest.mark.xfail  # this will fail until we can support is_hermitian checks for parametric ops
    def test_is_hermitian_tf(self):
        """Test that is_hermitian works when a tf type scalar is provided."""
        import tensorflow as tf

        theta = tf.Variable(1.23)
        ops = (
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=0), qml.PauliZ(wires=1)),
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=1), qml.PauliZ(wires=2)),
        )
        true_hermitian_states = (True, False)

        for op, hermitian_state in zip(ops, true_hermitian_states):
            assert op.is_hermitian == hermitian_state

    @pytest.mark.jax
    @pytest.mark.xfail  # this will fail until we can support is_hermitian checks for parametric ops
    def test_is_hermitian_jax(self):
        """Test that is_hermitian works when a jax type scalar is provided."""
        import jax.numpy as jnp

        theta = jnp.array(1.23)
        ops = (
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=0), qml.PauliZ(wires=1)),
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=1), qml.PauliZ(wires=2)),
        )
        true_hermitian_states = (True, False)

        for op, hermitian_state in zip(ops, true_hermitian_states):
            assert op.is_hermitian == hermitian_state

    @pytest.mark.torch
    @pytest.mark.xfail  # this will fail until we can support is_hermitian checks for parametric ops
    def test_is_hermitian_torch(self):
        """Test that is_hermitian works when a torch type scalar is provided."""
        import torch

        theta = torch.tensor(1.23)
        ops = (
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=0), qml.PauliZ(wires=1)),
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=1), qml.PauliZ(wires=2)),
        )
        true_hermitian_states = (True, False)

        for op, hermitian_state in zip(ops, true_hermitian_states):
            assert op.is_hermitian == hermitian_state

    @pytest.mark.parametrize("ops_lst", ops)
    def test_queue_category_ops(self, ops_lst):
        """Test _queue_category property is '_ops' when all factors are `_ops`."""
        prod_op = prod(*ops_lst)
        assert prod_op._queue_category == "_ops"

    def test_queue_category_none(self):
        """Test _queue_category property is None when any factor is not `_ops`."""

        class DummyOp(Operator):
            """Dummy op with None queue category"""

            _queue_category = None

            def __init__(self, wires):
                self._wires = qml.wires.Wires([wires])

            def num_wires(self):
                return len(self.wires)

        prod_op = prod(qml.Identity(wires=0), DummyOp(wires=0))
        assert prod_op._queue_category is None

    def test_eigendecompostion(self):
        """Test that the computed Eigenvalues and Eigenvectors are correct."""
        diag_prod_op = Prod(qml.PauliZ(wires=0), qml.PauliZ(wires=1))
        eig_decomp = diag_prod_op.eigendecomposition
        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        true_eigvecs = qnp.tensor(
            [  # the eigvecs ordered according to eigvals ordered smallest --> largest
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        true_eigvals = qnp.tensor([-1.0, -1.0, 1.0, 1.0])

        assert np.allclose(eig_vals, true_eigvals)
        assert np.allclose(eig_vecs, true_eigvecs)

    def test_eigen_caching(self):
        """Test that the eigendecomposition is stored in cache."""
        diag_prod_op = Prod(qml.PauliZ(wires=0), qml.PauliZ(wires=1))
        eig_decomp = diag_prod_op.eigendecomposition

        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]
        eigs_cache = diag_prod_op._eigs[
            (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        ]
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)

    def test_diagonalizing_gates(self):
        """Test that the diagonalizing gates are correct."""
        diag_prod_op = Prod(qml.PauliZ(wires=0), qml.PauliZ(wires=1))
        diagonalizing_gates = diag_prod_op.diagonalizing_gates()[0].matrix()
        true_diagonalizing_gates = qnp.array(
            (  # the gates that swap eigvals till they are ordered smallest --> largest
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )

        assert np.allclose(diagonalizing_gates, true_diagonalizing_gates)


class TestSimplify:
    """Test Prod simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        prod_op = (
            qml.RZ(1.32, wires=0) @ qml.Identity(wires=0) @ qml.RX(1.9, wires=1) @ qml.PauliX(0)
        )
        assert prod_op.arithmetic_depth == 3

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        prod_op = qml.RZ(1.32, wires=0) @ qml.Identity(wires=0) @ qml.RX(1.9, wires=1)
        final_op = Prod(qml.RZ(1.32, wires=0), qml.RX(1.9, wires=1))
        simplified_op = prod_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, Prod)
        for s1, s2 in zip(final_op.factors, simplified_op.factors):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_method_with_identity(self):
        """Test that the simplify method of a product of an operator with an identity returns
        the operator."""
        prod_op = Prod(qml.PauliX(0), qml.Identity(0))
        final_op = qml.PauliX(0)
        simplified_op = prod_op.simplify()

        assert isinstance(simplified_op, qml.PauliX)

    def test_simplify_method_product_of_sums(self):
        """Test the simplify method with a product of sums."""
        prod_op = Prod(qml.PauliX(0) + qml.RX(1, 0), qml.PauliX(1) + qml.RX(1, 1), qml.Identity(3))
        final_op = qml.op_sum(
            Prod(qml.PauliX(0), qml.PauliX(1)),
            qml.PauliX(0) @ qml.RX(1, 1),
            qml.RX(1, 0) @ qml.PauliX(1),
            qml.RX(1, 0) @ qml.RX(1, 1),
        )
        simplified_op = prod_op.simplify()
        assert isinstance(simplified_op, qml.ops.Sum)
        for s1, s2 in zip(final_op.summands, simplified_op.summands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_with_nested_prod_and_adjoints(self):
        """Test simplify method with nested product and adjoint operators."""
        prod_op = Prod(qml.adjoint(Prod(qml.RX(1, 0), qml.RY(1, 0))), qml.RZ(1, 0))
        final_op = Prod(qml.RY(-1, 0), qml.RX(-1, 0), qml.RZ(1, 0))
        simplified_op = prod_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, Prod)
        for s1, s2 in zip(final_op.factors, simplified_op.factors):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_method_with_nested_ops(self):
        """Test the simplify method with nested operators."""
        prod_op = Prod(
            Prod(
                qml.PauliX(0)
                + qml.adjoint(
                    Prod(
                        qml.PauliX(0),
                        qml.RX(1, 0) + qml.PauliX(0),
                        qml.Identity(0),
                    )
                ),
                qml.PauliX(1) + 5 * (qml.RX(1, 1) + qml.PauliX(1)),
            ),
            qml.Identity(0),
        )
        final_op = qml.op_sum(
            Prod(qml.PauliX(0), qml.PauliX(1)),
            qml.PauliX(0) @ (5 * qml.RX(1, 1)),
            qml.PauliX(0) @ qml.s_prod(5, qml.PauliX(1)),
            Prod(qml.RX(-1, 0), qml.PauliX(0), qml.PauliX(1)),
            Prod(qml.RX(-1, 0), qml.PauliX(0), 5 * qml.RX(1, 1)),
            Prod(qml.RX(-1, 0), qml.PauliX(0), qml.s_prod(5, qml.PauliX(1))),
            Prod(qml.PauliX(0), qml.PauliX(0), qml.PauliX(1)),
            Prod(qml.PauliX(0), qml.PauliX(0), 5 * qml.RX(1, 1)),
            Prod(qml.PauliX(0), qml.PauliX(0), qml.s_prod(5, qml.PauliX(1))),
        )
        simplified_op = prod_op.simplify()
        assert isinstance(simplified_op, qml.ops.Sum)
        for s1, s2 in zip(final_op.summands, simplified_op.summands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth


class TestWrapperFunc:
    """Test wrapper function."""

    def test_op_prod_top_level(self):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        factors = (qml.PauliX(wires=1), qml.RX(1.23, wires=0), qml.CNOT(wires=[0, 1]))
        op_id = "prod_op"
        do_queue = False

        prod_func_op = prod(*factors, id=op_id, do_queue=do_queue)
        prod_class_op = Prod(*factors, id=op_id, do_queue=do_queue)

        assert prod_class_op.factors == prod_func_op.factors
        assert np.allclose(prod_class_op.matrix(), prod_func_op.matrix())
        assert prod_class_op.id == prod_func_op.id
        assert prod_class_op.wires == prod_func_op.wires
        assert prod_class_op.parameters == prod_func_op.parameters


class TestIntegration:
    """Integration tests for the Prod class."""

    def test_measurement_process_expval(self):
        """Test Prod class instance in expval measurement process."""
        dev = qml.device("default.qubit", wires=2)
        prod_op = Prod(qml.PauliZ(wires=0), qml.Hadamard(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.expval(prod_op)

        exp_val = my_circ()
        true_exp_val = qnp.array(-1 / qnp.sqrt(2))
        assert qnp.allclose(exp_val, true_exp_val)

    def test_measurement_process_var(self):
        """Test Prod class instance in var measurement process."""
        dev = qml.device("default.qubit", wires=2)
        prod_op = Prod(qml.PauliZ(wires=0), qml.Hadamard(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.var(prod_op)

        var = my_circ()
        true_var = qnp.array(1 / 2)
        assert qnp.allclose(var, true_var)

    def test_measurement_process_probs(self):
        """Test Prod class instance in probs measurement process raises error."""  # currently can't support due to bug
        dev = qml.device("default.qubit", wires=2)
        prod_op = Prod(qml.PauliX(wires=0), qml.Hadamard(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.probs(op=prod_op)

        with pytest.raises(
            QuantumFunctionError,
            match="Symbolic Operations are not supported for " "rotating probabilities yet.",
        ):
            my_circ()

    def test_measurement_process_sample(self):
        """Test Prod class instance in sample measurement process raises error."""  # currently can't support due to bug
        dev = qml.device("default.qubit", wires=2, shots=2)
        prod_op = Prod(qml.PauliX(wires=0), qml.Hadamard(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.sample(op=prod_op)

        with pytest.raises(
            QuantumFunctionError,
            match="Symbolic Operations are not supported for sampling yet.",
        ):
            my_circ()

    def test_differentiable_measurement_process(self):
        """Test that the gradient can be computed with a Prod op in the measurement process."""
        prod_op = Prod(qml.PauliZ(wires=0), qml.Hadamard(wires=1))
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            return qml.expval(prod_op)

        weights = qnp.array([0.1], requires_grad=True)
        grad = qml.grad(circuit)(weights)

        true_grad = -qnp.sqrt(2) * qnp.cos(weights[0] / 2) * qnp.sin(weights[0] / 2)
        assert qnp.allclose(grad, true_grad)

    def test_non_hermitian_op_in_measurement_process(self):
        """Test that non-hermitian ops in a measurement process will raise an error."""
        wires = [0, 1]
        dev = qml.device("default.qubit", wires=wires)
        prod_op = Prod(qml.RX(1.23, wires=0), qml.Identity(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.expval(prod_op)

        with pytest.raises(QuantumFunctionError, match="Prod is not an observable:"):
            my_circ()

    def test_operation_integration(self):
        """Test that a Product operation can be queued and executed in a circuit"""
        dev = qml.device("default.qubit", wires=3)
        ops = (
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.RX(1.23, wires=1),
            qml.IsingZZ(0.56, wires=[0, 2]),
        )

        @qml.qnode(dev)
        def prod_state_circ():
            Prod(*ops)
            return qml.state()

        @qml.qnode(dev)
        def true_state_circ():
            qml.IsingZZ(0.56, wires=[0, 2])
            qml.RX(1.23, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)
            return qml.state()

        assert qnp.allclose(prod_state_circ(), true_state_circ())
