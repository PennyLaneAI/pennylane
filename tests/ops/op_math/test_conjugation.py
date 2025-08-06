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
Unit tests for the Conjugation arithmetic class of qubit operations
"""

# pylint:disable=protected-access, unused-argument
import gate_data as gd  # a file containing matrix rep of each gate
import numpy as np
import pytest

import pennylane as qml
import pennylane.numpy as qnp
from pennylane import PauliX, math
from pennylane.exceptions import DeviceError, MatrixUndefinedError
from pennylane.operation import Operator
from pennylane.ops import prod
from pennylane.ops.op_math.conjugation import Conjugation, conjugation
from pennylane.wires import Wires

X, Y, Z = qml.PauliX, qml.PauliY, qml.PauliZ

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
    (qml.PauliZ(0), qml.PauliX(1), qml.PauliZ(0)),
    (qml.Hadamard(wires=0), qml.PauliZ(wires=0), qml.Hadamard(wires=0)),
    (qml.CNOT(wires=[0, 1]), qml.RX(1.23, wires=1), qml.CNOT(wires=[0, 1])),
)


def test_basic_validity():
    """Run basic validity checks on a conjugation operator."""
    op1 = qml.PauliZ(0)
    op2 = qml.Rot(1.2, 2.3, 3.4, wires=0)
    op3 = qml.PauliZ(0)
    op = qml.conjugation(op1, op2, op3)
    qml.ops.functions.assert_valid(op)


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


class MyOp(qml.RX):  # pylint:disable=too-few-public-methods
    """Variant of qml.RX that claims to not have `adjoint` or a matrix defined."""

    has_matrix = False
    has_adjoint = False
    has_decomposition = False
    has_diagonalizing_gates = False


class TestInitialization:  # pylint:disable=too-many-public-methods
    """Test the initialization."""

    def test_init_conjugation_op(self):
        """Test the initialization of a Conjugation operator."""
        conjugation_op = Conjugation(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"))

        assert conjugation_op.wires == Wires((0, "a"))
        assert conjugation_op.num_wires == 2
        assert conjugation_op.name == "Conjugation"

        assert conjugation_op.data == (0.23,)
        assert conjugation_op.parameters == [0.23]
        assert conjugation_op.num_params == 1

    def test_hash(self):
        """Testing some situations for the hash property."""
        # test not the same hash if different order
        op1 = qml.conjugation(qml.PauliX("a"), qml.PauliY("a"), qml.PauliX(1))
        op2 = qml.conjugation(qml.PauliY("a"), qml.PauliX("a"), qml.PauliX(1))
        assert op1.hash != op2.hash

    PROD_TERMS_OP_PAIRS = (  # not all operands have pauli representation
        (
            qml.conjugation(qml.Hadamard(0), X(1), qml.Hadamard(0)),
            [1.0],
            [qml.conjugation(qml.Hadamard(0), X(1), qml.Hadamard(0))],
        ),  # trivial conjugation
        (
            qml.conjugation(X(0), X(1), X(0)),
            [1.0],
            [qml.conjugation(X(0), X(1), X(0))],
        ),  # trivial conjugation
        (
            qml.conjugation(qml.Hadamard(0), X(1)),
            [1.0],
            [qml.conjugation(qml.Hadamard(0), X(1), qml.Hadamard(0))],
        ),  # conjugation without adjoint provided
    )

    def test_batch_size(self):
        """Test that batch size returns the batch size of a base operation if it is batched."""
        x = qml.numpy.array([1.0, 2.0, 3.0])
        conjugation_op = conjugation(qml.PauliX(0), qml.RX(x, wires=0))
        assert conjugation_op.batch_size == 3

    def test_batch_size_None(self):
        """Test that the batch size is none if no factors have batching."""
        conjugation_op = conjugation(qml.PauliX(0), qml.RX(1.0, wires=0))
        assert conjugation_op.batch_size is None

    @pytest.mark.parametrize("ops_lst", ops)
    def test_decomposition(self, ops_lst):
        """Test the decomposition of a conjugation of operators is a list
        of the provided factors."""
        conjugation_op = conjugation(*ops_lst)
        decomposition = conjugation_op.decomposition()
        true_decomposition = list(ops_lst[::-1])  # reversed list of factors

        assert isinstance(decomposition, list)
        for op1, op2 in zip(decomposition, true_decomposition):
            qml.assert_equal(op1, op2)

    @pytest.mark.parametrize("ops_lst", ops)
    def test_decomposition_on_tape(self, ops_lst):
        """Test the decomposition of a conjugation of operators is a list
        of the provided factors on a tape."""
        conjugation_op = conjugation(*ops_lst)
        true_decomposition = list(ops_lst[::-1])  # reversed list of factors
        with qml.queuing.AnnotatedQueue() as q:
            conjugation_op.decomposition()

        tape = qml.tape.QuantumScript.from_queue(q)
        for op1, op2 in zip(tape.operations, true_decomposition):
            qml.assert_equal(op1, op2)

    def test_eigen_caching(self):
        """Test that the eigendecomposition is stored in cache."""
        conjugation_op = Conjugation(qml.PauliZ(wires=0), qml.Identity(wires=1))
        eig_decomp = conjugation_op.eigendecomposition

        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        eigs_cache = conjugation_op._eigs[conjugation_op.hash]
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)

    def test_has_matrix_true_via_factors_have_matrix(self):
        """Test that a conjugation of operators that have `has_matrix=True`
        has `has_matrix=True` as well."""

        conjugation_op = conjugation(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"))
        assert conjugation_op.has_matrix is True

    def test_has_matrix_true_via_factor_has_no_matrix_but_is_hamiltonian(self):
        """Test that a conjugation of operators of which one does not have `has_matrix=True`
        but is a Hamiltonian has `has_matrix=True`."""

        H = qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])
        conjugation_op = conjugation(H, qml.RZ(0.23, wires=5))
        assert conjugation_op.has_matrix is True

    @pytest.mark.parametrize(
        "first_factor", [qml.PauliX(wires=0), qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])]
    )
    def test_has_matrix_false_via_factor_has_no_matrix(self, first_factor):
        """Test that a conjugation of operators of which one does not have `has_matrix=True`
        has `has_matrix=False`."""

        conjugation_op = conjugation(first_factor, MyOp(0.23, wires="a"))
        assert conjugation_op.has_matrix is False

    @pytest.mark.parametrize(
        "factors",
        (
            [qml.PauliX(wires=0), qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])],
            [qml.PauliX(wires=0), qml.RZ(0.612, "r")],
            [
                qml.Hamiltonian([-0.3], [qml.PauliZ(wires=1)]),
                qml.Hamiltonian([0.5], [qml.PauliX(wires=1)]),
            ],
            [MyOp(3.1, 0), qml.CNOT([0, 2])],
        ),
    )
    def test_has_adjoint_true_always(self, factors):
        """Test that a conjugation of operators that have `has_matrix=True`
        has `has_matrix=True` as well."""

        conjugation_op = conjugation(*factors)
        assert conjugation_op.has_adjoint is True

    @pytest.mark.parametrize(
        "factors",
        (
            [qml.PauliX(wires=0), qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])],
            [qml.PauliX(wires=0), qml.RZ(0.612, "r")],
            [
                qml.Hamiltonian([-0.3], [qml.PauliZ(wires=1)]),
                qml.Hamiltonian([0.5], [qml.PauliX(wires=1)]),
            ],
            [MyOp(3.1, 0), qml.CNOT([0, 2])],
        ),
    )
    def test_has_decomposition_true_always(self, factors):
        """Test that a conjugation of operators that have `has_decomposition=True`
        has `has_decomposition=True` as well."""

        conjugation_op = conjugation(*factors)
        assert conjugation_op.has_decomposition is True

    @pytest.mark.parametrize(
        "factors",
        (
            [qml.PauliX(wires=0), qml.PauliY(wires=0)],
            [qml.PauliX(wires=0), qml.PauliZ(wires="r"), qml.PauliY(0)],
            [qml.Hamiltonian([0.523], [qml.PauliX(3)]), qml.PauliZ(3)],
            [
                qml.Hamiltonian([0.523], [qml.PauliX(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([-1.301], [qml.PauliY(2) @ qml.PauliZ(1)]),
            ],
        ),
    )
    def test_has_diagonalizing_gates_true_via_overlapping_factors(self, factors):
        """Test that a conjugation of operators that have `has_diagonalizing_gates=True`
        has `has_diagonalizing_gates=True` as well."""

        conjugation_op = conjugation(*factors)
        assert conjugation_op.has_diagonalizing_gates is True

    @pytest.mark.parametrize(
        "factors",
        (
            [qml.PauliX(wires=0), qml.PauliX(wires=1)],
            [qml.PauliX(wires=0), qml.PauliZ(wires="r")],
            [qml.Hermitian(np.eye(4), wires=[0, 2]), qml.PauliX(wires=1)],
        ),
    )
    def test_has_diagonalizing_gates_true_via_factors(self, factors):
        """Test that a conjugation of operators that have `has_diagonalizing_gates=True`
        has `has_diagonalizing_gates=True` as well."""

        conjugation_op = conjugation(*factors)
        assert conjugation_op.has_diagonalizing_gates is True

    def test_has_diagonalizing_gates_false_via_factor(self):
        """Test that a conjugation of operators of which one has
        `has_diagonalizing_gates=False` has `has_diagonalizing_gates=False` as well."""

        conjugation_op = conjugation(MyOp(3.1, 0), qml.PauliX(2))
        assert conjugation_op.has_diagonalizing_gates is False


# pylint: disable=too-many-public-methods
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
        """Test matrix method for a conjugation of non_parametric ops"""
        mat1, mat2 = compare_and_expand_mat(mat1, mat2)
        true_mat = mat1 @ mat2 @ np.conj(mat1).T

        conjugation_op = Conjugation(
            op1(wires=0 if op1.num_wires is None else range(op1.num_wires)),
            op2(wires=0 if op2.num_wires is None else range(op2.num_wires)),
        )
        conjugation_mat = conjugation_op.matrix()

        assert np.allclose(conjugation_mat, true_mat)

    @pytest.mark.parametrize("op1, mat1", param_ops)
    @pytest.mark.parametrize("op2, mat2", param_ops)
    def test_parametric_ops_two_terms(
        self,
        op1: Operator,
        mat1: np.ndarray,
        op2: Operator,
        mat2: np.ndarray,
    ):
        """Test matrix method for a conjugation of parametric ops"""
        par1 = tuple(range(op1.num_params))
        par2 = tuple(range(op2.num_params))
        mat1, mat2 = compare_and_expand_mat(mat1(*par1), mat2(*par2))

        conjugation_op = Conjugation(
            op1(*par1, wires=range(op1.num_wires)), op2(*par2, wires=range(op2.num_wires))
        )
        conjugation_mat = conjugation_op.matrix()
        true_mat = mat1 @ mat2 @ np.conj(mat1).T
        assert np.allclose(conjugation_mat, true_mat)

    @pytest.mark.parametrize("op", no_mat_ops)
    def test_error_no_mat(self, op: Operator):
        """Test that an error is raised if one of the factors doesn't
        have its matrix method defined."""
        conjugation_op = Conjugation(op(wires=0), qml.PauliX(wires=2), qml.PauliZ(wires=1))
        with pytest.raises(MatrixUndefinedError):
            conjugation_op.matrix()

    def test_conjugation_ops_multi_terms(self):
        """Test matrix is correct for a conjugation of more than two terms."""
        conjugation_op = Conjugation(qml.PauliX(wires=0), qml.PauliY(wires=0), qml.PauliX(wires=0))
        mat = conjugation_op.matrix()

        true_mat = math.array(
            [
                [0, 1j],
                [-1j, 0],
            ]
        )
        assert np.allclose(mat, true_mat)

    def test_conjugation_ops_multi_wires(self):
        """Test matrix is correct when multiple wires are used in the conjugation."""
        conjugation_op = Conjugation(
            qml.PauliX(wires=0), qml.Hadamard(wires=1), qml.PauliX(wires=0)
        )
        mat = conjugation_op.matrix()

        x = math.array([[0, 1], [1, 0]])
        h = 1 / math.sqrt(2) * math.array([[1, 1], [1, -1]])
        I = math.array([[1, 0], [0, 1]])

        true_mat = math.kron(x, I) @ math.kron(I, h) @ math.kron(x, I)
        assert np.allclose(mat, true_mat)

    def test_conjugation_ops_wire_order(self):
        """Test correct matrix is returned when the wire_order arg is provided."""
        conjugation_op = Conjugation(
            qml.Hadamard(wires=1), qml.PauliX(wires=0), qml.Hadamard(wires=1)
        )
        wire_order = [0, 1]
        mat = conjugation_op.matrix(wire_order=wire_order)

        x = math.array([[0, 1], [1, 0]])
        h = 1 / math.sqrt(2) * math.array([[1, 1], [1, -1]])
        I = math.array([[1, 0], [0, 1]])

        true_mat = math.kron(I, h) @ math.kron(x, I) @ math.kron(I, h)
        assert np.allclose(mat, true_mat)

    def test_conjugation_templates(self):
        """Test that we can compose templates and the generated matrix is correct."""

        def get_qft_mat(num_wires):
            omega = math.exp(np.pi * 1.0j / 2 ** (num_wires - 1))
            mat = math.zeros((2**num_wires, 2**num_wires), dtype="complex128")

            for m in range(2**num_wires):
                for n in range(2**num_wires):
                    mat[m, n] = omega ** (m * n)

            return 1 / math.sqrt(2**num_wires) * mat

        wires = [0, 1, 2]
        conjugation_op = Conjugation(qml.QFT(wires=wires), qml.GroverOperator(wires=wires))
        mat = conjugation_op.matrix()

        grov_mat = (1 / 4) * math.ones((8, 8), dtype="complex128") - math.eye(8, dtype="complex128")
        qft_mat = get_qft_mat(3)

        true_mat = qft_mat @ grov_mat @ np.conj(qft_mat).T
        assert np.allclose(mat, true_mat)

    def test_conjugation_qubit_unitary(self):
        """Test that an arbitrary QubitUnitary can be composed with correct matrix representation."""
        U = 1 / qnp.sqrt(2) * qnp.array([[1, 1], [1, -1]])  # Hadamard
        U_op = qml.QubitUnitary(U, wires=0)

        conjugation_op = Conjugation(U_op, qml.Identity(wires=1))
        mat = conjugation_op.matrix()

        true_mat = qnp.eye(4)
        assert np.allclose(mat, true_mat)

    def test_conjugation_hamiltonian(self):
        """Test that a hamiltonian object can be composed."""
        U = qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])
        conjugation_op = Conjugation(qml.PauliZ(wires=0), U)
        mat = conjugation_op.matrix()

        z = math.array([[1, 0], [0, -1]])
        I = math.array([[1, 0], [0, 1]])
        true_mat = [
            [0.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5, 0.0],
        ] @ np.kron(z, I)
        assert np.allclose(mat, true_mat)

    def test_matrix_all_batched(self):
        """Test that Conjugation matrix has batching support when all operands are batched."""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        y = qml.numpy.array([0.4, 0.5, 0.6])
        op = conjugation(qml.RX(x, wires=0), qml.RY(y, wires=2))
        mat = op.matrix()
        sum_list = [conjugation(qml.RX(i, wires=0), qml.RY(j, wires=2)) for i, j in zip(x, y)]
        compare = qml.math.stack([s.matrix() for s in sum_list])
        assert qml.math.allclose(mat, compare)
        assert mat.shape == (3, 4, 4)

    def test_matrix_not_all_batched(self):
        """Test that Conjugation matrix has batching support when all operands are not batched."""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        y = 0.5
        op = conjugation(qml.RX(x, wires=0), qml.RY(y, wires=2))
        mat = op.matrix()
        batched_y = [y for _ in x]
        sum_list = [
            conjugation(qml.RX(i, wires=0), qml.RY(j, wires=2)) for i, j in zip(x, batched_y)
        ]
        compare = qml.math.stack([s.matrix() for s in sum_list])
        assert qml.math.allclose(mat, compare)
        assert mat.shape == (3, 4, 4)

    @pytest.mark.jax
    def test_conjugation_jax(self):
        """Test matrix is cast correctly using jax parameters."""
        import jax.numpy as jnp

        theta = jnp.array(1.23)
        rot_params = jnp.array([0.12, 3.45, 6.78])

        conjugation_op = Conjugation(
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0), qml.RX(theta, wires=1)
        )
        mat = conjugation_op.matrix()

        true_mat = (
            jnp.kron(gd.Rot3(rot_params[0], rot_params[1], rot_params[2]), qnp.eye(2))
            @ jnp.kron(qnp.eye(2), gd.Rotx(theta))
            @ jnp.conj(jnp.kron(gd.Rot3(rot_params[0], rot_params[1], rot_params[2]), qnp.eye(2))).T
        )
        true_mat = jnp.array(true_mat)

        assert jnp.allclose(mat, true_mat)

    @pytest.mark.torch
    def test_conjugation_torch(self):
        """Test matrix is cast correctly using torch parameters."""
        import torch

        theta = torch.tensor(1.23)
        rot_params = torch.tensor([0.12, 3.45, 6.78])

        conjugation_op = Conjugation(
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0), qml.RX(theta, wires=1)
        )
        mat = conjugation_op.matrix()

        true_mat = (
            qnp.kron(gd.Rot3(rot_params[0], rot_params[1], rot_params[2]), qnp.eye(2))
            @ qnp.kron(qnp.eye(2), gd.Rotx(theta))
            @ qnp.conj(qnp.kron(gd.Rot3(rot_params[0], rot_params[1], rot_params[2]), qnp.eye(2))).T
        )
        true_mat = torch.tensor(true_mat, dtype=torch.complex64)

        assert torch.allclose(mat, true_mat)

    @pytest.mark.tf
    def test_conjugation_tf(self):
        """Test matrix is cast correctly using tf parameters."""
        import tensorflow as tf

        theta = tf.Variable(1.23)
        rot_params = tf.Variable([0.12, 3.45, 6.78])

        conjugation_op = Conjugation(
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0), qml.RX(theta, wires=1)
        )
        mat = conjugation_op.matrix()

        true_mat = (
            qnp.kron(gd.Rot3(0.12, 3.45, 6.78), qnp.eye(2))
            @ qnp.kron(qnp.eye(2), gd.Rotx(1.23))
            @ qnp.conj(qnp.kron(gd.Rot3(0.12, 3.45, 6.78), qnp.eye(2))).T
        )
        true_mat = tf.Variable(true_mat)
        true_mat = tf.Variable(true_mat, dtype=tf.complex128)

        assert isinstance(mat, tf.Tensor)
        assert mat.dtype == true_mat.dtype
        assert np.allclose(mat, true_mat)

    # sparse matrix tests:

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix of a Conjugation op is defined and correct."""
        conjugation_op = conjugation(op1(wires=0), op2(wires=1))
        true_mat = (
            math.kron(mat1, np.eye(2))
            @ math.kron(np.eye(2), mat2)
            @ np.conj(math.kron(mat1, np.eye(2))).T
        )
        conjugation_mat = conjugation_op.sparse_matrix().todense()

        assert np.allclose(true_mat, conjugation_mat)

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix_same_wires(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix of a Conjugation op is defined and correct."""
        conjugation_op = conjugation(op1(wires=0), op2(wires=0))
        true_mat = mat1 @ mat2 @ np.conj(mat1).T
        conjugation_mat = conjugation_op.sparse_matrix().todense()

        assert np.allclose(true_mat, conjugation_mat)

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix_wire_order(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix of a Conjugation op is defined
        with wire order and correct."""
        true_mat = (
            math.kron(mat1, np.eye(2))
            @ math.kron(np.eye(2), mat2)
            @ np.conj(math.kron(mat1, np.eye(2))).T
        )

        conjugation_op = conjugation(op1(wires=1), op2(wires=0))
        conjugation_mat = conjugation_op.sparse_matrix(wire_order=[1, 0]).todense()

        assert np.allclose(true_mat, conjugation_mat)

    def test_sparse_matrix_undefined_error(self):
        """Test that an error is raised when the sparse matrix method
        is undefined for any of the factors."""

        class DummyOp(qml.operation.Operation):  # pylint:disable=too-few-public-methods
            num_wires = 1

            def sparse_matrix(self, wire_order=None):
                raise qml.operation.SparseMatrixUndefinedError

        conjugation_op = conjugation(qml.PauliX(wires=0), DummyOp(wires=1))

        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            conjugation_op.sparse_matrix()


class TestProperties:
    """Test class properties."""

    @pytest.mark.parametrize("ops_lst", ops)
    def test_queue_category_ops(self, ops_lst):
        """Test _queue_category property is '_ops' when all factors are `_ops`."""
        conjugation_op = conjugation(*ops_lst)
        assert conjugation_op._queue_category == "_ops"

    def test_eigen_caching(self):
        """Test that the eigendecomposition is stored in cache."""
        diag_conjugation_op = Conjugation(qml.PauliZ(wires=0), qml.PauliZ(wires=1))
        eig_decomp = diag_conjugation_op.eigendecomposition

        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]
        eigs_cache = diag_conjugation_op._eigs[diag_conjugation_op.hash]
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)

    @pytest.mark.jax
    def test_eigvals_jax_jit(self):
        """Assert computing the eigvals of a Conjugation is compatible with jax-jit."""
        import jax

        def f(t1, t2):
            return qml.conjugation(qml.RX(t1, 0), qml.RX(t2, 0)).eigvals()

        assert qml.math.allclose(f(0.5, 1.0), jax.jit(f)(0.5, 1.0))


class TestWrapperFunc:
    """Test wrapper function."""

    def test_op_conjugation_top_level(self):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        factors = (qml.PauliX(wires=1), qml.RX(1.23, wires=0), qml.CNOT(wires=[0, 1]))
        op_id = "conjugation_op"

        conjugation_func_op = conjugation(*factors)
        conjugation_class_op = Conjugation(*factors)
        qml.assert_equal(conjugation_func_op, conjugation_class_op)


class TestIntegration:
    """Integration tests for the Conjugation class."""

    def test_measurement_process_expval(self):
        """Test Conjugation class instance in expval measurement process."""
        dev = qml.device("default.qubit", wires=2)
        conjugation_op = Conjugation(qml.PauliZ(wires=0), qml.Hadamard(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.expval(conjugation_op)

        exp_val = my_circ()
        true_exp_val = qnp.array(-1 / qnp.sqrt(2))
        assert qnp.allclose(exp_val, true_exp_val)

    def test_measurement_process_var(self):
        """Test Conjugation class instance in var measurement process."""
        dev = qml.device("default.qubit", wires=2)
        conjugation_op = Conjugation(qml.PauliZ(wires=0), qml.Hadamard(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.var(conjugation_op)

        var = my_circ()
        true_var = qnp.array(1 / 2)
        assert qnp.allclose(var, true_var)

    def test_measurement_process_probs(self):
        """Test Conjugation class instance in probs measurement process raises error."""
        dev = qml.device("default.qubit", wires=2)
        conjugation_op = Conjugation(qml.PauliX(wires=0), qml.Hadamard(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.probs(op=conjugation_op)

        x_probs = np.array([0.5, 0.5])
        h_probs = np.array([np.cos(-np.pi / 4 / 2) ** 2, np.sin(-np.pi / 4 / 2) ** 2])
        expected = np.tensordot(x_probs, h_probs, axes=0).flatten()
        out = my_circ()
        assert qml.math.allclose(out, expected)

    def test_differentiable_measurement_process(self):
        """Test that the gradient can be computed with a Conjugation op in the measurement process."""
        conjugation_op = Conjugation(qml.PauliZ(wires=0), qml.Hadamard(wires=1))
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            return qml.expval(conjugation_op)

        weights = qnp.array([0.1], requires_grad=True)
        grad = qml.grad(circuit)(weights)

        true_grad = -qnp.sqrt(2) * qnp.cos(weights[0] / 2) * qnp.sin(weights[0] / 2)
        assert qnp.allclose(grad, true_grad)

    def test_non_supported_obs_not_supported(self):
        """Test that non-supported ops in a measurement process will raise an error."""
        wires = [0, 1]
        dev = qml.device("default.qubit", wires=wires)
        conjugation_op = Conjugation(qml.RX(1.23, wires=0), qml.Identity(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.expval(conjugation_op)

        with pytest.raises(DeviceError):
            my_circ()

    def test_operation_integration(self):
        """Test that a Conjugationuct operation can be queued and executed in a circuit"""
        dev = qml.device("default.qubit", wires=3)
        operands = (
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
        )

        @qml.qnode(dev)
        def conjugation_state_circ():
            Conjugation(*operands)
            return qml.state()

        @qml.qnode(dev)
        def true_state_circ():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)
            return qml.state()

        assert qnp.allclose(conjugation_state_circ(), true_state_circ())

    def test_batched_operation(self):
        """Test that conjugation with batching gives expected results."""
        x = qml.numpy.array([1.0, 2.0, 3.0])
        y = qml.numpy.array([4.0, 5.0, 6.0])

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def batched_conjugation(x, y):
            qml.conjugation(qml.RX(x, wires=0), qml.RY(y, wires=0))
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def batched_no_conjugation(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.adjoint(qml.RX(x, wires=0))
            return qml.expval(qml.PauliZ(0))

        res1 = batched_conjugation(x, y)
        res2 = batched_no_conjugation(x, y)
        assert qml.math.allclose(res1, res2)

    def test_params_can_be_considered_trainable(self):
        """Tests that the parameters of a Conjugation are considered trainable."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, U):
            qml.RX(x, 0)
            return qml.expval(qml.conjugation(qml.Hermitian(U, 0), qml.PauliX(1)))

        x = qnp.array(0.1, requires_grad=False)
        U = qnp.array([[1.0, 0.0], [0.0, -1.0]], requires_grad=True)

        tape = qml.workflow.construct_tape(circuit)(x, U)
        assert tape.trainable_params == [1, 2]


class TestDecomposition:

    def test_resource_keys(self):
        """Test that the resource keys of `Conjugation` are op_reps."""
        assert Conjugation.resource_keys == frozenset({"resources"})
        conjugation = qml.X(0) @ qml.Y(1) @ qml.X(2)
        resources = {qml.resource_rep(qml.X): 2, qml.resource_rep(qml.Y): 1}
        assert conjugation.resource_params == {"resources": resources}

    def test_registered_decomp(self):
        """Test that the decomposition of conjugation is registered."""

        decomps = qml.decomposition.list_decomps(Conjugation)

        default_decomp = decomps[0]
        _ops = [qml.X(0), qml.X(1), qml.X(2), qml.MultiRZ(0.5, wires=(0, 1))]
        resources = {qml.resource_rep(qml.X): 3, qml.resource_rep(qml.MultiRZ, num_wires=2): 1}

        resource_obj = default_decomp.compute_resources(resources=resources)

        assert resource_obj.num_gates == 4
        assert resource_obj.gate_counts == resources

        with qml.queuing.AnnotatedQueue() as q:
            default_decomp(operands=_ops)

        assert q.queue == _ops[::-1]

    def test_integration(self, enable_graph_decomposition):
        """Test that conjugation's can be integrated into the decomposition."""

        op = Conjugation(qml.S(0), qml.S(1))

        graph = qml.decomposition.DecompositionGraph([op], gate_set=set(qml.ops.__all__))
        graph.solve()
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op)(**op.hyperparameters)

        assert q.queue == list(op[::-1])
