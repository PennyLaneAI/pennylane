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
import gate_data as gd  # a file containing matrix rep of each gate
import numpy as np
import pytest

import pennylane as qml
import pennylane.numpy as qnp
from pennylane import QuantumFunctionError, math
from pennylane.operation import AnyWires, MatrixUndefinedError, Operator
from pennylane.ops.op_math.prod import Prod, _swappable_ops, prod
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


class MyOp(qml.RX):
    """Variant of qml.RX that claims to not have `adjoint` or a matrix defined."""

    has_matrix = False
    has_adjoint = False
    has_decomposition = False
    has_diagonalizing_gates = False


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

        for f1, f2 in zip(prod_op.operands, prod_term_ops[0].operands):
            assert qml.equal(f1, f2)

    def test_batch_size(self):
        """Test that batch size returns the batch size of a base operation if it is batched."""
        x = qml.numpy.array([1.0, 2.0, 3.0])
        prod_op = prod(qml.PauliX(0), qml.RX(x, wires=0))
        assert prod_op.batch_size == 3

    def test_batch_size_None(self):
        """Test that the batch size is none if no factors have batching."""
        prod_op = prod(qml.PauliX(0), qml.RX(1.0, wires=0))
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

    @pytest.mark.parametrize("ops_lst", ops)
    def test_decomposition_on_tape(self, ops_lst):
        """Test the decomposition of a prod of operators is a list
        of the provided factors on a tape."""
        prod_op = prod(*ops_lst)
        true_decomposition = list(ops_lst[::-1])  # reversed list of factors
        with qml.tape.QuantumTape() as tape:
            prod_op.decomposition()

        for op1, op2 in zip(tape.operations, true_decomposition):
            assert qml.equal(op1, op2)

    def test_eigen_caching(self):
        """Test that the eigendecomposition is stored in cache."""
        prod_op = Prod(qml.PauliZ(wires=0), qml.Identity(wires=1))
        eig_decomp = prod_op.eigendecomposition

        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        eigs_cache = prod_op._eigs[prod_op.hash]
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)

    def test_has_matrix_true_via_factors_have_matrix(self):
        """Test that a product of operators that have `has_matrix=True`
        has `has_matrix=True` as well."""

        prod_op = prod(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"), do_queue=True)
        assert prod_op.has_matrix is True

    def test_has_matrix_true_via_factor_has_no_matrix_but_is_hamiltonian(self):
        """Test that a product of operators of which one does not have `has_matrix=True`
        but is a Hamiltonian has `has_matrix=True`."""

        H = qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])
        prod_op = prod(H, qml.RZ(0.23, wires=5), do_queue=True)
        assert prod_op.has_matrix is True

    @pytest.mark.parametrize(
        "first_factor", [qml.PauliX(wires=0), qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])]
    )
    def test_has_matrix_false_via_factor_has_no_matrix(self, first_factor):
        """Test that a product of operators of which one does not have `has_matrix=True`
        has `has_matrix=False`."""

        prod_op = prod(first_factor, MyOp(0.23, wires="a"), do_queue=True)
        assert prod_op.has_matrix is False

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
        """Test that a product of operators that have `has_matrix=True`
        has `has_matrix=True` as well."""

        prod_op = prod(*factors, do_queue=True)
        assert prod_op.has_adjoint is True

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
        """Test that a product of operators that have `has_decomposition=True`
        has `has_decomposition=True` as well."""

        prod_op = prod(*factors, do_queue=True)
        assert prod_op.has_decomposition is True

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
        """Test that a product of operators that have `has_diagonalizing_gates=True`
        has `has_diagonalizing_gates=True` as well."""

        prod_op = prod(*factors, do_queue=True)
        assert prod_op.has_diagonalizing_gates is True

    @pytest.mark.parametrize(
        "factors",
        (
            [qml.PauliX(wires=0), qml.PauliX(wires=1)],
            [qml.PauliX(wires=0), qml.PauliZ(wires="r"), qml.PauliY("s")],
            [qml.Hermitian(np.eye(4), wires=[0, 2]), qml.PauliX(wires=1)],
        ),
    )
    def test_has_diagonalizing_gates_true_via_factors(self, factors):
        """Test that a product of operators that have `has_diagonalizing_gates=True`
        has `has_diagonalizing_gates=True` as well."""

        prod_op = prod(*factors, do_queue=True)
        assert prod_op.has_diagonalizing_gates is True

    def test_has_diagonalizing_gates_false_via_factor(self):
        """Test that a product of operators of which one has
        `has_diagonalizing_gates=False` has `has_diagonalizing_gates=False` as well."""

        prod_op = prod(MyOp(3.1, 0), qml.PauliX(2), do_queue=True)
        assert prod_op.has_diagonalizing_gates is False


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

        prod_op = Prod(
            op1(wires=0 if op1.num_wires is AnyWires else range(op1.num_wires)),
            op2(wires=0 if op2.num_wires is AnyWires else range(op2.num_wires)),
        )
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

    def test_prod_hamiltonian(self):
        """Test that a hamiltonian object can be composed."""
        U = qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])
        prod_op = Prod(qml.PauliZ(wires=0), U)
        mat = prod_op.matrix()

        true_mat = [
            [0.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5, 0.0],
        ]
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

    # sparse matrix tests:

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix of a Prod op is defined and correct."""
        prod_op = prod(op1(wires=0), op2(wires=1))
        true_mat = math.kron(mat1, mat2)
        prod_mat = prod_op.sparse_matrix().todense()

        assert np.allclose(true_mat, prod_mat)

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix_same_wires(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix of a Prod op is defined and correct."""
        prod_op = prod(op1(wires=0), op2(wires=0))
        true_mat = mat1 @ mat2
        prod_mat = prod_op.sparse_matrix().todense()

        assert np.allclose(true_mat, prod_mat)

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix_wire_order(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix of a Prod op is defined
        with wire order and correct."""
        true_mat = math.kron(math.kron(mat2, np.eye(2)), mat1)

        prod_op = prod(op1(wires=2), op2(wires=0))
        prod_mat = prod_op.sparse_matrix(wire_order=[0, 1, 2]).todense()

        assert np.allclose(true_mat, prod_mat)

    def test_sparse_matrix_undefined_error(self):
        """Test that an error is raised when the sparse matrix method
        is undefined for any of the factors."""

        class DummyOp(qml.operation.Operation):
            num_wires = 1

            def sparse_matrix(self, wire_order=None):
                raise qml.operation.SparseMatrixUndefinedError

        prod_op = prod(qml.PauliX(wires=0), DummyOp(wires=1))

        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            prod_op.sparse_matrix()


class TestProperties:
    """Test class properties."""

    @pytest.mark.parametrize(
        "ops_lst, hermitian_status",
        [(ops_tup, status) for ops_tup, status in zip(ops, ops_hermitian_status)],
    )
    def test_is_hermitian(self, ops_lst, hermitian_status):
        """Test is_hermitian property updates correctly."""
        prod_op = prod(*ops_lst)
        assert prod_op.is_hermitian == hermitian_status

    @pytest.mark.tf
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
            assert qml.is_hermitian(op) == hermitian_state

    @pytest.mark.jax
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
            assert qml.is_hermitian(op) == hermitian_state

    @pytest.mark.torch
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
            assert qml.is_hermitian(op) == hermitian_state

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
            [
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
        eigs_cache = diag_prod_op._eigs[diag_prod_op.hash]
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)

    def test_diagonalizing_gates(self):
        """Test that the diagonalizing gates are correct."""
        diag_prod_op = Prod(qml.PauliZ(wires=0), qml.PauliZ(wires=1))
        assert diag_prod_op.diagonalizing_gates() == []


class TestSimplify:
    """Test Prod simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        prod_op = (
            qml.RZ(1.32, wires=0) @ qml.Identity(wires=0) @ qml.RX(1.9, wires=1) @ qml.PauliX(0)
        )
        assert prod_op.arithmetic_depth == 3

        op_constructed = Prod(
            qml.RZ(1.32, wires=0), qml.Identity(wires=0), qml.RX(1.9, wires=1), qml.PauliX(0)
        )
        assert op_constructed.arithmetic_depth == 1

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        prod_op = qml.RZ(1.32, wires=0) @ qml.Identity(wires=0) @ qml.RX(1.9, wires=1)
        final_op = Prod(qml.RZ(1.32, wires=0), qml.RX(1.9, wires=1))
        simplified_op = prod_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, Prod)
        for s1, s2 in zip(final_op.operands, simplified_op.operands):
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

        assert qml.equal(final_op, simplified_op)

    def test_simplify_method_product_of_sums(self):
        """Test the simplify method with a product of sums."""
        prod_op = Prod(qml.PauliX(0) + qml.RX(1, 0), qml.PauliX(1) + qml.RX(1, 1), qml.Identity(3))
        final_op = qml.op_sum(
            Prod(qml.PauliX(0), qml.PauliX(1)),
            qml.PauliX(0) @ qml.RX(1, 1),
            qml.PauliX(1) @ qml.RX(1, 0),
            qml.RX(1, 0) @ qml.RX(1, 1),
        )
        simplified_op = prod_op.simplify()
        assert isinstance(simplified_op, qml.ops.Sum)
        for s1, s2 in zip(final_op.operands, simplified_op.operands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_with_nested_prod_and_adjoints(self):
        """Test simplify method with nested product and adjoint operators."""
        prod_op = Prod(qml.adjoint(Prod(qml.RX(1, 0), qml.RY(1, 0))), qml.RZ(1, 0))
        final_op = Prod(qml.RY(4 * np.pi - 1, 0), qml.RX(4 * np.pi - 1, 0), qml.RZ(1, 0))
        simplified_op = prod_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, Prod)
        for s1, s2 in zip(final_op.operands, simplified_op.operands):
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
                qml.PauliX(0) + 5 * (qml.RX(1, 1) + qml.PauliX(1)),
            ),
            qml.Identity(0),
        )
        mod_angle = -1 % (4 * np.pi)
        final_op = qml.op_sum(
            qml.Identity(0),
            5 * Prod(qml.PauliX(0), qml.RX(1, 1)),
            5 * Prod(qml.PauliX(0), qml.PauliX(1)),
            qml.RX(mod_angle, 0),
            5 * Prod(qml.RX(mod_angle, 0), qml.PauliX(0), qml.RX(1, 1)),
            5 * Prod(qml.RX(mod_angle, 0), qml.PauliX(0), qml.PauliX(1)),
            qml.PauliX(0),
            5 * qml.RX(1, 1),
            qml.s_prod(5, qml.PauliX(1)),
        )
        simplified_op = prod_op.simplify()
        assert isinstance(simplified_op, qml.ops.Sum)
        for s1, s2 in zip(final_op.operands, simplified_op.operands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_method_groups_rotations(self):
        """Test that the simplify method groups rotation operators."""
        prod_op = qml.prod(
            qml.RX(1, 0), qml.RZ(1, 1), qml.CNOT((1, 2)), qml.RZ(1, 1), qml.RX(3, 0), qml.RZ(1, 1)
        )
        final_op = qml.prod(qml.RZ(1, 1), qml.CNOT((1, 2)), qml.RX(4, 0), qml.RZ(2, 1))
        simplified_op = prod_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, Prod)
        for s1, s2 in zip(final_op.operands, simplified_op.operands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_method_with_pauli_words(self):
        """Test that the simplify method groups pauli words."""
        prod_op = qml.prod(
            qml.op_sum(qml.PauliX(0), qml.PauliX(1)), qml.PauliZ(1), qml.PauliX(0), qml.PauliY(1)
        )
        final_op = qml.op_sum(qml.s_prod(0 - 1j, qml.PauliX(1)), qml.s_prod(0 - 1j, qml.PauliX(0)))
        simplified_op = prod_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, qml.ops.Sum)
        for s1, s2 in zip(final_op.operands, simplified_op.operands):
            assert repr(s1) == repr(s2)
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_method_groups_identical_operators(self):
        """Test that the simplify method groups identical operators."""
        prod_op = qml.prod(
            qml.PauliX(0),
            qml.CNOT((1, 2)),
            qml.PauliZ(3),
            qml.Toffoli((4, 5, 6)),
            qml.CNOT((1, 2)),
            qml.Toffoli((4, 5, 6)),
            qml.PauliX(0),
            qml.PauliZ(3),
        )
        final_op = qml.Identity(range(7))
        simplified_op = prod_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, qml.Identity)
        assert final_op.name == simplified_op.name
        assert final_op.wires == simplified_op.wires
        assert final_op.data == simplified_op.data
        assert final_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_simplify_method_removes_grouped_elements_with_zero_coeff(self):
        """Test that the simplify method removes grouped elements with zero coeff."""
        prod_op = qml.prod(
            qml.U3(1.23, 2.34, 3.45, wires=0),
            qml.pow(z=-1, base=qml.U3(1.23, 2.34, 3.45, wires=0)),
        )
        final_op = qml.Identity(0)
        simplified_op = prod_op.simplify()

        assert qml.equal(final_op, simplified_op)

    def test_grouping_with_product_of_sum(self):
        """Test that grouping works with product of a sum"""
        prod_op = qml.prod(
            qml.PauliX(0), qml.op_sum(qml.PauliY(0), qml.Identity(0)), qml.PauliZ(0), qml.PauliX(0)
        )
        final_op = qml.op_sum(qml.s_prod(1j, qml.PauliX(0)), qml.s_prod(-1, qml.PauliZ(0)))
        simplified_op = prod_op.simplify()

        assert isinstance(simplified_op, qml.ops.Sum)
        for s1, s2 in zip(final_op.operands, simplified_op.operands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_grouping_with_product_of_sums(self):
        """Test that grouping works with product of two sums"""
        prod_op = qml.prod(qml.S(0) + qml.T(1), qml.S(0) + qml.T(1))
        final_op = qml.op_sum(
            qml.PauliZ(wires=[0]),
            2 * qml.prod(qml.S(wires=[0]), qml.T(wires=[1])),
            qml.S(wires=[1]),
        )
        simplified_op = prod_op.simplify()
        assert isinstance(simplified_op, qml.ops.Sum)
        for s1, s2 in zip(final_op.operands, simplified_op.operands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_grouping_with_barriers(self):
        """Test that grouping is not done when a barrier is present."""
        prod_op = qml.prod(qml.S(0), qml.Barrier(0), qml.S(0)).simplify()
        simplified_op = prod_op.simplify()
        assert isinstance(simplified_op, Prod)
        for s1, s2 in zip(prod_op.operands, simplified_op.operands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_grouping_with_only_visual_barriers(self):
        """Test that grouping is implemented when an only-visual barrier is present."""
        prod_op = qml.prod(qml.S(0), qml.Barrier(0, only_visual=True), qml.S(0)).simplify()
        assert qml.equal(prod_op.simplify(), qml.PauliZ(0))


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

        assert prod_class_op.operands == prod_func_op.operands
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
        """Test Prod class instance in probs measurement process raises error."""
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
        """Test Prod class instance in sample measurement process."""
        dev = qml.device("default.qubit", wires=2, shots=20)
        prod_op = Prod(qml.PauliX(wires=0), qml.PauliX(wires=1))

        @qml.qnode(dev)
        def my_circ():
            Prod(qml.Hadamard(0), qml.Hadamard(1))
            return qml.sample(op=prod_op)

        results = my_circ()

        assert len(results) == 20
        assert (results == 1).all()

    def test_measurement_process_counts(self):
        """Test Prod class instance in sample measurement process."""
        dev = qml.device("default.qubit", wires=2, shots=20)
        prod_op = Prod(qml.PauliX(wires=0), qml.PauliX(wires=1))

        @qml.qnode(dev)
        def my_circ():
            Prod(qml.Hadamard(0), qml.Hadamard(1))
            return qml.counts(op=prod_op)

        results = my_circ()

        assert sum(results.values()) == 20
        assert 1 in results
        assert -1 not in results

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
        """Test that non-hermitian ops in a measurement process will raise a warning."""
        wires = [0, 1]
        dev = qml.device("default.qubit", wires=wires)
        prod_op = Prod(qml.RX(1.23, wires=0), qml.Identity(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.expval(prod_op)

        with pytest.warns(UserWarning, match="Prod might not be hermitian."):
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

    def test_batched_operation(self):
        """Test that prod with batching gives expected results."""
        x = qml.numpy.array([1.0, 2.0, 3.0])
        y = qml.numpy.array(0.5)

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def batched_prod(x, y):
            qml.prod(qml.RX(x, wires=0), qml.RY(y, wires=0))
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def batched_no_prod(x, y):
            qml.RY(y, wires=0)
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        res1 = batched_prod(x, y)
        res2 = batched_no_prod(x, y)
        assert qml.math.allclose(res1, res2)


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
        sorted_list = Prod._sort(op_list)
        final_list = [
            qml.PauliY(0),
            qml.PauliY(1),
            qml.PauliZ(2),
            qml.PauliX(3),
            qml.PauliZ(3),
            qml.RX(1, 5),
            qml.PauliX(5),
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
        sorted_list = Prod._sort(op_tuple)
        final_list = [
            qml.PauliY(0),
            qml.PauliX(3),
            qml.Toffoli([2, 3, 4]),
            qml.PauliX(5),
            qml.CNOT([2, 5]),
            qml.CRX(1, [0, 2]),
            qml.CRY(1, [1, 2]),
            qml.PauliZ(3),
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
            qml.CRX(1, [0, 2]),
            qml.PauliZ("three"),
            qml.CRY(1, ["test", 2]),
        ]
        sorted_list = Prod._sort(op_list, wire_map={0: 0, "test": 1, 2: 2, "three": 3, 4: 4, 5: 5})
        final_list = [
            qml.PauliY(0),
            qml.PauliX("three"),
            qml.Toffoli([2, "three", 4]),
            qml.PauliX(5),
            qml.CNOT([2, 5]),
            qml.CRX(1, [0, 2]),
            qml.CRY(1, ["test", 2]),
            qml.PauliZ("three"),
            qml.RX(1, 5),
        ]

        for op1, op2 in zip(final_list, sorted_list):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert op1.data == op2.data


swappable_ops = {
    (qml.PauliX(1), qml.PauliY(0)),
    (qml.PauliY(5), qml.PauliX(2)),
    (qml.PauliZ(3), qml.PauliX(2)),
    (qml.CNOT((1, 2)), qml.PauliX(0)),
    (qml.PauliX(3), qml.Toffoli((0, 1, 2))),
}

non_swappable_ops = {
    (qml.PauliX(1), qml.PauliY(1)),
    (qml.PauliY(5), qml.RY(1, 5)),
    (qml.PauliZ(0), qml.PauliX(1)),
    (qml.CNOT((1, 2)), qml.PauliX(1)),
    (qml.PauliX(2), qml.Toffoli((0, 1, 2))),
}


class TestSwappableOps:
    """Tests for the _swappable_ops function."""

    @pytest.mark.parametrize(["op1", "op2"], swappable_ops)
    def test_swappable_ops(self, op1, op2):
        """Test the check for swappable operators."""
        assert _swappable_ops(op1, op2)
        assert not _swappable_ops(op2, op1)

    @pytest.mark.parametrize(["op1", "op2"], non_swappable_ops)
    def test_non_swappable_ops(self, op1, op2):
        """Test the check for non-swappable operators."""
        assert not _swappable_ops(op1, op2)
