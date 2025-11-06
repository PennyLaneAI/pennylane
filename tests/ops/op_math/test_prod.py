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

# pylint:disable=protected-access, unused-argument
import gate_data as gd  # a file containing matrix rep of each gate
import numpy as np
import pytest

import pennylane as qml
import pennylane.numpy as qnp
from pennylane import math
from pennylane.exceptions import DeviceError, MatrixUndefinedError
from pennylane.operation import Operator
from pennylane.ops.op_math.prod import Prod, _swappable_ops, prod
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


def test_basic_validity():
    """Run basic validity checks on a prod operator."""
    op1 = qml.PauliZ(0)
    op2 = qml.Rot(1.2, 2.3, 3.4, wires=0)
    op3 = qml.IsingZZ(4.32, wires=("a", "b"))
    op = qml.prod(op1, op2, op3)
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

    @pytest.mark.parametrize("id", ("foo", "bar"))
    def test_init_prod_op(self, id):
        """Test the initialization of a Prod operator."""
        prod_op = prod(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"), id=id)

        assert prod_op.wires == Wires((0, "a"))
        assert prod_op.num_wires == 2
        assert prod_op.name == "Prod"
        assert prod_op.id == id

        assert prod_op.data == (0.23,)
        assert prod_op.parameters == [0.23]
        assert prod_op.num_params == 1

    def test_hash(self):
        """Testing some situations for the hash property."""
        # same hash if different order but can be permuted to right order
        op1 = qml.prod(qml.PauliX(0), qml.PauliY("a"))
        op2 = qml.prod(qml.PauliY("a"), qml.PauliX(0))
        assert op1.hash == op2.hash

        # test not the same hash if different order and cant be exchanged to correct order
        op3 = qml.prod(qml.PauliX("a"), qml.PauliY("a"), qml.PauliX(1))
        op4 = qml.prod(qml.PauliY("a"), qml.PauliX("a"), qml.PauliX(1))
        assert op3.hash != op4.hash

    PROD_TERMS_OP_PAIRS_MIXED = (  # not all operands have pauli representation
        (
            qml.prod(qml.Hadamard(0), X(1), X(2)),
            [1.0],
            [qml.prod(qml.Hadamard(0), X(1), X(2))],
        ),  # trivial product
        (
            qml.prod(qml.Hadamard(0), X(1), qml.Identity(2)),
            [1.0],
            [qml.prod(qml.Hadamard(0), X(1))],
        ),
        (
            qml.prod(qml.Hadamard(0), qml.s_prod(4, X(1)), qml.s_prod(2, X(2))),
            [2 * 4],
            [qml.prod(qml.Hadamard(0), X(1), X(2))],
        ),  # product with scalar products inside
        (
            qml.prod(qml.Hadamard(0), qml.s_prod(4, X(0)), qml.s_prod(2, X(1))),
            [2 * 4],
            [qml.prod(qml.Hadamard(0), X(0), X(1))],
        ),  # product with scalar products on same wire
        (
            qml.prod(qml.Hadamard(0), qml.s_prod(4, Y(1)), qml.sum(X(2), X(3))),
            [4, 4],
            [qml.prod(qml.Hadamard(0), Y(1), X(2)), qml.prod(qml.Hadamard(0), Y(1), X(3))],
        ),  # product with sums inside
        (
            qml.prod(
                qml.prod(qml.Hadamard(0), X(2), X(3)),
                qml.s_prod(0.5, qml.sum(qml.Hadamard(5), qml.s_prod(0.4, X(6)))),
            ),
            [0.5, 0.2],
            [
                qml.prod(X(2), X(3), qml.Hadamard(5), qml.Hadamard(0)),
                qml.prod(X(6), X(2), X(3), qml.Hadamard(0)),
            ],
        ),  # contrived example
    )

    @pytest.mark.parametrize("op, coeffs_true, ops_true", PROD_TERMS_OP_PAIRS_MIXED)
    def test_terms_no_pauli_rep(self, op, coeffs_true, ops_true):
        """Test that Prod.terms() is correct for operators that dont all have a pauli_rep"""
        coeffs, ops1 = op.terms()
        assert coeffs == coeffs_true
        assert ops1 == ops_true

    PROD_TERMS_OP_PAIRS_PAULI = (  # all operands have pauli representation
        (qml.prod(X(0), X(1), X(2)), [1.0], [qml.prod(X(0), X(1), X(2))]),  # trivial product
        (
            qml.prod(X(0), X(1), X(2), qml.Identity(0)),
            [1.0],
            [qml.prod(X(0), X(1), X(2))],
        ),  # trivial product
        (
            qml.prod(X(0), qml.s_prod(4, X(1)), qml.s_prod(2, X(2))),
            [2 * 4],
            [qml.prod(X(0), X(1), X(2))],
        ),  # product with scalar products inside
        (
            qml.prod(X(0), qml.s_prod(4, X(0)), qml.s_prod(2, X(1))),
            [2 * 4],
            [X(1)],
        ),  # product with scalar products on same wire
        (
            qml.prod(X(0), qml.s_prod(4, Y(0)), qml.s_prod(2, X(1))),
            [1j * 2 * 4],
            [qml.prod(Z(0), X(1))],
        ),  # product with scalar products on same wire
        (
            qml.prod(X(0), qml.s_prod(4, Y(1)), qml.sum(X(2), X(3))),
            [4, 4],
            [qml.prod(X(0), Y(1), X(2)), qml.prod(X(0), Y(1), X(3))],
        ),  # product with sums inside
    )

    @pytest.mark.parametrize("op, coeffs_true, ops_true", PROD_TERMS_OP_PAIRS_PAULI)
    def test_terms_pauli_rep(self, op, coeffs_true, ops_true):
        """Test that Prod.terms() is correct for operators that all have a pauli_rep"""
        coeffs, ops1 = op.terms()
        assert coeffs == coeffs_true
        assert ops1 == ops_true

    def test_terms_pauli_rep_wire_order(self):
        """Test that the wire order of the terms is the same as the wire order of the original
        operands when the Prod has a valid pauli_rep"""
        H = qml.prod(X(0), X(1), X(2))
        _, H_ops = H.terms()

        assert len(H_ops) == 1
        assert H_ops[0].wires == H.wires

    def test_batch_size(self):
        """Test that batch size returns the batch size of a base operation if it is batched."""
        x = qml.numpy.array([1.0, 2.0, 3.0])
        prod_op = prod(qml.PauliX(0), qml.RX(x, wires=0))
        assert prod_op.batch_size == 3

    @pytest.mark.parametrize(
        "op, coeffs_true, ops_true", PROD_TERMS_OP_PAIRS_PAULI + PROD_TERMS_OP_PAIRS_MIXED
    )
    def test_terms_does_not_change_queue(self, op, coeffs_true, ops_true):
        """Test that calling Prod.terms does not queue anything."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.apply(op)
            _, _ = op.terms()

        assert q.queue == [op]

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
            qml.assert_equal(op1, op2)

    @pytest.mark.parametrize("ops_lst", ops)
    def test_decomposition_on_tape(self, ops_lst):
        """Test the decomposition of a prod of operators is a list
        of the provided factors on a tape."""
        prod_op = prod(*ops_lst)
        true_decomposition = list(ops_lst[::-1])  # reversed list of factors
        with qml.queuing.AnnotatedQueue() as q:
            prod_op.decomposition()

        tape = qml.tape.QuantumScript.from_queue(q)
        for op1, op2 in zip(tape.operations, true_decomposition):
            qml.assert_equal(op1, op2)

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

        prod_op = prod(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"))
        assert prod_op.has_matrix is True

    def test_has_matrix_true_via_factor_has_no_matrix_but_is_hamiltonian(self):
        """Test that a product of operators of which one does not have `has_matrix=True`
        but is a Hamiltonian has `has_matrix=True`."""

        H = qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])
        prod_op = prod(H, qml.RZ(0.23, wires=5))
        assert prod_op.has_matrix is True

    @pytest.mark.parametrize(
        "first_factor", [qml.PauliX(wires=0), qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])]
    )
    def test_has_matrix_false_via_factor_has_no_matrix(self, first_factor):
        """Test that a product of operators of which one does not have `has_matrix=True`
        has `has_matrix=False`."""

        prod_op = prod(first_factor, MyOp(0.23, wires="a"))
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

        prod_op = prod(*factors)
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

        prod_op = prod(*factors)
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

        prod_op = prod(*factors)
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

        prod_op = prod(*factors)
        assert prod_op.has_diagonalizing_gates is True

    def test_has_diagonalizing_gates_false_via_factor(self):
        """Test that a product of operators of which one has
        `has_diagonalizing_gates=False` has `has_diagonalizing_gates=False` as well."""

        prod_op = prod(MyOp(3.1, 0), qml.PauliX(2))
        assert prod_op.has_diagonalizing_gates is False

    def test_qfunc_init(self):
        """Tests prod initialization with a qfunc argument."""

        def qfunc():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.RZ(1.1, 1)

        prod_gen = prod(qfunc)
        assert callable(prod_gen)
        prod_op = prod_gen()
        expected = prod(qml.RZ(1.1, 1), qml.CNOT([0, 1]), qml.Hadamard(0))
        qml.assert_equal(prod_op, expected)
        assert prod_op.wires == Wires([1, 0])

    def test_qfunc_single_operator(self):
        """Test prod initialization with qfunc that queues a single operator."""

        def qfunc():
            qml.S(0)

        with qml.queuing.AnnotatedQueue() as q:
            out = prod(qfunc)()

        assert len(q) == 1
        assert q.queue[0] == qml.S(0)
        assert out == qml.S(0)

    def test_qfunc_init_accepts_args_kwargs(self):
        """Tests that prod preserves args when wrapping qfuncs."""

        def qfunc(x, run_had=False):
            if run_had:
                qml.Hadamard(0)
            qml.RX(x, 0)
            qml.CNOT([0, 1])

        prod_gen = prod(qfunc)
        qml.assert_equal(prod_gen(1.1), prod(qml.CNOT([0, 1]), qml.RX(1.1, 0)))
        qml.assert_equal(
            prod_gen(2.2, run_had=True), prod(qml.CNOT([0, 1]), qml.RX(2.2, 0), qml.Hadamard(0))
        )

    def test_qfunc_init_propagates_Prod_kwargs(self):
        """Tests that additional kwargs for Prod are propagated using qfunc initialization."""

        def qfunc(x):
            qml.prod(qml.RX(x, 0), qml.PauliZ(1))
            qml.CNOT([0, 1])

        prod_gen = prod(qfunc, id=123987, lazy=False)
        prod_op = prod_gen(1.1)

        assert prod_op.id == 123987  # id was set
        qml.assert_equal(prod_op, prod(qml.CNOT([0, 1]), qml.PauliZ(1), qml.RX(1.1, 0)))  # eager

    def test_qfunc_init_only_works_with_one_qfunc(self):
        """Test that the qfunc init only occurs when one callable is passed to prod."""

        def qfunc():
            qml.Hadamard(0)
            qml.CNOT([0, 1])

        prod_op = prod(qfunc)()
        qml.assert_equal(prod_op, prod(qml.CNOT([0, 1]), qml.Hadamard(0)))

        def fn2():
            qml.PauliX(0)
            qml.PauliY(1)

        for args in [(qfunc, fn2), (qfunc, qml.PauliX), (qml.PauliX, qfunc)]:
            with pytest.raises(AttributeError, match="has no attribute 'wires'"):
                prod(*args)

    def test_qfunc_init_returns_single_op(self):
        """Tests that if a qfunc only queues one operator, that operator is returned."""

        def qfunc():
            qml.PauliX(0)

        prod_op = prod(qfunc)()
        qml.assert_equal(prod_op, qml.PauliX(0))
        assert not isinstance(prod_op, Prod)

    @pytest.mark.xfail  # this requirement has been lifted
    def test_prod_accepts_single_operator_but_Prod_does_not(self):
        """Tests that the prod wrapper can accept a single operator, and return it."""

        x = qml.PauliX(0)
        prod_op = prod(x)
        assert prod_op is x
        assert not isinstance(prod_op, Prod)

        with pytest.raises(ValueError, match="Require at least two operators"):
            Prod(x)

    def test_prod_fails_with_non_callable_arg(self):
        """Tests that prod explicitly checks that a single-arg is either an Operator or callable."""
        with pytest.raises(TypeError, match="Unexpected argument of type int passed to qml.prod"):
            prod(1)


def test_empty_repr():
    """Test that an empty prod still has a repr that indicates it's a prod."""
    assert repr(Prod()) == "Prod()"


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
        """Test matrix method for a product of non_parametric ops"""
        mat1, mat2 = compare_and_expand_mat(mat1, mat2)
        true_mat = mat1 @ mat2

        prod_op = Prod(
            op1(wires=0 if op1.num_wires is None else range(op1.num_wires)),
            op2(wires=0 if op2.num_wires is None else range(op2.num_wires)),
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

    def test_prod_templates(self):
        """Test that we can compose templates and the generated matrix is correct."""

        def get_qft_mat(num_wires):
            omega = math.exp(np.pi * 1.0j / 2 ** (num_wires - 1))
            mat = math.zeros((2**num_wires, 2**num_wires), dtype="complex128")

            for m in range(2**num_wires):
                for n in range(2**num_wires):
                    mat[m, n] = omega ** (m * n)

            return 1 / math.sqrt(2**num_wires) * mat

        wires = [0, 1, 2]
        prod_op = Prod(qml.QFT(wires=wires), qml.GroverOperator(wires=wires), qml.PauliX(wires=0))
        mat = prod_op.matrix()

        grov_mat = (1 / 4) * math.ones((8, 8), dtype="complex128") - math.eye(8, dtype="complex128")
        qft_mat = get_qft_mat(3)
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
            qml.Projector(state=qnp.array([0, 1]), wires=wires),
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

    def test_matrix_all_batched(self):
        """Test that Prod matrix has batching support when all operands are batched."""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        y = qml.numpy.array([0.4, 0.5, 0.6])
        op = prod(qml.RX(x, wires=0), qml.RY(y, wires=2), qml.PauliZ(1))
        mat = op.matrix()
        sum_list = [
            prod(qml.RX(i, wires=0), qml.RY(j, wires=2), qml.PauliZ(1)) for i, j in zip(x, y)
        ]
        compare = qml.math.stack([s.matrix() for s in sum_list])
        assert qml.math.allclose(mat, compare)
        assert mat.shape == (3, 8, 8)

    def test_matrix_not_all_batched(self):
        """Test that Prod matrix has batching support when all operands are not batched."""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        y = 0.5
        z = qml.numpy.array([0.4, 0.5, 0.6])
        op = prod(
            qml.RX(x, wires=0),
            qml.RY(y, wires=2),
            qml.RZ(z, wires=1),
            qml.prod(qml.PauliX(2), qml.PauliY(3)),
        )
        mat = op.matrix()
        batched_y = [y for _ in x]
        sum_list = [
            prod(
                qml.RX(i, wires=0),
                qml.RY(j, wires=2),
                qml.RZ(k, wires=1),
                qml.prod(qml.PauliX(2), qml.PauliY(3)),
            )
            for i, j, k in zip(x, batched_y, z)
        ]
        compare = qml.math.stack([s.matrix() for s in sum_list])
        assert qml.math.allclose(mat, compare)
        assert mat.shape == (3, 16, 16)

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
    def test_sparse_matrix_format(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix accepts the format parameter."""
        from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix

        prod_op = qml.sum(op1(wires=0), op2(wires=1))
        true_mat = math.kron(mat1, np.eye(2)) + math.kron(np.eye(2), mat2)
        assert isinstance(prod_op.sparse_matrix(), csr_matrix)
        prod_op_csc = prod_op.sparse_matrix(format="csc")
        prod_op_lil = prod_op.sparse_matrix(format="lil")
        prod_op_coo = prod_op.sparse_matrix(format="coo")
        assert isinstance(prod_op_csc, csc_matrix)
        assert isinstance(prod_op_lil, lil_matrix)
        assert isinstance(prod_op_coo, coo_matrix)
        assert np.allclose(true_mat, prod_op_csc.todense())
        assert np.allclose(true_mat, prod_op_lil.todense())
        assert np.allclose(true_mat, prod_op_coo.todense())

    def test_sparse_matrix_global_phase(self):
        """Test that a prod with a global phase still defines a sparse matrix."""

        op = qml.GlobalPhase(0.5) @ qml.X(0) @ qml.X(0)

        sparse_mat = op.sparse_matrix(wire_order=(0, 1))
        mat = sparse_mat.todense()
        assert qml.math.allclose(mat, np.exp(-0.5j) * np.eye(4))

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

        class DummyOp(qml.operation.Operation):  # pylint:disable=too-few-public-methods
            num_wires = 1

            def sparse_matrix(self, wire_order=None):
                raise qml.operation.SparseMatrixUndefinedError

        prod_op = prod(qml.PauliX(wires=0), DummyOp(wires=1))

        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            prod_op.sparse_matrix()


class TestProperties:
    """Test class properties."""

    @pytest.mark.parametrize("ops_lst, hermitian_status", list(zip(ops, ops_hermitian_status)))
    def test_is_hermitian(self, ops_lst, hermitian_status):
        """Test is_hermitian property updates correctly."""
        prod_op = prod(*ops_lst)
        assert prod_op.is_hermitian == hermitian_status

    @pytest.mark.tf
    def test_is_hermitian_tf(self):
        """Test that is_hermitian works when a tf type scalar is provided."""
        # pylint:disable=invalid-unary-operand-type
        import tensorflow as tf

        theta = tf.Variable(1.23)
        prod_ops = (
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=0), qml.PauliZ(wires=1)),
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=1), qml.PauliZ(wires=2)),
        )
        true_hermitian_states = (True, False)

        for op, hermitian_state in zip(prod_ops, true_hermitian_states):
            assert qml.is_hermitian(op) == hermitian_state

    @pytest.mark.jax
    def test_is_hermitian_jax(self):
        """Test that is_hermitian works when a jax type scalar is provided."""
        import jax.numpy as jnp

        theta = jnp.array(1.23)
        prod_ops = (
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=0), qml.PauliZ(wires=1)),
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=1), qml.PauliZ(wires=2)),
        )
        true_hermitian_states = (True, False)

        for op, hermitian_state in zip(prod_ops, true_hermitian_states):
            assert qml.is_hermitian(op) == hermitian_state

    @pytest.mark.torch
    def test_is_hermitian_torch(self):
        """Test that is_hermitian works when a torch type scalar is provided."""
        import torch

        theta = torch.tensor(1.23)
        prod_ops = (
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=0), qml.PauliZ(wires=1)),
            prod(qml.RX(theta, wires=0), qml.RX(-theta, wires=1), qml.PauliZ(wires=2)),
        )
        true_hermitian_states = (True, False)

        for op, hermitian_state in zip(prod_ops, true_hermitian_states):
            assert qml.is_hermitian(op) == hermitian_state

    @pytest.mark.parametrize("ops_lst", ops)
    def test_queue_category_ops(self, ops_lst):
        """Test _queue_category property is '_ops' when all factors are `_ops`."""
        prod_op = prod(*ops_lst)
        assert prod_op._queue_category == "_ops"

    def test_queue_category_none(self):
        """Test _queue_category property is None when any factor is not `_ops`."""

        class DummyOp(Operator):  # pylint:disable=too-few-public-methods
            """Dummy op with None queue category"""

            _queue_category = None
            num_wires = 1

        prod_op = prod(qml.Identity(wires=0), DummyOp(wires=0))
        assert prod_op._queue_category is None

    def test_eigendecomposition(self):
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

    def test_qutrit_eigvals(self):
        """Test that the eigvals can be computed with qutrit observables."""

        op1 = qml.GellMann(wires=0)
        op2 = qml.GellMann(index=8, wires=1)

        prod_op = qml.prod(op1, op2)
        eigs = prod_op.eigvals()

        mat_eigs = np.linalg.eigvals(prod_op.matrix())

        sorted_eigs = np.sort(eigs)
        sorted_mat_eigs = np.sort(mat_eigs)
        assert qml.math.allclose(sorted_eigs, sorted_mat_eigs)

        # pylint: disable=import-outside-top-level
        from pennylane.ops.functions.assert_valid import _check_eigendecomposition

        _check_eigendecomposition(prod_op)

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

    @pytest.mark.jax
    def test_eigvals_jax_jit(self):
        """Assert computing the eigvals of a Prod is compatible with jax-jit."""
        import jax

        def f(t1, t2):
            return qml.prod(qml.RX(t1, 0), qml.RX(t2, 0)).eigvals()

        assert qml.math.allclose(f(0.5, 1.0), jax.jit(f)(0.5, 1.0))

    def test_eigvals_no_wires_identity(self):
        """Test that eigvals can be computed if a component is an identity on no wires."""
        op = qml.X(0) @ qml.Y(1) @ qml.I()
        op2 = qml.X(0) @ qml.Y(1)

        assert qml.math.allclose(op.eigvals(), op2.eigvals())

    # pylint: disable=use-implicit-booleaness-not-comparison
    def test_diagonalizing_gates(self):
        """Test that the diagonalizing gates are correct."""
        diag_prod_op = Prod(qml.PauliZ(wires=0), qml.PauliZ(wires=1))
        assert diag_prod_op.diagonalizing_gates() == []

    op_pauli_reps = (
        (
            qml.prod(qml.PauliX(wires=0), qml.PauliY(wires=1), qml.PauliZ(wires="a")),
            qml.pauli.PauliSentence({qml.pauli.PauliWord({0: "X", 1: "Y", "a": "Z"}): 1}),
        ),
        (
            qml.prod(qml.PauliX(wires=0), qml.PauliX(wires=0)),
            qml.pauli.PauliSentence({qml.pauli.PauliWord({}): 1}),
        ),
        (
            qml.prod(
                qml.PauliX(wires=0),
                qml.PauliY(wires=0),
                qml.PauliZ(wires="a"),
                qml.PauliZ(wires="a"),
            ),
            qml.pauli.PauliSentence({qml.pauli.PauliWord({0: "Z"}): 1j}),
        ),
    )

    def test_pauli_rep_order(self):
        """Lightning qubit tests are relying on the specific order of the pauli rep keys.
        If the order of pauli word keys is changed, lightning tests must be changed as well.
        """
        op = qml.prod(qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2))
        pw = list(op.pauli_rep.keys())[0]
        assert list(pw.keys()) == [0, 1, 2]
        assert list(pw.values()) == ["X", "Y", "Z"]

    @pytest.mark.parametrize("op, rep", op_pauli_reps)
    def test_pauli_rep(self, op, rep):
        """Test that the pauli rep gives the expected result."""
        assert op.pauli_rep == rep

    def test_pauli_rep_none(self):
        """Test that None is produced if any of the terms don't have a _pauli_rep property."""
        op = qml.prod(qml.PauliX(wires=0), qml.RX(1.23, wires=1))
        assert op.pauli_rep is None

    op_pauli_reps_nested = (
        (
            qml.prod(
                qml.prod(qml.PauliX(wires=0), qml.PauliY(wires=1)),
                qml.prod(qml.PauliY(wires=0), qml.PauliZ(wires=2)),
            ),
            qml.pauli.PauliSentence({qml.pauli.PauliWord({0: "Z", 1: "Y", 2: "Z"}): 1j}),
        ),
        (
            qml.prod(
                qml.s_prod(
                    -2j,
                    qml.prod(
                        qml.s_prod(0.5, qml.PauliX(wires=0)), qml.s_prod(2, qml.PauliZ(wires=1))
                    ),
                ),
                qml.s_prod(3, qml.PauliY(wires=0)),
            ),
            qml.pauli.PauliSentence({qml.pauli.PauliWord({0: "Z", 1: "Z"}): 6}),
        ),  # prod + s_prod
    )

    @pytest.mark.parametrize("op, rep", op_pauli_reps_nested)
    def test_pauli_rep_nested(self, op, rep):
        """Test that the pauli rep gives the expected result."""
        assert op.pauli_rep == rep


class TestSimplify:
    """Test Prod simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        o1 = qml.prod(qml.RZ(1.32, wires=0), qml.I(wires=0))
        o2 = qml.prod(o1, qml.RX(1.9, wires=1))
        prod_op = qml.prod(o2, qml.X(0))
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
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_method_with_identity(self):
        """Test that the simplify method of a product of an operator with an identity returns
        the operator."""
        prod_op = Prod(qml.PauliX(0), qml.Identity(0))
        final_op = qml.PauliX(0)
        simplified_op = prod_op.simplify()
        qml.assert_equal(final_op, simplified_op)

    def test_simplify_method_product_of_sums(self):
        """Test the simplify method with a product of sums."""
        prod_op = Prod(qml.PauliX(0) + qml.RX(1, 0), qml.PauliX(1) + qml.RX(1, 1), qml.Identity(3))
        final_op = qml.sum(
            Prod(qml.PauliX(0), qml.PauliX(1)),
            qml.PauliX(0) @ qml.RX(1, 1),
            qml.PauliX(1) @ qml.RX(1, 0),
            qml.RX(1, 0) @ qml.RX(1, 1),
        )
        simplified_op = prod_op.simplify()
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_with_nested_prod_and_adjoints(self):
        """Test simplify method with nested product and adjoint operators."""
        prod_op = Prod(qml.adjoint(Prod(qml.RX(1, 0), qml.RY(1, 0))), qml.RZ(1, 0))
        final_op = Prod(qml.RY(4 * np.pi - 1, 0), qml.RX(4 * np.pi - 1, 0), qml.RZ(1, 0))
        simplified_op = prod_op.simplify()
        qml.assert_equal(simplified_op, final_op)

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
        final_op = qml.sum(
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
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_method_groups_rotations(self):
        """Test that the simplify method groups rotation operators."""
        prod_op = qml.prod(
            qml.RX(1, 0), qml.RZ(1, 1), qml.CNOT((1, 2)), qml.RZ(1, 1), qml.RX(3, 0), qml.RZ(1, 1)
        )
        final_op = qml.prod(qml.RZ(1, 1), qml.CNOT((1, 2)), qml.RX(4, 0), qml.RZ(2, 1))
        simplified_op = prod_op.simplify()
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_method_with_pauli_words(self):
        """Test that the simplify method groups pauli words."""
        prod_op = qml.prod(
            qml.sum(qml.PauliX(0), qml.PauliX(1)), qml.PauliZ(1), qml.PauliX(0), qml.PauliY(1)
        )
        final_op = qml.sum(qml.s_prod(0 - 1j, qml.PauliX(1)), qml.s_prod(0 - 1j, qml.PauliX(0)))
        simplified_op = prod_op.simplify()
        qml.assert_equal(simplified_op, final_op)

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
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_method_removes_grouped_elements_with_zero_coeff(self):
        """Test that the simplify method removes grouped elements with zero coeff."""
        prod_op = qml.prod(
            qml.RX(1.23, wires=0),
            qml.RX(-1.23, wires=0),
        )
        final_op = qml.Identity(0)
        simplified_op = prod_op.simplify()
        qml.assert_equal(final_op, simplified_op)

    def test_grouping_with_product_of_sum(self):
        """Test that grouping works with product of a sum"""
        prod_op = qml.prod(
            qml.PauliX(0), qml.sum(qml.PauliY(0), qml.Identity(0)), qml.PauliZ(0), qml.PauliX(0)
        )
        final_op = qml.sum(qml.s_prod(1j, qml.PauliX(0)), qml.s_prod(-1, qml.PauliZ(0)))
        simplified_op = prod_op.simplify()
        qml.assert_equal(simplified_op, final_op)

    def test_grouping_with_equal_paulis_single_wire(self):
        """Test that equal Pauli operators, creating global phase contributions, are simplified
        correctly on one wire."""
        prod_op = qml.prod(qml.X(0) @ qml.Y(0) @ qml.Z(0) @ qml.H(0))
        final_op = 1j * qml.H(0)
        simplified_op = prod_op.simplify()
        assert np.allclose(qml.matrix(prod_op), qml.matrix(final_op))
        qml.assert_equal(simplified_op, final_op)

    def test_grouping_with_equal_paulis_two_wires(self):
        """Test that equal Pauli operators, creating global phase contributions, are simplified
        correctly on two wires."""
        prod_op = qml.prod(
            qml.X(0)
            @ qml.Z("a")
            @ qml.Y(0)
            @ qml.Z(0)
            @ qml.X("a")
            @ qml.Y("a")
            @ qml.H(0)
            @ qml.H("a")
        )
        final_op = qml.simplify(-1 * qml.H(0) @ qml.H("a"))
        simplified_op = prod_op.simplify()
        assert np.allclose(qml.matrix(prod_op), qml.matrix(final_op))
        qml.assert_equal(simplified_op, final_op)

    def test_grouping_with_product_of_sums(self):
        """Test that grouping works with product of two sums"""
        prod_op = qml.prod(qml.S(0) + qml.H(1), qml.S(0) + qml.H(1))
        final_op = qml.sum(
            qml.PauliZ(wires=[0]),
            2 * qml.prod(qml.S(wires=[0]), qml.H(wires=[1])),
            qml.I(wires=[1]),
        )
        simplified_op = prod_op.simplify()
        qml.assert_equal(simplified_op, final_op)

    def test_grouping_with_barriers(self):
        """Test that grouping is not done when a barrier is present."""
        prod_op = qml.prod(qml.S(0), qml.Barrier(0), qml.S(0)).simplify()
        simplified_op = prod_op.simplify()
        qml.assert_equal(simplified_op, prod_op)

    def test_grouping_with_only_visual_barriers(self):
        """Test that grouping is implemented when an only-visual barrier is present."""
        prod_op = qml.prod(qml.S(0), qml.Barrier(0, only_visual=True), qml.S(0)).simplify()
        qml.assert_equal(prod_op.simplify(), qml.PauliZ(0))

    @pytest.mark.jax
    def test_simplify_pauli_rep_jax(self):
        """Test that simplifying operators with a valid pauli representation works with jax interface."""
        import jax.numpy as jnp

        c1, c2, c3 = jnp.array(1.23), jnp.array(2.0), jnp.array(2.46j)

        op = prod(qml.s_prod(c1, qml.PauliX(0)), qml.s_prod(c2, prod(qml.PauliY(0), qml.PauliZ(1))))
        result = qml.s_prod(c3, prod(qml.PauliZ(0), qml.PauliZ(1)))
        simplified_op = op.simplify()

        qml.assert_equal(simplified_op, result)

    @pytest.mark.tf
    def test_simplify_pauli_rep_tf(self):
        """Test that simplifying operators with a valid pauli representation works with tf interface."""
        import tensorflow as tf

        c1, c2, c3 = (
            tf.Variable(1.23, dtype=tf.complex128),
            tf.Variable(2.0, dtype=tf.complex128),
            tf.Variable(2.46j, dtype=tf.complex128),
        )

        op = prod(qml.s_prod(c1, qml.PauliX(0)), qml.s_prod(c2, prod(qml.PauliY(0), qml.PauliZ(1))))
        result = qml.s_prod(c3, prod(qml.PauliZ(0), qml.PauliZ(1)))
        simplified_op = op.simplify()
        qml.assert_equal(simplified_op, result)

    @pytest.mark.torch
    def test_simplify_pauli_rep_torch(self):
        """Test that simplifying operators with a valid pauli representation works with torch interface."""
        import torch

        c1, c2, c3 = torch.tensor(1.23), torch.tensor(2.0), torch.tensor(2.46j)

        op = prod(qml.s_prod(c1, qml.PauliX(0)), qml.s_prod(c2, prod(qml.PauliY(0), qml.PauliZ(1))))
        result = qml.s_prod(c3, prod(qml.PauliZ(0), qml.PauliZ(1)))
        simplified_op = op.simplify()

        qml.assert_equal(simplified_op, result)


class TestWrapperFunc:
    """Test wrapper function."""

    def test_op_prod_top_level(self):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        factors = (qml.PauliX(wires=1), qml.RX(1.23, wires=0), qml.CNOT(wires=[0, 1]))
        op_id = "prod_op"

        prod_func_op = prod(*factors, id=op_id)
        prod_class_op = Prod(*factors, id=op_id)
        qml.assert_equal(prod_func_op, prod_class_op)

    def test_lazy_mode(self):
        """Test that by default, the operator is simply wrapped in `Prod`, even if a simplification exists."""
        op = prod(qml.S(0), Prod(qml.S(1), qml.T(1)))

        assert isinstance(op, Prod)
        assert len(op) == 2

    def test_non_lazy_mode(self):
        """Test the lazy=False keyword."""
        op = prod(qml.S(0), Prod(qml.S(1), qml.T(1)), lazy=False)

        assert isinstance(op, Prod)
        assert len(op) == 3

    def test_nonlazy_mode_queueing(self):
        """Test that if a simpification is accomplished, the metadata for the original op
        and the new simplified op is updated."""
        with qml.queuing.AnnotatedQueue() as q:
            prod1 = prod(qml.S(1), qml.T(1))
            prod2 = prod(qml.S(0), prod1, lazy=False)

        assert len(q) == 1
        assert q.queue[0] is prod2

    def test_correct_queued_operators(self):
        """Test that args and kwargs do not add operators to the queue."""

        def f(op):
            qml.apply(op)

        def f2(op, op2=None):
            qml.X(0)
            qml.apply(op)
            if op2:
                qml.apply(op2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.prod(f)(qml.Z(0))
            qml.prod(f2)(qml.Y(0), op2=qml.Z(0))

        assert len(q.queue) == 2
        assert q.queue[0] == qml.Z(0)
        assert q.queue[1] == qml.Z(0) @ qml.Y(0) @ qml.X(0)


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

        x_probs = np.array([0.5, 0.5])
        h_probs = np.array([np.cos(-np.pi / 4 / 2) ** 2, np.sin(-np.pi / 4 / 2) ** 2])
        expected = np.tensordot(x_probs, h_probs, axes=0).flatten()
        out = my_circ()
        assert qml.math.allclose(out, expected)

    def test_measurement_process_sample(self):
        """Test Prod class instance in sample measurement process."""
        dev = qml.device("default.qubit", wires=2)
        prod_op = Prod(qml.PauliX(wires=0), qml.PauliX(wires=1))

        @qml.set_shots(20)
        @qml.qnode(dev)
        def my_circ():
            Prod(qml.Hadamard(0), qml.Hadamard(1))
            return qml.sample(op=prod_op)

        results = my_circ()

        assert len(results) == 20
        assert (results == 1).all()

    def test_measurement_process_counts(self):
        """Test Prod class instance in sample measurement process."""
        dev = qml.device("default.qubit", wires=2)
        prod_op = Prod(qml.PauliX(wires=0), qml.PauliX(wires=1))

        @qml.set_shots(20)
        @qml.qnode(dev)
        def my_circ():
            Prod(qml.Hadamard(0), qml.Hadamard(1))
            return qml.counts(op=prod_op)

        results = my_circ()

        assert sum(results.values()) == 20
        assert 1 in results  # pylint:disable=unsupported-membership-test
        assert -1 not in results  # pylint:disable=unsupported-membership-test

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

    def test_non_supported_obs_not_supported(self):
        """Test that non-supported ops in a measurement process will raise an error."""
        wires = [0, 1]
        dev = qml.device("default.qubit", wires=wires)
        prod_op = Prod(qml.RX(1.23, wires=0), qml.Identity(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.expval(prod_op)

        with pytest.raises(DeviceError):
            my_circ()

    def test_operation_integration(self):
        """Test that a Product operation can be queued and executed in a circuit"""
        dev = qml.device("default.qubit", wires=3)
        operands = (
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.RX(1.23, wires=1),
            qml.IsingZZ(0.56, wires=[0, 2]),
        )

        @qml.qnode(dev)
        def prod_state_circ():
            Prod(*operands)
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
        y = qml.numpy.array([4.0, 5.0, 6.0])

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

    def test_params_can_be_considered_trainable(self):
        """Tests that the parameters of a Prod are considered trainable."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, U):
            qml.RX(x, 0)
            return qml.expval(qml.prod(qml.Hermitian(U, 0), qml.PauliX(1)))

        x = qnp.array(0.1, requires_grad=False)
        U = qnp.array([[1.0, 0.0], [0.0, -1.0]], requires_grad=True)

        tape = qml.workflow.construct_tape(circuit)(x, U)
        assert tape.trainable_params == [1]


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
            qml.assert_equal(op1, op2)

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
            qml.assert_equal(op1, op2)

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

    def test_sorting_operators_with_no_wires(self):
        """Test that sorting can occur when an operator acts on no wires."""

        op_list = (qml.GlobalPhase(0.5), qml.X(0), qml.I(), qml.CNOT((1, 2)), qml.I())

        sorted_list = qml.ops.Prod._sort(op_list)
        expected = [qml.X(0), qml.CNOT((1, 2)), qml.I(), qml.I(), qml.GlobalPhase(0.5)]
        assert sorted_list == expected


swappable_ops = [
    (qml.PauliX(1), qml.PauliY(0)),
    (qml.PauliY(5), qml.PauliX(2)),
    (qml.PauliZ(3), qml.PauliX(2)),
    (qml.CNOT((1, 2)), qml.PauliX(0)),
    (qml.PauliX(3), qml.Toffoli((0, 1, 2))),
]

non_swappable_ops = [
    (qml.PauliX(1), qml.PauliY(1)),
    (qml.PauliY(5), qml.RY(1, 5)),
    (qml.PauliZ(0), qml.PauliX(1)),
    (qml.CNOT((1, 2)), qml.PauliX(1)),
    (qml.PauliX(2), qml.Toffoli((0, 1, 2))),
]


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


class TestDecomposition:

    def test_resource_keys(self):
        """Test that the resource keys of `Prod` are op_reps."""
        assert Prod.resource_keys == frozenset({"resources"})
        product = qml.X(0) @ qml.Y(1) @ qml.X(2)
        resources = {qml.resource_rep(qml.X): 2, qml.resource_rep(qml.Y): 1}
        assert product.resource_params == {"resources": resources}

    def test_registered_decomp(self):
        """Test that the decomposition of prod is registered."""

        decomps = qml.decomposition.list_decomps(Prod)

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
        """Test that prod's can be integrated into the decomposition."""

        op = qml.S(0) @ qml.S(1) @ qml.T(0) @ qml.Y(1)

        graph = qml.decomposition.DecompositionGraph([op], gate_set=set(qml.ops.__all__))
        solution = graph.solve()
        with qml.queuing.AnnotatedQueue() as q:
            solution.decomposition(op)(**op.hyperparameters)

        assert q.queue == list(op[::-1])
