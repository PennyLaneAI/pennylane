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
from pennylane import math
from pennylane.exceptions import DeviceError, MatrixUndefinedError
from pennylane.operation import Operator
from pennylane.ops.op_math.conjugation import Conjugation, _swappable_ops, conjugation
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
    """Run basic validity checks on a conjugation operator."""
    op1 = qml.PauliZ(0)
    op2 = qml.Rot(1.2, 2.3, 3.4, wires=0)
    op3 = qml.IsingZZ(4.32, wires=("a", "b"))
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

    @pytest.mark.parametrize("id", ("foo", "bar"))
    def test_init_conjugation_op(self, id):
        """Test the initialization of a Conjugation operator."""
        conjugation_op = conjugation(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"), id=id)

        assert conjugation_op.wires == Wires((0, "a"))
        assert conjugation_op.num_wires == 2
        assert conjugation_op.name == "Conjugation"
        assert conjugation_op.id == id

        assert conjugation_op.data == (0.23,)
        assert conjugation_op.parameters == [0.23]
        assert conjugation_op.num_params == 1

    def test_hash(self):
        """Testing some situations for the hash property."""
        # same hash if different order but can be permuted to right order
        op1 = qml.conjugation(qml.PauliX(0), qml.PauliY("a"))
        op2 = qml.conjugation(qml.PauliY("a"), qml.PauliX(0))
        assert op1.hash == op2.hash

        # test not the same hash if different order and cant be exchanged to correct order
        op3 = qml.conjugation(qml.PauliX("a"), qml.PauliY("a"), qml.PauliX(1))
        op4 = qml.conjugation(qml.PauliY("a"), qml.PauliX("a"), qml.PauliX(1))
        assert op3.hash != op4.hash

    PROD_TERMS_OP_PAIRS_MIXED = (  # not all operands have pauli representation
        (
            qml.conjugation(qml.Hadamard(0), X(1), X(2)),
            [1.0],
            [qml.conjugation(qml.Hadamard(0), X(1), X(2))],
        ),  # trivial conjugationuct
        (
            qml.conjugation(qml.Hadamard(0), X(1), qml.Identity(2)),
            [1.0],
            [qml.conjugation(qml.Hadamard(0), X(1))],
        ),
        (
            qml.conjugation(qml.Hadamard(0), qml.s_conjugation(4, X(1)), qml.s_conjugation(2, X(2))),
            [2 * 4],
            [qml.conjugation(qml.Hadamard(0), X(1), X(2))],
        ),  # conjugationuct with scalar conjugationucts inside
        (
            qml.conjugation(qml.Hadamard(0), qml.s_conjugation(4, X(0)), qml.s_conjugation(2, X(1))),
            [2 * 4],
            [qml.conjugation(qml.Hadamard(0), X(0), X(1))],
        ),  # conjugationuct with scalar conjugationucts on same wire
        (
            qml.conjugation(qml.Hadamard(0), qml.s_conjugation(4, Y(1)), qml.sum(X(2), X(3))),
            [4, 4],
            [qml.conjugation(qml.Hadamard(0), Y(1), X(2)), qml.conjugation(qml.Hadamard(0), Y(1), X(3))],
        ),  # conjugationuct with sums inside
        (
            qml.conjugation(
                qml.conjugation(qml.Hadamard(0), X(2), X(3)),
                qml.s_conjugation(0.5, qml.sum(qml.Hadamard(5), qml.s_conjugation(0.4, X(6)))),
            ),
            [0.5, 0.2],
            [
                qml.conjugation(X(2), X(3), qml.Hadamard(5), qml.Hadamard(0)),
                qml.conjugation(X(6), X(2), X(3), qml.Hadamard(0)),
            ],
        ),  # contrived example
    )

    @pytest.mark.parametrize("op, coeffs_true, ops_true", PROD_TERMS_OP_PAIRS_MIXED)
    def test_terms_no_pauli_rep(self, op, coeffs_true, ops_true):
        """Test that Conjugation.terms() is correct for operators that dont all have a pauli_rep"""
        coeffs, ops1 = op.terms()
        assert coeffs == coeffs_true
        assert ops1 == ops_true

    PROD_TERMS_OP_PAIRS_PAULI = (  # all operands have pauli representation
        (qml.conjugation(X(0), X(1), X(2)), [1.0], [qml.conjugation(X(0), X(1), X(2))]),  # trivial conjugationuct
        (
            qml.conjugation(X(0), X(1), X(2), qml.Identity(0)),
            [1.0],
            [qml.conjugation(X(0), X(1), X(2))],
        ),  # trivial conjugationuct
        (
            qml.conjugation(X(0), qml.s_conjugation(4, X(1)), qml.s_conjugation(2, X(2))),
            [2 * 4],
            [qml.conjugation(X(0), X(1), X(2))],
        ),  # conjugationuct with scalar conjugationucts inside
        (
            qml.conjugation(X(0), qml.s_conjugation(4, X(0)), qml.s_conjugation(2, X(1))),
            [2 * 4],
            [X(1)],
        ),  # conjugationuct with scalar conjugationucts on same wire
        (
            qml.conjugation(X(0), qml.s_conjugation(4, Y(0)), qml.s_conjugation(2, X(1))),
            [1j * 2 * 4],
            [qml.conjugation(Z(0), X(1))],
        ),  # conjugationuct with scalar conjugationucts on same wire
        (
            qml.conjugation(X(0), qml.s_conjugation(4, Y(1)), qml.sum(X(2), X(3))),
            [4, 4],
            [qml.conjugation(X(0), Y(1), X(2)), qml.conjugation(X(0), Y(1), X(3))],
        ),  # conjugationuct with sums inside
    )

    @pytest.mark.parametrize("op, coeffs_true, ops_true", PROD_TERMS_OP_PAIRS_PAULI)
    def test_terms_pauli_rep(self, op, coeffs_true, ops_true):
        """Test that Conjugation.terms() is correct for operators that all have a pauli_rep"""
        coeffs, ops1 = op.terms()
        assert coeffs == coeffs_true
        assert ops1 == ops_true

    def test_terms_pauli_rep_wire_order(self):
        """Test that the wire order of the terms is the same as the wire order of the original
        operands when the Conjugation has a valid pauli_rep"""
        H = qml.conjugation(X(0), X(1), X(2))
        _, H_ops = H.terms()

        assert len(H_ops) == 1
        assert H_ops[0].wires == H.wires

    def test_batch_size(self):
        """Test that batch size returns the batch size of a base operation if it is batched."""
        x = qml.numpy.array([1.0, 2.0, 3.0])
        conjugation_op = conjugation(qml.PauliX(0), qml.RX(x, wires=0))
        assert conjugation_op.batch_size == 3

    @pytest.mark.parametrize(
        "op, coeffs_true, ops_true", PROD_TERMS_OP_PAIRS_PAULI + PROD_TERMS_OP_PAIRS_MIXED
    )
    def test_terms_does_not_change_queue(self, op, coeffs_true, ops_true):
        """Test that calling Conjugation.terms does not queue anything."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.apply(op)
            _, _ = op.terms()

        assert q.queue == [op]

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
        """Test that a conjugationuct of operators that have `has_matrix=True`
        has `has_matrix=True` as well."""

        conjugation_op = conjugation(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"))
        assert conjugation_op.has_matrix is True

    def test_has_matrix_true_via_factor_has_no_matrix_but_is_hamiltonian(self):
        """Test that a conjugationuct of operators of which one does not have `has_matrix=True`
        but is a Hamiltonian has `has_matrix=True`."""

        H = qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])
        conjugation_op = conjugation(H, qml.RZ(0.23, wires=5))
        assert conjugation_op.has_matrix is True

    @pytest.mark.parametrize(
        "first_factor", [qml.PauliX(wires=0), qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])]
    )
    def test_has_matrix_false_via_factor_has_no_matrix(self, first_factor):
        """Test that a conjugationuct of operators of which one does not have `has_matrix=True`
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
        """Test that a conjugationuct of operators that have `has_matrix=True`
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
        """Test that a conjugationuct of operators that have `has_decomposition=True`
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
        """Test that a conjugationuct of operators that have `has_diagonalizing_gates=True`
        has `has_diagonalizing_gates=True` as well."""

        conjugation_op = conjugation(*factors)
        assert conjugation_op.has_diagonalizing_gates is True

    @pytest.mark.parametrize(
        "factors",
        (
            [qml.PauliX(wires=0), qml.PauliX(wires=1)],
            [qml.PauliX(wires=0), qml.PauliZ(wires="r"), qml.PauliY("s")],
            [qml.Hermitian(np.eye(4), wires=[0, 2]), qml.PauliX(wires=1)],
        ),
    )
    def test_has_diagonalizing_gates_true_via_factors(self, factors):
        """Test that a conjugationuct of operators that have `has_diagonalizing_gates=True`
        has `has_diagonalizing_gates=True` as well."""

        conjugation_op = conjugation(*factors)
        assert conjugation_op.has_diagonalizing_gates is True

    def test_has_diagonalizing_gates_false_via_factor(self):
        """Test that a conjugationuct of operators of which one has
        `has_diagonalizing_gates=False` has `has_diagonalizing_gates=False` as well."""

        conjugation_op = conjugation(MyOp(3.1, 0), qml.PauliX(2))
        assert conjugation_op.has_diagonalizing_gates is False

    def test_qfunc_init(self):
        """Tests conjugation initialization with a qfunc argument."""

        def qfunc():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.RZ(1.1, 1)

        conjugation_gen = conjugation(qfunc)
        assert callable(conjugation_gen)
        conjugation_op = conjugation_gen()
        expected = conjugation(qml.RZ(1.1, 1), qml.CNOT([0, 1]), qml.Hadamard(0))
        qml.assert_equal(conjugation_op, expected)
        assert conjugation_op.wires == Wires([1, 0])

    def test_qfunc_single_operator(self):
        """Test conjugation initialization with qfunc that queues a single operator."""

        def qfunc():
            qml.S(0)

        with qml.queuing.AnnotatedQueue() as q:
            out = conjugation(qfunc)()

        assert len(q) == 1
        assert q.queue[0] == qml.S(0)
        assert out == qml.S(0)

    def test_qfunc_init_accepts_args_kwargs(self):
        """Tests that conjugation preserves args when wrapping qfuncs."""

        def qfunc(x, run_had=False):
            if run_had:
                qml.Hadamard(0)
            qml.RX(x, 0)
            qml.CNOT([0, 1])

        conjugation_gen = conjugation(qfunc)
        qml.assert_equal(conjugation_gen(1.1), conjugation(qml.CNOT([0, 1]), qml.RX(1.1, 0)))
        qml.assert_equal(
            conjugation_gen(2.2, run_had=True), conjugation(qml.CNOT([0, 1]), qml.RX(2.2, 0), qml.Hadamard(0))
        )

    def test_qfunc_init_propagates_Conjugation_kwargs(self):
        """Tests that additional kwargs for Conjugation are propagated using qfunc initialization."""

        def qfunc(x):
            qml.conjugation(qml.RX(x, 0), qml.PauliZ(1))
            qml.CNOT([0, 1])

        conjugation_gen = conjugation(qfunc, id=123987, lazy=False)
        conjugation_op = conjugation_gen(1.1)

        assert conjugation_op.id == 123987  # id was set
        qml.assert_equal(conjugation_op, conjugation(qml.CNOT([0, 1]), qml.PauliZ(1), qml.RX(1.1, 0)))  # eager

    def test_qfunc_init_only_works_with_one_qfunc(self):
        """Test that the qfunc init only occurs when one callable is passed to conjugation."""

        def qfunc():
            qml.Hadamard(0)
            qml.CNOT([0, 1])

        conjugation_op = conjugation(qfunc)()
        qml.assert_equal(conjugation_op, conjugation(qml.CNOT([0, 1]), qml.Hadamard(0)))

        def fn2():
            qml.PauliX(0)
            qml.PauliY(1)

        for args in [(qfunc, fn2), (qfunc, qml.PauliX), (qml.PauliX, qfunc)]:
            with pytest.raises(AttributeError, match="has no attribute 'wires'"):
                conjugation(*args)

    def test_qfunc_init_returns_single_op(self):
        """Tests that if a qfunc only queues one operator, that operator is returned."""

        def qfunc():
            qml.PauliX(0)

        conjugation_op = conjugation(qfunc)()
        qml.assert_equal(conjugation_op, qml.PauliX(0))
        assert not isinstance(conjugation_op, Conjugation)

    @pytest.mark.xfail  # this requirement has been lifted
    def test_conjugation_accepts_single_operator_but_Conjugation_does_not(self):
        """Tests that the conjugation wrapper can accept a single operator, and return it."""

        x = qml.PauliX(0)
        conjugation_op = conjugation(x)
        assert conjugation_op is x
        assert not isinstance(conjugation_op, Conjugation)

        with pytest.raises(ValueError, match="Require at least two operators"):
            Conjugation(x)

    def test_conjugation_fails_with_non_callable_arg(self):
        """Tests that conjugation explicitly checks that a single-arg is either an Operator or callable."""
        with pytest.raises(TypeError, match="Unexpected argument of type int passed to qml.conjugation"):
            conjugation(1)


def test_empty_repr():
    """Test that an empty conjugation still has a repr that indicates it's a conjugation."""
    assert repr(Conjugation()) == "Conjugation()"


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
        """Test matrix method for a conjugationuct of non_parametric ops"""
        mat1, mat2 = compare_and_expand_mat(mat1, mat2)
        true_mat = mat1 @ mat2

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
        """Test matrix method for a conjugationuct of parametric ops"""
        par1 = tuple(range(op1.num_params))
        par2 = tuple(range(op2.num_params))
        mat1, mat2 = compare_and_expand_mat(mat1(*par1), mat2(*par2))

        conjugation_op = Conjugation(
            op1(*par1, wires=range(op1.num_wires)), op2(*par2, wires=range(op2.num_wires))
        )
        conjugation_mat = conjugation_op.matrix()
        true_mat = mat1 @ mat2
        assert np.allclose(conjugation_mat, true_mat)

    @pytest.mark.parametrize("op", no_mat_ops)
    def test_error_no_mat(self, op: Operator):
        """Test that an error is raised if one of the factors doesn't
        have its matrix method defined."""
        conjugation_op = Conjugation(op(wires=0), qml.PauliX(wires=2), qml.PauliZ(wires=1))
        with pytest.raises(MatrixUndefinedError):
            conjugation_op.matrix()

    def test_conjugation_ops_multi_terms(self):
        """Test matrix is correct for a conjugationuct of more than two terms."""
        conjugation_op = Conjugation(qml.PauliX(wires=0), qml.PauliY(wires=0), qml.PauliZ(wires=0))
        mat = conjugation_op.matrix()

        true_mat = math.array(
            [
                [1j, 0],
                [0, 1j],
            ]
        )
        assert np.allclose(mat, true_mat)

    def test_conjugation_ops_multi_wires(self):
        """Test matrix is correct when multiple wires are used in the conjugationuct."""
        conjugation_op = Conjugation(qml.PauliX(wires=0), qml.Hadamard(wires=1), qml.PauliZ(wires=2))
        mat = conjugation_op.matrix()

        x = math.array([[0, 1], [1, 0]])
        h = 1 / math.sqrt(2) * math.array([[1, 1], [1, -1]])
        z = math.array([[1, 0], [0, -1]])

        true_mat = math.kron(x, math.kron(h, z))
        assert np.allclose(mat, true_mat)

    def test_conjugation_ops_wire_order(self):
        """Test correct matrix is returned when the wire_order arg is provided."""
        conjugation_op = Conjugation(qml.PauliZ(wires=2), qml.PauliX(wires=0), qml.Hadamard(wires=1))
        wire_order = [0, 1, 2]
        mat = conjugation_op.matrix(wire_order=wire_order)

        x = math.array([[0, 1], [1, 0]])
        h = 1 / math.sqrt(2) * math.array([[1, 1], [1, -1]])
        z = math.array([[1, 0], [0, -1]])

        true_mat = math.kron(x, math.kron(h, z))
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
        conjugation_op = Conjugation(qml.QFT(wires=wires), qml.GroverOperator(wires=wires), qml.PauliX(wires=0))
        mat = conjugation_op.matrix()

        grov_mat = (1 / 4) * math.ones((8, 8), dtype="complex128") - math.eye(8, dtype="complex128")
        qft_mat = get_qft_mat(3)
        x = math.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]])
        x_mat = math.kron(x, math.eye(4, dtype="complex128"))

        true_mat = qft_mat @ grov_mat @ x_mat
        assert np.allclose(mat, true_mat)

    def test_conjugation_qchem_ops(self):
        """Test that qchem operations can be composed and the generated matrix is correct."""
        wires = [0, 1, 2, 3]
        conjugation_op = Conjugation(
            qml.OrbitalRotation(4.56, wires=wires),
            qml.SingleExcitation(1.23, wires=[0, 1]),
            qml.PauliX(4),
        )
        mat = conjugation_op.matrix()

        or_mat = math.kron(gd.OrbitalRotation(4.56), math.eye(2))
        se_mat = math.kron(gd.SingleExcitation(1.23), math.eye(8, dtype="complex128"))
        x = math.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]])
        x_mat = math.kron(math.eye(16, dtype="complex128"), x)

        true_mat = or_mat @ se_mat @ x_mat
        assert np.allclose(mat, true_mat)

    def test_conjugation_observables(self):
        """Test that observable objects can also be composed with correct matrix representation."""
        wires = [0, 1]
        conjugation_op = Conjugation(
            qml.Hermitian(qnp.array([[0.0, 1.0], [1.0, 0.0]]), wires=2),
            qml.Projector(state=qnp.array([0, 1]), wires=wires),
        )
        mat = conjugation_op.matrix()

        hermitian_mat = qnp.array([[0.0, 1.0], [1.0, 0.0]])
        proj_mat = qnp.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )

        true_mat = qnp.kron(hermitian_mat, proj_mat)
        assert np.allclose(mat, true_mat)

    def test_conjugation_qubit_unitary(self):
        """Test that an arbitrary QubitUnitary can be composed with correct matrix representation."""
        U = 1 / qnp.sqrt(2) * qnp.array([[1, 1], [1, -1]])  # Hadamard
        U_op = qml.QubitUnitary(U, wires=0)

        conjugation_op = Conjugation(U_op, qml.Identity(wires=1))
        mat = conjugation_op.matrix()

        true_mat = qnp.kron(U, qnp.eye(2)) @ qnp.eye(4)
        assert np.allclose(mat, true_mat)

    def test_conjugation_hamiltonian(self):
        """Test that a hamiltonian object can be composed."""
        U = qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])
        conjugation_op = Conjugation(qml.PauliZ(wires=0), U)
        mat = conjugation_op.matrix()

        true_mat = [
            [0.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5, 0.0],
        ]
        assert np.allclose(mat, true_mat)

    def test_matrix_all_batched(self):
        """Test that Conjugation matrix has batching support when all operands are batched."""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        y = qml.numpy.array([0.4, 0.5, 0.6])
        op = conjugation(qml.RX(x, wires=0), qml.RY(y, wires=2), qml.PauliZ(1))
        mat = op.matrix()
        sum_list = [
            conjugation(qml.RX(i, wires=0), qml.RY(j, wires=2), qml.PauliZ(1)) for i, j in zip(x, y)
        ]
        compare = qml.math.stack([s.matrix() for s in sum_list])
        assert qml.math.allclose(mat, compare)
        assert mat.shape == (3, 8, 8)

    def test_matrix_not_all_batched(self):
        """Test that Conjugation matrix has batching support when all operands are not batched."""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        y = 0.5
        z = qml.numpy.array([0.4, 0.5, 0.6])
        op = conjugation(
            qml.RX(x, wires=0),
            qml.RY(y, wires=2),
            qml.RZ(z, wires=1),
            qml.conjugation(qml.PauliX(2), qml.PauliY(3)),
        )
        mat = op.matrix()
        batched_y = [y for _ in x]
        sum_list = [
            conjugation(
                qml.RX(i, wires=0),
                qml.RY(j, wires=2),
                qml.RZ(k, wires=1),
                qml.conjugation(qml.PauliX(2), qml.PauliY(3)),
            )
            for i, j, k in zip(x, batched_y, z)
        ]
        compare = qml.math.stack([s.matrix() for s in sum_list])
        assert qml.math.allclose(mat, compare)
        assert mat.shape == (3, 16, 16)

    # Add interface tests for each interface !

    @pytest.mark.jax
    def test_conjugation_jax(self):
        """Test matrix is cast correctly using jax parameters."""
        import jax.numpy as jnp

        theta = jnp.array(1.23)
        rot_params = jnp.array([0.12, 3.45, 6.78])

        conjugation_op = Conjugation(
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qml.RX(theta, wires=1),
            qml.Identity(wires=0),
        )
        mat = conjugation_op.matrix()

        true_mat = (
            jnp.kron(gd.Rot3(rot_params[0], rot_params[1], rot_params[2]), qnp.eye(2))
            @ jnp.kron(qnp.eye(2), gd.Rotx(theta))
            @ qnp.eye(4)
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
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qml.RX(theta, wires=1),
            qml.Identity(wires=0),
        )
        mat = conjugation_op.matrix()

        true_mat = (
            qnp.kron(gd.Rot3(rot_params[0], rot_params[1], rot_params[2]), qnp.eye(2))
            @ qnp.kron(qnp.eye(2), gd.Rotx(theta))
            @ qnp.eye(4)
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
            qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qml.RX(theta, wires=1),
            qml.Identity(wires=0),
        )
        mat = conjugation_op.matrix()

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
        """Test that the sparse matrix of a Conjugation op is defined and correct."""
        conjugation_op = conjugation(op1(wires=0), op2(wires=1))
        true_mat = math.kron(mat1, mat2)
        conjugation_mat = conjugation_op.sparse_matrix().todense()

        assert np.allclose(true_mat, conjugation_mat)

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix_format(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix accepts the format parameter."""
        from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix

        conjugation_op = qml.sum(op1(wires=0), op2(wires=1))
        true_mat = math.kron(mat1, np.eye(2)) + math.kron(np.eye(2), mat2)
        assert isinstance(conjugation_op.sparse_matrix(), csr_matrix)
        conjugation_op_csc = conjugation_op.sparse_matrix(format="csc")
        conjugation_op_lil = conjugation_op.sparse_matrix(format="lil")
        conjugation_op_coo = conjugation_op.sparse_matrix(format="coo")
        assert isinstance(conjugation_op_csc, csc_matrix)
        assert isinstance(conjugation_op_lil, lil_matrix)
        assert isinstance(conjugation_op_coo, coo_matrix)
        assert np.allclose(true_mat, conjugation_op_csc.todense())
        assert np.allclose(true_mat, conjugation_op_lil.todense())
        assert np.allclose(true_mat, conjugation_op_coo.todense())

    def test_sparse_matrix_global_phase(self):
        """Test that a conjugation with a global phase still defines a sparse matrix."""

        op = qml.GlobalPhase(0.5) @ qml.X(0) @ qml.X(0)

        sparse_mat = op.sparse_matrix(wire_order=(0, 1))
        mat = sparse_mat.todense()
        assert qml.math.allclose(mat, np.exp(-0.5j) * np.eye(4))

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix_same_wires(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix of a Conjugation op is defined and correct."""
        conjugation_op = conjugation(op1(wires=0), op2(wires=0))
        true_mat = mat1 @ mat2
        conjugation_mat = conjugation_op.sparse_matrix().todense()

        assert np.allclose(true_mat, conjugation_mat)

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix_wire_order(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix of a Conjugation op is defined
        with wire order and correct."""
        true_mat = math.kron(math.kron(mat2, np.eye(2)), mat1)

        conjugation_op = conjugation(op1(wires=2), op2(wires=0))
        conjugation_mat = conjugation_op.sparse_matrix(wire_order=[0, 1, 2]).todense()

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

    def test_queue_category_none(self):
        """Test _queue_category property is None when any factor is not `_ops`."""

        class DummyOp(Operator):  # pylint:disable=too-few-public-methods
            """Dummy op with None queue category"""

            _queue_category = None
            num_wires = 1

        conjugation_op = conjugation(qml.Identity(wires=0), DummyOp(wires=0))
        assert conjugation_op._queue_category is None

    def test_eigendecomposition(self):
        """Test that the computed Eigenvalues and Eigenvectors are correct."""
        diag_conjugation_op = Conjugation(qml.PauliZ(wires=0), qml.PauliZ(wires=1))
        eig_decomp = diag_conjugation_op.eigendecomposition
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

        conjugation_op = qml.conjugation(op1, op2)
        eigs = conjugation_op.eigvals()

        mat_eigs = np.linalg.eigvals(conjugation_op.matrix())

        sorted_eigs = np.sort(eigs)
        sorted_mat_eigs = np.sort(mat_eigs)
        assert qml.math.allclose(sorted_eigs, sorted_mat_eigs)

        # pylint: disable=import-outside-top-level
        from pennylane.ops.functions.assert_valid import _check_eigendecomposition

        _check_eigendecomposition(conjugation_op)

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

    def test_eigvals_no_wires_identity(self):
        """Test that eigvals can be computed if a component is an identity on no wires."""
        op = qml.X(0) @ qml.Y(1) @ qml.I()
        op2 = qml.X(0) @ qml.Y(1)

        assert qml.math.allclose(op.eigvals(), op2.eigvals())


class TestWrapperFunc:
    """Test wrapper function."""

    def test_op_conjugation_top_level(self):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        factors = (qml.PauliX(wires=1), qml.RX(1.23, wires=0), qml.CNOT(wires=[0, 1]))
        op_id = "conjugation_op"

        conjugation_func_op = conjugation(*factors, id=op_id)
        conjugation_class_op = Conjugation(*factors, id=op_id)
        qml.assert_equal(conjugation_func_op, conjugation_class_op)

    def test_lazy_mode(self):
        """Test that by default, the operator is simply wrapped in `Conjugation`, even if a simplification exists."""
        op = conjugation(qml.S(0), Conjugation(qml.S(1), qml.T(1)))

        assert isinstance(op, Conjugation)
        assert len(op) == 2

    def test_non_lazy_mode(self):
        """Test the lazy=False keyword."""
        op = conjugation(qml.S(0), Conjugation(qml.S(1), qml.T(1)), lazy=False)

        assert isinstance(op, Conjugation)
        assert len(op) == 3

    def test_nonlazy_mode_queueing(self):
        """Test that if a simpification is accomplished, the metadata for the original op
        and the new simplified op is updated."""
        with qml.queuing.AnnotatedQueue() as q:
            conjugation1 = conjugation(qml.S(1), qml.T(1))
            conjugation2 = conjugation(qml.S(0), conjugation1, lazy=False)

        assert len(q) == 1
        assert q.queue[0] is conjugation2

    def test_correct_queued_operators(self):
        """Test that args and kwargs do not add operators to the queue."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.conjugation(qml.QSVT)(qml.X(1), [qml.Z(1)])
            qml.conjugation(qml.QSVT(qml.X(1), [qml.Z(1)]))

        for op in q.queue:
            assert op.name == "QSVT"

        assert len(q.queue) == 2


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

    def test_measurement_process_sample(self):
        """Test Conjugation class instance in sample measurement process."""
        dev = qml.device("default.qubit", wires=2, shots=20)
        conjugation_op = Conjugation(qml.PauliX(wires=0), qml.PauliX(wires=1))

        @qml.qnode(dev)
        def my_circ():
            Conjugation(qml.Hadamard(0), qml.Hadamard(1))
            return qml.sample(op=conjugation_op)

        results = my_circ()

        assert len(results) == 20
        assert (results == 1).all()

    def test_measurement_process_counts(self):
        """Test Conjugation class instance in sample measurement process."""
        dev = qml.device("default.qubit", wires=2, shots=20)
        conjugation_op = Conjugation(qml.PauliX(wires=0), qml.PauliX(wires=1))

        @qml.qnode(dev)
        def my_circ():
            Conjugation(qml.Hadamard(0), qml.Hadamard(1))
            return qml.counts(op=conjugation_op)

        results = my_circ()

        assert sum(results.values()) == 20
        assert 1 in results  # pylint:disable=unsupported-membership-test
        assert -1 not in results  # pylint:disable=unsupported-membership-test

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
            qml.RX(1.23, wires=1),
            qml.IsingZZ(0.56, wires=[0, 2]),
        )

        @qml.qnode(dev)
        def conjugation_state_circ():
            Conjugation(*operands)
            return qml.state()

        @qml.qnode(dev)
        def true_state_circ():
            qml.IsingZZ(0.56, wires=[0, 2])
            qml.RX(1.23, wires=1)
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
            qml.RY(y, wires=0)
            qml.RX(x, wires=0)
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
        assert tape.trainable_params == [1]


class TestDecomposition:

    def test_resource_keys(self):
        """Test that the resource keys of `Conjugation` are op_reps."""
        assert Conjugation.resource_keys == frozenset({"resources"})
        conjugationuct = qml.X(0) @ qml.Y(1) @ qml.X(2)
        resources = {qml.resource_rep(qml.X): 2, qml.resource_rep(qml.Y): 1}
        assert conjugationuct.resource_params == {"resources": resources}

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

        op = qml.S(0) @ qml.S(1) @ qml.T(0) @ qml.Y(1)

        graph = qml.decomposition.DecompositionGraph([op], gate_set=set(qml.ops.__all__))
        graph.solve()
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op)(**op.hyperparameters)

        assert q.queue == list(op[::-1])
