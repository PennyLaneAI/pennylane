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
# pylint: disable=eval-used, unused-argument


import gate_data as gd  # a file containing matrix rep of each gate
import numpy as np
import pytest

import pennylane as qp
import pennylane.numpy as qnp
from pennylane import X, Y, Z, math
from pennylane.exceptions import MatrixUndefinedError
from pennylane.operation import Operator
from pennylane.ops.op_math import Prod, Sum
from pennylane.wires import Wires

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
    (qp.PauliX(wires=0), qp.PauliZ(wires=0), qp.Hadamard(wires=0)),
    (qp.CNOT(wires=[0, 1]), qp.RX(1.23, wires=1), qp.Identity(wires=0)),
    (
        qp.IsingXX(4.56, wires=[2, 3]),
        qp.Toffoli(wires=[1, 2, 3]),
        qp.Rot(0.34, 1.0, 0, wires=0),
    ),
)


def _get_pw(w, pauli_op):
    return qp.pauli.PauliWord({w: pauli_op})


# pylint: disable=unused-argument
def sum_using_dunder_method(*summands, id=None):
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

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, qp.sum])
    @pytest.mark.parametrize("id", ("foo", "bar"))
    def test_init_sum_op(self, id, sum_method):
        """Test the initialization of a Sum operator."""
        sum_op = sum_method(qp.PauliX(wires=0), qp.RZ(0.23, wires="a"), id=id)

        assert sum_op.wires == Wires((0, "a"))
        assert sum_op.num_wires == 2
        assert sum_op.name == "Sum"
        if sum_method.__name__ == sum.__name__:
            assert sum_op.id == id

        assert sum_op.data == (0.23,)
        assert sum_op.parameters == [0.23]
        assert sum_op.num_params == 1

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, qp.sum])
    def test_init_sum_op_with_sum_summands(self, sum_method):
        """Test the initialization of a Sum operator which contains a summand that is another
        Sum operator."""
        sum_op = sum_method(
            Sum(qp.PauliX(wires=0), qp.RZ(0.23, wires="a")), qp.RX(9.87, wires=0)
        )
        assert sum_op.wires == Wires((0, "a"))
        assert sum_op.num_wires == 2
        assert sum_op.name == "Sum"
        assert sum_op.id is None

        assert sum_op.data == (0.23, 9.87)
        assert sum_op.parameters == [0.23, 9.87]
        assert sum_op.num_params == 2

    coeffs_ = [1.0, 1.0, 1.0, 3.0, 4.0, 4.0, 5.0]
    h6 = qp.sum(
        qp.s_prod(2.0, qp.prod(qp.PauliX(0), qp.PauliZ(10))),
        qp.s_prod(3.0, qp.prod(qp.PauliX(1), qp.PauliZ(11))),
    )
    ops_ = [
        qp.s_prod(1.0, qp.PauliX(0)),
        qp.s_prod(1.0, qp.prod(qp.PauliX(0), qp.PauliX(1))),
        qp.s_prod(2.0, qp.prod(qp.PauliZ(0), qp.PauliZ(1))),
        qp.s_prod(2.0, qp.PauliY(0)),
        qp.s_prod(2.0, qp.prod(qp.PauliY(0), qp.PauliY(1))),
        qp.s_prod(2.0, qp.prod(qp.s_prod(2.0, qp.PauliY(2)), qp.PauliY(3))),
        h6,
    ]
    Hamiltonian_with_Paulis = qp.dot(coeffs_, ops_)

    SUM_TERMS_OP_PAIRS_PAULI = (  # all operands have pauli representation
        (
            qp.sum(*(i * X(i) for i in range(1, 5))),
            [float(i) for i in range(1, 5)],
            [X(i) for i in range(1, 5)],
        ),
        (
            qp.sum(*(qp.s_prod(i, qp.prod(X(i), X(i + 1))) for i in range(1, 5))),
            [float(i) for i in range(1, 5)],
            [qp.prod(X(i), X(i + 1)) for i in range(1, 5)],
        ),
        (
            Hamiltonian_with_Paulis,
            [1.0, 1.0, 2.0, 6.0, 8.0, 16.0, 10.0, 15.0],
            [
                X(0),
                qp.prod(X(1), X(0)),
                qp.prod(Z(1), Z(0)),
                Y(0),
                qp.prod(Y(1), Y(0)),
                qp.prod(Y(3), Y(2)),
                qp.prod(Z(10), X(0)),
                qp.prod(Z(11), X(1)),
            ],
        ),
    )

    @pytest.mark.parametrize("op, coeffs_true, ops_true", SUM_TERMS_OP_PAIRS_PAULI)
    def test_terms_pauli_rep(self, op, coeffs_true, ops_true):
        """Test that Sum.terms() is correct for operators that all have a pauli_rep"""
        coeffs, ops1 = op.terms()
        assert coeffs == coeffs_true
        assert ops1 == ops_true

    def test_terms_pauli_rep_wire_order(self):
        """Test that the wire order of the terms is the same as the wire order of the original
        operands when the Sum has a valid pauli_rep"""
        w0, w1, w2, w3 = [0, 1, 2, 3]
        coeffs = [0.5, -0.5]

        obs = [
            qp.X(w0) @ qp.Y(w1) @ qp.X(w2) @ qp.Z(w3),
            qp.X(w0) @ qp.X(w1) @ qp.Y(w2) @ qp.Z(w3),
        ]

        H = qp.dot(coeffs, obs)
        _, H_ops = H.terms()

        assert all(o1.wires == o2.wires for o1, o2 in zip(obs, H_ops))
        assert H_ops[0] == qp.prod(qp.X(w0), qp.Y(w1), qp.X(w2), qp.Z(w3))
        assert H_ops[1] == qp.prod(qp.X(w0), qp.X(w1), qp.Y(w2), qp.Z(w3))

    coeffs_ = [1.0, 1.0, 1.0, 3.0, 4.0, 4.0, 5.0]
    h6 = qp.sum(
        qp.s_prod(2.0, qp.prod(qp.Hadamard(0), qp.PauliZ(10))),
        qp.s_prod(3.0, qp.prod(qp.PauliX(1), qp.PauliZ(11))),
    )
    ops_ = [
        qp.s_prod(1.0, qp.Hadamard(0)),
        qp.s_prod(1.0, qp.prod(qp.Hadamard(0), qp.PauliX(1))),
        qp.s_prod(2.0, qp.prod(qp.PauliZ(0), qp.PauliZ(1))),
        qp.s_prod(2.0, qp.PauliY(0)),
        qp.s_prod(2.0, qp.prod(qp.PauliY(0), qp.PauliY(1))),
        qp.s_prod(2.0, qp.prod(qp.s_prod(2.0, qp.PauliY(2)), qp.PauliY(3))),
        h6,
    ]
    Hamiltonian_mixed = qp.dot(coeffs_, ops_)

    SUM_TERMS_OP_PAIRS_MIXEDPAULI = (  # not all operands have pauli representation
        (
            qp.sum(*(i * qp.Hadamard(i) for i in range(1, 5))),
            [float(i) for i in range(1, 5)],
            [qp.Hadamard(i) for i in range(1, 5)],
        ),
        (
            qp.sum(qp.sum(*(i * qp.Hadamard(i) for i in range(1, 5))), 0.0 * qp.Identity(0)),
            [float(i) for i in range(1, 5)],
            [qp.Hadamard(i) for i in range(1, 5)],
        ),
        (
            qp.sum(qp.sum(*(i * qp.Hadamard(i) for i in range(1, 5))), qp.Identity(0)),
            [float(i) for i in range(1, 5)] + [1.0],
            [qp.Hadamard(i) for i in range(1, 5)] + [qp.Identity(0)],
        ),
        (
            qp.sum(*(qp.s_prod(i, qp.prod(X(i), qp.Hadamard(i + 1))) for i in range(1, 5))),
            [float(i) for i in range(1, 5)],
            [qp.prod(X(i), qp.Hadamard(i + 1)) for i in range(1, 5)],
        ),
        (
            Hamiltonian_mixed,
            [1.0, 1.0, 2.0, 6.0, 8.0, 16.0, 10.0, 15.0],
            [
                qp.Hadamard(0),
                qp.prod(X(1), qp.Hadamard(0)),
                qp.prod(Z(1), Z(0)),
                Y(0),
                qp.prod(Y(1), Y(0)),
                qp.prod(Y(3), Y(2)),
                qp.prod(Z(10), qp.Hadamard(0)),
                qp.prod(Z(11), X(1)),
            ],
        ),
    )

    @pytest.mark.parametrize("op, coeffs_true, ops_true", SUM_TERMS_OP_PAIRS_MIXEDPAULI)
    def test_terms_mixed(self, op, coeffs_true, ops_true):
        """Test that Sum.terms() is correct for operators that dont all have a pauli_rep"""
        coeffs, ops1 = op.terms()
        assert coeffs == coeffs_true
        assert ops1 == ops_true

    @pytest.mark.parametrize(
        "op, coeffs_true, ops_true", SUM_TERMS_OP_PAIRS_PAULI + SUM_TERMS_OP_PAIRS_MIXEDPAULI
    )
    def test_terms_does_not_change_queue(self, op, coeffs_true, ops_true):
        """Test that calling Prod.terms does not queue anything."""
        with qp.queuing.AnnotatedQueue() as q:
            qp.apply(op)
            _, _ = op.terms()

        assert q.queue == [op]

    def test_eigen_caching(self):
        """Test that the eigendecomposition is stored in cache."""
        diag_sum_op = Sum(qp.PauliZ(wires=0), qp.Identity(wires=1))
        eig_decomp = diag_sum_op.eigendecomposition

        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        eigs_cache = diag_sum_op._eigs[diag_sum_op.hash]  # pylint: disable=protected-access
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)

    SUM_REPR = (
        (qp.sum(), "Sum()"),
        (qp.sum(X(0), Y(1), Z(2)), "X(0) + Y(1) + Z(2)"),
        (X(0) + X(1) + X(2), "X(0) + X(1) + X(2)"),
        (0.5 * X(0) + 0.7 * X(1), "0.5 * X(0) + 0.7 * X(1)"),
        (0.5 * (X(0) @ X(1)) + 0.7 * X(1), "0.5 * (X(0) @ X(1)) + 0.7 * X(1)"),
        (
            0.5 * (X(0) @ (0.5 * X(1))) + 0.7 * X(1) + 0.8 * qp.CNOT((0, 1)),
            "(\n    0.5 * (X(0) @ (0.5 * X(1)))\n  + 0.7 * X(1)\n  + 0.8 * CNOT(wires=[0, 1])\n)",
        ),
        (
            0.5 * (X(0) @ (0.5 * X(1))) + 0.7 * X(1) + 0.8 * (X(0) @ Y(1) @ Z(1)),
            "(\n    0.5 * (X(0) @ (0.5 * X(1)))\n  + 0.7 * X(1)\n  + 0.8 * (X(0) @ Y(1) @ Z(1))\n)",
        ),
    )

    @pytest.mark.parametrize("op, repr_true", SUM_REPR)
    def test_repr(self, op, repr_true):
        """Test the string representation of Sum instances"""
        assert repr(op) == repr_true

    SUM_REPR_EVAL = (
        X(0) + Y(1) + Z(2),  # single line output
        0.5 * X(0) + 3.5 * Y(1) + 10 * Z(2),  # single line output
        X(0) @ X(1) + Y(1) @ Y(2) + Z(2),  # single line output
        0.5 * (X(0) @ X(1) @ X(2))
        + 1000 * (Y(1) @ X(0) @ X(1))
        + 1000000000 * Z(2),  # multiline output
        # qp.sum(*[0.5 * X(i) for i in range(10)]) # multiline output needs fixing of https://github.com/PennyLaneAI/pennylane/issues/5162 before working
    )

    @pytest.mark.parametrize("op", SUM_REPR_EVAL)
    def test_eval_sum(self, op):
        """Test that string representations of Sum can be evaluated and yield the same operator"""
        qp.assert_equal(eval(repr(op)), op)


class TestMatrix:
    """Test matrix-related methods."""

    @pytest.mark.parametrize("op_and_mat1", non_param_ops)
    @pytest.mark.parametrize("op_and_mat2", non_param_ops)
    def test_non_parametric_ops_two_terms(
        self,
        op_and_mat1: tuple[Operator, np.ndarray],
        op_and_mat2: tuple[Operator, np.ndarray],
    ):
        """Test matrix method for a sum of non_parametric ops"""
        op1, mat1 = op_and_mat1
        op2, mat2 = op_and_mat2
        mat1, mat2 = compare_and_expand_mat(mat1, mat2)
        true_mat = mat1 + mat2

        sum_op = Sum(
            op1(wires=0 if op1.num_wires is None else range(op1.num_wires)),
            op2(wires=0 if op2.num_wires is None else range(op2.num_wires)),
        )
        sum_mat = sum_op.matrix()

        assert np.allclose(sum_mat, true_mat)

    @pytest.mark.parametrize("op_mat1", param_ops)
    @pytest.mark.parametrize("op_mat2", param_ops)
    def test_parametric_ops_two_terms(
        self, op_mat1: tuple[Operator, np.ndarray], op_mat2: tuple[Operator, np.ndarray]
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
        sum_op = Sum(op(wires=0), qp.PauliX(wires=2), qp.PauliZ(wires=1))
        with pytest.raises(MatrixUndefinedError):
            sum_op.matrix()

    def test_sum_ops_multi_terms(self):
        """Test matrix is correct for a sum of more than two terms."""
        sum_op = Sum(qp.PauliX(wires=0), qp.Hadamard(wires=0), qp.PauliZ(wires=0))
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
        sum_op = Sum(qp.PauliX(wires=0), qp.Hadamard(wires=1), qp.PauliZ(wires=2))
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
        sum_op = Sum(qp.PauliZ(wires=2), qp.PauliX(wires=0), qp.Hadamard(wires=1))
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
        sum_op = Sum(qp.QFT(wires=wires), qp.GroverOperator(wires=wires), qp.PauliX(wires=0))
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
            qp.OrbitalRotation(4.56, wires=wires),
            qp.SingleExcitation(1.23, wires=[0, 1]),
            qp.Identity(3),
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
            qp.Hermitian(qnp.array([[0.0, 1.0], [1.0, 0.0]]), wires=0),
            qp.Projector(state=qnp.array([0, 1]), wires=wires),
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
        U_op = qp.QubitUnitary(U, wires=0)

        sum_op = Sum(U_op, qp.Identity(wires=1))
        mat = sum_op.matrix()

        true_mat = qnp.kron(U, qnp.eye(2)) + qnp.eye(4)
        assert np.allclose(mat, true_mat)

    def test_sum_hamiltonian(self):
        """Test that a hamiltonian object can be summed."""
        U = 0.5 * (qp.PauliX(wires=0) @ qp.PauliZ(wires=1))
        sum_op = Sum(U, qp.PauliX(wires=0))
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
            qp.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qp.RX(theta, wires=1),
            qp.Identity(wires=0),
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
            qp.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qp.RX(theta, wires=1),
            qp.Identity(wires=0),
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
            qp.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0),
            qp.RX(theta, wires=1),
            qp.Identity(wires=0),
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
        sum_op = qp.sum(op1(wires=0), op2(wires=1))
        true_mat = math.kron(mat1, np.eye(2)) + math.kron(np.eye(2), mat2)
        sum_mat = sum_op.sparse_matrix().todense()

        assert np.allclose(true_mat, sum_mat)

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix_format(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix accepts the format parameter."""
        from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix

        sum_op = qp.sum(op1(wires=0), op2(wires=1))
        true_mat = math.kron(mat1, np.eye(2)) + math.kron(np.eye(2), mat2)
        assert isinstance(sum_op.sparse_matrix(), csr_matrix)
        sum_op_csc = sum_op.sparse_matrix(format="csc")
        sum_op_lil = sum_op.sparse_matrix(format="lil")
        sum_op_coo = sum_op.sparse_matrix(format="coo")
        assert isinstance(sum_op_csc, csc_matrix)
        assert isinstance(sum_op_lil, lil_matrix)
        assert isinstance(sum_op_coo, coo_matrix)
        assert np.allclose(true_mat, sum_op_csc.todense())
        assert np.allclose(true_mat, sum_op_lil.todense())
        assert np.allclose(true_mat, sum_op_coo.todense())

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix_wire_order(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix of a Prod op is defined
        with wire order and correct."""
        true_mat = math.kron(mat2, np.eye(4)) + math.kron(np.eye(4), mat1)

        sum_op = qp.sum(op1(wires=2), op2(wires=0))
        sum_mat = sum_op.sparse_matrix(wire_order=[0, 1, 2]).todense()

        assert np.allclose(true_mat, sum_mat)

    def test_sparse_matrix_undefined_error(self):
        """Test that an error is raised when the sparse matrix method
        is undefined for any of the factors."""

        # pylint: disable=too-few-public-methods
        class DummyOp(qp.operation.Operation):
            num_wires = 1

            def sparse_matrix(self, wire_order=None):
                raise qp.operation.SparseMatrixUndefinedError

        sum_op = qp.sum(qp.PauliX(wires=0), DummyOp(wires=1))

        with pytest.raises(qp.operation.SparseMatrixUndefinedError):
            sum_op.sparse_matrix()


class TestProperties:
    """Test class properties."""

    def test_hash(self):
        """Test the hash property is independent of order."""
        op1 = Sum(qp.PauliX("a"), qp.PauliY("b"))
        op2 = Sum(qp.PauliY("b"), qp.PauliX("a"))
        assert op1.hash == op2.hash

        op3 = Sum(qp.PauliX("a"), qp.PauliY("b"), qp.PauliZ(-1))
        assert op3.hash != op1.hash

        op4 = Sum(qp.X("a"), qp.X("a"), qp.Y("b"))
        assert op4.hash != op1.hash

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, qp.sum])
    @pytest.mark.parametrize("ops_lst", ops)
    def test_is_hermitian(self, ops_lst, sum_method):
        """Test is_hermitian property updates correctly."""
        sum_op = sum_method(*ops_lst)
        true_hermitian_state = True

        for op in ops_lst:
            true_hermitian_state = true_hermitian_state and op.is_verified_hermitian

        assert sum_op.is_verified_hermitian == true_hermitian_state

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, qp.sum])
    @pytest.mark.parametrize("ops_lst", ops)
    def test_queue_category(self, ops_lst, sum_method):
        """Test queue_category property is always None."""  # currently not supporting queuing Sum
        sum_op = sum_method(*ops_lst)
        assert sum_op._queue_category is None  # pylint: disable=protected-access

    def test_eigvals_Identity_no_wires(self):
        """Test that eigenvalues can be computed for a sum containing identity with no wires."""

        op1 = qp.X(0) + 2 * qp.I()
        op2 = qp.X(0) + 2 * qp.I(0)
        assert qp.math.allclose(sorted(op1.eigvals()), sorted(op2.eigvals()))

    def test_eigendecomposition(self):
        """Test that the computed Eigenvalues and Eigenvectors are correct."""
        diag_sum_op = Sum(qp.PauliZ(wires=0), qp.Identity(wires=1))
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

    def test_eigendecomposition_repeat_operations(self):
        """Test that the eigendecomposition works with a repeated operation."""
        op1 = qp.X(0) + qp.X(0) + qp.X(0)
        op2 = qp.X(0) + qp.X(0)

        assert op1.diagonalizing_gates() == op2.diagonalizing_gates()
        assert qp.math.allclose(op1.eigvals(), np.array([-3.0, 3.0]))
        assert qp.math.allclose(op2.eigvals(), np.array([-2.0, 2.0]))

    op_pauli_reps = (
        (
            qp.sum(qp.PauliX(wires=0), qp.PauliY(wires=0), qp.PauliZ(wires=0)),
            qp.pauli.PauliSentence({_get_pw(0, "X"): 1, _get_pw(0, "Y"): 1, _get_pw(0, "Z"): 1}),
        ),
        (
            qp.sum(qp.PauliX(wires=0), qp.PauliX(wires=0), qp.PauliZ(wires=0)),
            qp.pauli.PauliSentence({_get_pw(0, "X"): 2, _get_pw(0, "Z"): 1}),
        ),
        (
            qp.sum(
                qp.PauliX(wires=0),
                qp.PauliY(wires=1),
                qp.PauliZ(wires="a"),
                qp.PauliZ(wires="a"),
            ),
            qp.pauli.PauliSentence({_get_pw(0, "X"): 1, _get_pw(1, "Y"): 1, _get_pw("a", "Z"): 2}),
        ),
    )

    @pytest.mark.parametrize("op, rep", op_pauli_reps)
    def test_pauli_rep(self, op, rep):
        """Test that the pauli rep gives the expected result."""
        assert op.pauli_rep == rep

    def test_pauli_rep_none(self):
        """Test that None is produced if any of the summands don't have a _pauli_rep."""
        op = qp.sum(qp.PauliX(wires=0), qp.RX(1.23, wires=1))
        assert op.pauli_rep is None

    op_pauli_reps_nested = (
        (
            qp.sum(
                qp.pow(
                    qp.sum(
                        qp.pow(qp.PauliZ(wires=0), z=3),
                        qp.pow(qp.PauliX(wires=1), z=2),
                        qp.pow(qp.PauliY(wires=2), z=1),
                    ),
                    z=3,
                ),
                qp.PauliY(wires=2),
            ),
            qp.pauli.PauliSentence(
                {
                    qp.pauli.PauliWord({0: "Z"}): 7,
                    qp.pauli.PauliWord({2: "Y"}): 8,
                    qp.pauli.PauliWord({0: "Z", 2: "Y"}): 6,
                    qp.pauli.PauliWord({}): 7,  # identity
                }
            ),
        ),  # sum + pow
        (
            qp.prod(
                qp.sum(
                    qp.prod(
                        qp.sum(qp.PauliX(wires=0), qp.PauliY(wires=1), qp.PauliZ(wires=2)),
                        qp.sum(qp.PauliZ(wires=0), qp.PauliZ(wires=1), qp.PauliZ(wires=2)),
                    ),
                    qp.Identity(wires=1),
                ),
                qp.PauliY(wires=3),
            ),
            qp.pauli.PauliSentence(
                {
                    qp.pauli.PauliWord({0: "Y", 3: "Y"}): -1j,
                    qp.pauli.PauliWord({0: "X", 1: "Z", 3: "Y"}): 1,
                    qp.pauli.PauliWord({0: "X", 2: "Z", 3: "Y"}): 1,
                    qp.pauli.PauliWord({0: "Z", 1: "Y", 3: "Y"}): 1,
                    qp.pauli.PauliWord({1: "X", 3: "Y"}): 1j,
                    qp.pauli.PauliWord({1: "Y", 2: "Z", 3: "Y"}): 1,
                    qp.pauli.PauliWord({0: "Z", 2: "Z", 3: "Y"}): 1,
                    qp.pauli.PauliWord({1: "Z", 2: "Z", 3: "Y"}): 1,
                    qp.pauli.PauliWord({3: "Y"}): 2,
                }
            ),
        ),  # sum + prod
        (
            qp.sum(
                qp.s_prod(
                    0.5,
                    qp.sum(
                        qp.s_prod(2j, qp.PauliX(wires=0)),
                        qp.s_prod(-4, qp.PauliY(wires=1)),
                    ),
                ),
                qp.s_prod(1.23 - 0.4j, qp.PauliZ(wires=2)),
            ),
            qp.pauli.PauliSentence(
                {_get_pw(0, "X"): 1.0j, _get_pw(1, "Y"): -2.0, _get_pw(2, "Z"): 1.23 - 0.4j}
            ),
        ),  # sum + s_prod
        (
            qp.prod(
                qp.s_prod(
                    -2,
                    qp.sum(
                        qp.s_prod(1j, qp.PauliX(wires=0)),
                        qp.PauliY(wires=1),
                    ),
                ),
                qp.pow(
                    qp.sum(
                        qp.s_prod(3, qp.PauliZ(wires=0)),
                        qp.PauliZ(wires=1),
                    ),
                    z=2,
                ),
            ),
            qp.pauli.PauliSentence(
                {
                    qp.pauli.PauliWord({0: "X"}): -20j,
                    qp.pauli.PauliWord({1: "Y"}): -20,
                    qp.pauli.PauliWord({0: "Y", 1: "Z"}): -12,
                    qp.pauli.PauliWord({0: "Z", 1: "X"}): -12j,
                }
            ),
        ),  # mixed
    )

    @pytest.mark.parametrize("op, rep", op_pauli_reps_nested)
    def test_pauli_rep_nested(self, op, rep):
        """Test that the pauli rep gives the expected result."""
        assert op.pauli_rep == rep

    def test_flatten_unflatten(self):
        """Test that we can flatten/unflatten a Sum correctly."""
        # pylint: disable=protected-access
        op = Sum(qp.PauliX(0), qp.PauliZ(0))
        new_op = Sum._unflatten(*op._flatten())

        assert op == new_op
        assert new_op.grouping_indices is None

    @pytest.mark.parametrize("grouping_type", ("qwc", "commuting", "anticommuting"))
    @pytest.mark.parametrize("method", ("lf", "rlf"))
    def test_flatten_unflatten_with_groups(self, grouping_type, method):
        """Test that we can flatten/unflatten a Sum correctly when grouping indices are available."""
        # pylint: disable=protected-access
        op = Sum(
            qp.PauliX(0),
            qp.s_prod(2.0, qp.PauliX(1)),
            qp.s_prod(3.0, qp.PauliZ(0)),
            grouping_type=grouping_type,
            method=method,
        )
        new_op = Sum._unflatten(*op._flatten())

        assert new_op == op
        assert new_op.grouping_indices == op.grouping_indices

        old_coeffs, old_ops = op.terms()
        new_coeffs, new_ops = new_op.terms()

        assert old_coeffs == new_coeffs
        assert old_ops == new_ops

    def test_grouping_indices_setter(self):
        """Test that grouping indices can be set"""
        H = qp.sum(*[qp.X("a"), qp.X("b"), qp.Y("b")])

        H.grouping_indices = [[0, 1], [2]]

        assert isinstance(H.grouping_indices, tuple)
        assert H.grouping_indices == ((0, 1), (2,))

    def test_grouping_indices_setter_error(self):
        """Test that setting incompatible indices raises an error"""
        H = qp.sum(*[qp.X("a"), qp.X("b"), qp.Y("b")])

        with pytest.raises(
            ValueError,
            match="The grouped index value needs to be a tuple of tuples of integers between 0",
        ):
            H.grouping_indices = [[0, 1, 3], [2]]

    def test_label(self):
        """Tests the label method of Sum when <=3 coefficients."""
        H = qp.ops.Sum(-0.8 * Z(0))
        assert H.label() == "𝓗"
        assert H.label(decimals=2) == "𝓗\n(-0.80)"

    def test_label_many_coefficients(self):
        """Tests the label method of Sum when >3 coefficients."""
        H = qp.ops.Sum(*(0.1 * qp.Z(0) for _ in range(5)))
        assert H.label() == "𝓗"
        assert H.label(decimals=2) == "𝓗"


class TestSimplify:
    """Test Sum simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        ops_to_sum = (
            qp.RZ(1.32, wires=0),
            qp.Identity(wires=0),
            qp.RX(1.9, wires=1),
            qp.PauliX(0),
        )
        s1 = qp.sum(ops_to_sum[0], ops_to_sum[1])
        s2 = qp.sum(s1, ops_to_sum[2])
        nested_sum = qp.sum(s2, ops_to_sum[3])
        class_sum_op = Sum(*ops_to_sum)
        assert nested_sum.arithmetic_depth == 3
        assert class_sum_op.arithmetic_depth == 1

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        sum_op = qp.RZ(1.32, wires=0) + qp.Identity(wires=0) + qp.RX(1.9, wires=1)
        final_op = Sum(qp.RZ(1.32, wires=0), qp.Identity(wires=0), qp.RX(1.9, wires=1))
        simplified_op = sum_op.simplify()
        qp.assert_equal(simplified_op, final_op)

    def test_simplify_grouping(self):
        """Test that the simplify method groups equal terms."""
        sum_op = qp.sum(
            qp.prod(qp.RX(1, 0), qp.PauliX(0), qp.PauliZ(1)),
            qp.prod(qp.RX(1.0, 0), qp.PauliX(0), qp.PauliZ(1)),
            qp.adjoint(qp.sum(qp.RY(1, 0), qp.PauliZ(1))),
            qp.adjoint(qp.RY(1, 0)),
            qp.adjoint(qp.PauliZ(1)),
        )
        mod_angle = -1 % (4 * np.pi)
        final_op = qp.sum(
            qp.s_prod(2, qp.prod(qp.RX(1, 0), qp.PauliX(0), qp.PauliZ(1))),
            qp.s_prod(2, qp.RY(mod_angle, 0)),
            qp.s_prod(2, qp.PauliZ(1)),
        )
        simplified_op = sum_op.simplify()
        qp.assert_equal(simplified_op, final_op)

    def test_simplify_grouping_delete_terms(self):
        """Test that the simplify method deletes all terms with coefficient equal to 0."""
        sum_op = qp.sum(
            qp.PauliX(0),
            qp.s_prod(0.3, qp.PauliX(0)),
            qp.s_prod(0.8, qp.PauliX(0)),
            qp.s_prod(0.2, qp.PauliX(0)),
            qp.s_prod(0.4, qp.PauliX(0)),
            qp.s_prod(0.3, qp.PauliX(0)),
            qp.s_prod(-3, qp.PauliX(0)),
        )
        simplified_op = sum_op.simplify()
        final_op = qp.s_prod(0, qp.Identity(0))
        qp.assert_equal(simplified_op, final_op)

    def test_simplify_grouping_with_tolerance(self):
        """Test the simplify method with a specific tolerance."""
        sum_op = qp.sum(-0.9 * qp.RX(1, 0), qp.RX(1, 0))
        final_op = qp.s_prod(0, qp.Identity(0))
        simplified_op = sum_op.simplify(cutoff=0.1)
        qp.assert_equal(simplified_op, final_op)

    @pytest.mark.jax
    def test_simplify_pauli_rep_jax(self):
        """Test that simplifying operators with a valid pauli representation works with jax interface."""
        import jax.numpy as jnp

        c1, c2, c3 = jnp.array(1.23), jnp.array(-1.23), jnp.array(0.5)

        op = qp.sum(
            qp.s_prod(c1, qp.PauliX(0)),
            qp.s_prod(c2, qp.PauliX(0)),
            qp.s_prod(c3, qp.PauliZ(1)),
        )
        result = qp.s_prod(c3, qp.PauliZ(1))
        simplified_op = op.simplify()

        qp.assert_equal(simplified_op, result)

    @pytest.mark.tf
    def test_simplify_pauli_rep_tf(self):
        """Test that simplifying operators with a valid pauli representation works with tf interface."""
        import tensorflow as tf

        c1, c2, c3 = tf.Variable(1.23), tf.Variable(-1.23), tf.Variable(0.5)

        op = qp.sum(
            qp.s_prod(c1, qp.PauliX(0)),
            qp.s_prod(c2, qp.PauliX(0)),
            qp.s_prod(c3, qp.PauliZ(1)),
        )
        result = qp.s_prod(c3, qp.PauliZ(1))
        simplified_op = op.simplify()
        qp.assert_equal(simplified_op, result)

    @pytest.mark.torch
    def test_simplify_pauli_rep_torch(self):
        """Test that simplifying operators with a valid pauli representation works with torch interface."""
        import torch

        c1, c2, c3 = torch.tensor(1.23), torch.tensor(-1.23), torch.tensor(0.5)

        op = qp.sum(
            qp.s_prod(c1, qp.PauliX(0)),
            qp.s_prod(c2, qp.PauliX(0)),
            qp.s_prod(c3, qp.PauliZ(1)),
        )
        result = qp.s_prod(c3, qp.PauliZ(1))
        simplified_op = op.simplify()

        qp.assert_equal(simplified_op, result)


class TestSortWires:
    """Tests for the wire sorting algorithm."""

    def test_sorting_operators_with_one_wire(self):
        """Test that the sorting algorithm works for operators that act on one wire."""
        op_list = [
            qp.X(3),
            qp.Z(2),
            qp.Y("a"),
            qp.RX(1, 5),
            qp.Y(0),
            qp.Y(1),
            qp.Z("c"),
            qp.X(5),
            qp.Z("ba"),
        ]
        sorted_list = Sum._sort(op_list)  # pylint: disable=protected-access
        final_list = [
            qp.Y(0),
            qp.Y(1),
            qp.Z(2),
            qp.X(3),
            qp.RX(1, 5),
            qp.X(5),
            qp.Y("a"),
            qp.Z("ba"),
            qp.Z("c"),
        ]

        for op1, op2 in zip(final_list, sorted_list):
            qp.assert_equal(op1, op2)

    def test_sorting_operators_with_multiple_wires(self):
        """Test that the sorting algorithm works for operators that act on multiple wires."""
        op_tuple = (
            qp.X(3),
            qp.X(5),
            qp.Toffoli([2, 3, 4]),
            qp.CNOT([2, 5]),
            qp.Z("ba"),
            qp.RX(1, 5),
            qp.Y(0),
            qp.CRX(1, [0, 2]),
            qp.Z(3),
            qp.Toffoli([1, "c", "ab"]),
            qp.CRY(1, [1, 2]),
            qp.X("d"),
        )
        sorted_list = Sum._sort(op_tuple)  # pylint: disable=protected-access
        final_list = [
            qp.Y(0),
            qp.CRX(1, [0, 2]),
            qp.CRY(1, [1, 2]),
            qp.Toffoli([1, "c", "ab"]),
            qp.CNOT([2, 5]),
            qp.Toffoli([2, 3, 4]),
            qp.X(3),
            qp.Z(3),
            qp.RX(1, 5),
            qp.X(5),
            qp.Z("ba"),
            qp.X("d"),
        ]

        for op1, op2 in zip(final_list, sorted_list):
            qp.assert_equal(op1, op2)

    def test_sorting_operators_with_wire_map(self):
        """Test that the sorting algorithm works using a wire map."""
        op_list = [
            qp.X("three"),
            qp.X(5),
            qp.Toffoli([2, "three", 4]),
            qp.CNOT([2, 5]),
            qp.RX(1, 5),
            qp.Y(0),
            qp.CRX(1, ["test", 2]),
            qp.Z("three"),
            qp.CRY(1, ["test", 2]),
        ]
        sorted_list = Sum._sort(  # pylint: disable=protected-access
            op_list, wire_map={0: 0, "test": 1, 2: 2, "three": 3, 4: 4, 5: 5}
        )
        final_list = [
            qp.Y(0),
            qp.CRX(1, ["test", 2]),
            qp.CRY(1, ["test", 2]),
            qp.CNOT([2, 5]),
            qp.Toffoli([2, "three", 4]),
            qp.X("three"),
            qp.Z("three"),
            qp.RX(1, 5),
            qp.X(5),
        ]

        for op1, op2 in zip(final_list, sorted_list):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert op1.data == op2.data

    def test_sort_wires_alphabetically(self):
        """Test that the summands are sorted alphabetically."""
        mixed_list = [
            qp.PauliY(1),
            qp.PauliZ(0),
            qp.PauliX(1),
            qp.PauliY(0),
            qp.PauliX(0),
            qp.PauliZ(1),
        ]
        final_list = [
            qp.PauliX(0),
            qp.PauliY(0),
            qp.PauliZ(0),
            qp.PauliX(1),
            qp.PauliY(1),
            qp.PauliZ(1),
        ]
        sorted_list = Sum._sort(mixed_list)  # pylint: disable=protected-access
        for op1, op2 in zip(final_list, sorted_list):
            qp.assert_equal(op1, op2)

    def test_sorting_operators_with_no_wires(self):
        """Test that sorting can occur when an operator acts on no wires."""

        op_list = [qp.GlobalPhase(0.5), qp.X(0), qp.Y(1), qp.I(), qp.CNOT((1, 2)), qp.I()]

        sorted_list = Sum._sort(op_list)  # pylint: disable=protected-access

        expected = [qp.GlobalPhase(0.5), qp.I(), qp.I(), qp.X(0), qp.Y(1), qp.CNOT((1, 2))]
        assert sorted_list == expected


class TestWrapperFunc:
    """Test wrapper function."""

    def test_op_sum_top_level(self):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        summands = (qp.PauliX(wires=1), qp.RX(1.23, wires=0), qp.CNOT(wires=[0, 1]))
        op_id = "sum_op"

        sum_func_op = qp.sum(*summands, id=op_id)
        sum_class_op = Sum(*summands, id=op_id)
        qp.assert_equal(sum_func_op, sum_class_op)

    def test_lazy_mode(self):
        """Test that by default, the operator is simply wrapped in `Sum`, even if a simplification exists."""
        op = qp.sum(qp.S(0), Sum(qp.S(1), qp.T(1)))

        assert isinstance(op, Sum)
        assert len(op) == 2

    def test_non_lazy_mode(self):
        """Test the lazy=False keyword."""
        op = qp.sum(qp.S(0), Sum(qp.S(1), qp.T(1)), lazy=False)

        assert isinstance(op, Sum)
        assert len(op) == 3

    def test_non_lazy_mode_queueing(self):
        """Test that if a simpification is accomplished, the metadata for the original op
        and the new simplified op is updated."""
        with qp.queuing.AnnotatedQueue() as q:
            sum1 = qp.sum(qp.S(1), qp.T(1))
            sum2 = qp.sum(qp.S(0), sum1, lazy=False)

        assert len(q) == 1
        assert q.queue[0] is sum2


class TestIntegration:
    """Integration tests for the Sum class."""

    def test_measurement_process_expval(self):
        """Test Sum class instance in expval measurement process."""
        dev = qp.device("default.qubit", wires=2)
        sum_op = Sum(qp.PauliX(0), qp.Hadamard(1))

        @qp.qnode(dev)
        def my_circ():
            qp.PauliX(0)
            return qp.expval(sum_op)

        exp_val = my_circ()
        true_exp_val = qnp.array(1 / qnp.sqrt(2))
        assert qnp.allclose(exp_val, true_exp_val)

    def test_measurement_process_var(self):
        """Test Sum class instance in var measurement process."""
        dev = qp.device("default.qubit", wires=2)
        sum_op = Sum(qp.PauliX(0), qp.Hadamard(1))

        @qp.qnode(dev)
        def my_circ():
            qp.PauliX(0)
            return qp.var(sum_op)

        var = my_circ()
        true_var = qnp.array(3 / 2)
        assert qnp.allclose(var, true_var)

    def test_measurement_process_probs(self):
        """Test Sum class instance in probs measurement process raises error."""
        dev = qp.device("default.qubit", wires=2)
        sum_op = Sum(qp.PauliX(0), qp.Hadamard(1))

        @qp.qnode(dev)
        def my_circ():
            qp.PauliX(0)
            return qp.probs(op=sum_op)

        x_probs = np.array([0.5, 0.5])
        h_probs = np.array([np.cos(-np.pi / 4 / 2) ** 2, np.sin(-np.pi / 4 / 2) ** 2])
        expected = np.tensordot(x_probs, h_probs, axes=0).flatten()
        out = my_circ()
        assert qp.math.allclose(out, expected)

    def test_measurement_process_sample(self):
        """Test Sum class instance in sample measurement process."""
        dev = qp.device("default.qubit", wires=2)
        sum_op = Sum(qp.PauliX(0), qp.PauliX(0))

        @qp.set_shots(20)
        @qp.qnode(dev)
        def my_circ():
            qp.prod(qp.Hadamard(0), qp.Hadamard(1))
            return qp.sample(op=sum_op)

        results = my_circ()

        assert len(results) == 20
        assert np.allclose(results, qnp.tensor([2.0] * 20, requires_grad=True))

    def test_measurement_process_count(self):
        """Test Sum class instance in counts measurement process."""
        dev = qp.device("default.qubit", wires=2)
        sum_op = Sum(qp.PauliX(0), qp.PauliX(0))

        @qp.set_shots(20)
        @qp.qnode(dev)
        def my_circ():
            qp.prod(qp.Hadamard(0), qp.Hadamard(1))
            return qp.counts(op=sum_op)

        results = my_circ()

        assert sum(results.values()) == 20
        assert np.allclose(
            2, list(results.keys())[0]
        )  # rounding errors due to float type of measurement outcome

    def test_differentiable_measurement_process(self):
        """Test that the gradient can be computed with a Sum op in the measurement process."""
        sum_op = Sum(qp.PauliX(0), qp.PauliZ(1))
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="best")
        def circuit(weights):
            qp.RX(weights[0], wires=0)
            qp.RY(weights[1], wires=1)
            qp.CNOT(wires=[0, 1])
            qp.RX(weights[2], wires=1)
            return qp.expval(sum_op)

        weights = qnp.array([0.1, 0.2, 0.3], requires_grad=True)
        grad = qp.grad(circuit)(weights)

        true_grad = qnp.array([-0.09347337, -0.18884787, -0.28818254])
        assert qnp.allclose(grad, true_grad)

    def test_params_can_be_considered_trainable(self):
        """Tests that the parameters of a Sum are considered trainable."""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface=None)
        def circuit():
            return qp.expval(
                Sum(qp.s_prod(1.1, qp.PauliX(0)), qp.s_prod(qnp.array(2.2), qp.PauliY(1)))
            )

        tape = qp.workflow.construct_tape(circuit)()
        assert tape.trainable_params == [1]


# pylint: disable=too-few-public-methods
class TestArithmetic:
    """Test arithmetic decomposition methods."""

    def test_adjoint(self):
        """Test the adjoint method for Sum Operators."""

        sum_op = Sum(qp.RX(1.23, wires=0), qp.Identity(wires=1))
        final_op = Sum(qp.adjoint(qp.RX(1.23, wires=0)), qp.adjoint(qp.Identity(wires=1)))
        adj_op = sum_op.adjoint()
        qp.assert_equal(adj_op, final_op)


class TestGrouping:
    """Test grouping functionality of Sum"""

    def test_set_on_initialization(self):
        """Test that grouping indices can be set on initialization."""

        op = qp.ops.Sum(qp.X(0), qp.Y(1), _grouping_indices=[[0, 1]])
        assert op.grouping_indices == [[0, 1]]
        op_ac = qp.ops.Sum(qp.X(0), qp.Y(1), grouping_type="anticommuting")
        assert op_ac.grouping_indices == ((0,), (1,))
        with pytest.raises(ValueError, match=r"cannot be specified at the same time."):
            qp.ops.Sum(
                qp.X(0), qp.Y(1), grouping_type="anticommuting", _grouping_indices=[[0, 1]]
            )

    def test_non_pauli_error(self):
        """Test that grouping non-Pauli observables is not supported."""
        op = Sum(qp.PauliX(0), Prod(qp.PauliZ(0), qp.PauliX(1)), qp.Hadamard(2))

        with pytest.raises(
            ValueError, match="Cannot compute grouping for Sums containing non-Pauli"
        ):
            op.compute_grouping()

    def test_grouping_is_correct_kwarg(self):
        """Basic test checking that grouping with a kwarg works as expected"""
        a = qp.PauliX(0)
        b = qp.PauliX(1)
        c = qp.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        # check with qp.dot
        op1 = qp.dot(coeffs, obs, grouping_type="qwc")
        assert op1.grouping_indices == ((0, 1), (2,))

        # check with qp.sum
        sprods = [qp.s_prod(c, o) for c, o in zip(coeffs, obs)]
        op2 = qp.sum(*sprods, grouping_type="qwc")
        assert op2.grouping_indices == ((0, 1), (2,))

        # check with Sum directly
        op3 = Sum(*sprods, grouping_type="qwc")
        assert op3.grouping_indices == ((0, 1), (2,))

    def test_grouping_is_correct_compute_grouping(self):
        """Basic test checking that grouping with compute_grouping works as expected"""
        a = qp.PauliX(0)
        b = qp.PauliX(1)
        c = qp.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        op = qp.dot(coeffs, obs, grouping_type=None)
        assert op.grouping_indices is None

        op.compute_grouping(grouping_type="qwc")
        assert op.grouping_indices == ((0, 1), (2,))

    def test_grouping_does_not_alter_queue(self):
        """Tests that grouping is invisible to the queue."""
        a = qp.PauliX(0)
        b = qp.PauliX(1)
        c = qp.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        with qp.queuing.AnnotatedQueue() as q:
            op = qp.dot(coeffs, obs, grouping_type="qwc")

        assert q.queue == [op]

    def test_grouping_for_non_groupable_sums(self):
        """Test that grouping is computed correctly, even if no observables commute"""
        a = qp.PauliX(0)
        b = qp.PauliY(0)
        c = qp.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        op = qp.dot(coeffs, obs, grouping_type="qwc")
        assert op.grouping_indices == ((0,), (1,), (2,))

    def test_grouping_method_can_be_set(self):
        """Tests that the grouping method can be controlled by kwargs.
        This is done by changing from default to 'lf' and checking the result."""
        a = qp.PauliX(0)
        b = qp.PauliX(1)
        c = qp.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        # compute grouping during construction with qp.dot
        op1 = qp.dot(coeffs, obs, grouping_type="qwc", method="lf")
        assert set(op1.grouping_indices) == {(0, 1), (2,)}

        # compute grouping during construction with qp.sum
        sprods = [qp.s_prod(c, o) for c, o in zip(coeffs, obs)]
        op2 = qp.sum(*sprods, grouping_type="qwc", method="lf")
        assert set(op2.grouping_indices) == {(0, 1), (2,)}

        # compute grouping during construction with Sum
        op3 = Sum(*sprods, grouping_type="qwc", method="lf")
        assert set(op3.grouping_indices) == {(0, 1), (2,)}

        # compute grouping separately
        op4 = qp.dot(coeffs, obs, grouping_type=None)
        op4.compute_grouping(method="lf")
        assert set(op4.grouping_indices) == {(0, 1), (2,)}

    @pytest.mark.parametrize(
        "grouping_type, grouping_indices",
        [("commuting", {(0, 1), (2,)}), ("anticommuting", {(1,), (0, 2)})],
    )
    def test_grouping_type_can_be_set(self, grouping_type, grouping_indices):
        """Tests that the grouping type can be controlled by kwargs.
        This is done by changing from default to 'commuting' or 'anticommuting'
        and checking the result."""
        a = qp.PauliX(0)
        b = qp.PauliX(1)
        c = qp.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        # compute grouping during construction with qp.dot
        op1 = qp.dot(coeffs, obs, grouping_type=grouping_type)
        assert set(op1.grouping_indices) == grouping_indices

        # compute grouping during construction with qp.sum
        sprods = [qp.s_prod(c, o) for c, o in zip(coeffs, obs)]
        op2 = qp.sum(*sprods, grouping_type=grouping_type)
        assert set(op2.grouping_indices) == grouping_indices

        # compute grouping during construction with Sum
        op3 = Sum(*sprods, grouping_type=grouping_type)
        assert set(op3.grouping_indices) == grouping_indices

        # compute grouping separately
        op4 = qp.dot(coeffs, obs, grouping_type=None)
        op4.compute_grouping(grouping_type=grouping_type)
        assert set(op4.grouping_indices) == grouping_indices

    @pytest.mark.parametrize("shots", [None, 1000])
    def test_grouping_integration(self, shots):
        """Test that grouping does not impact the results of a circuit."""
        dev = qp.device("default.qubit")

        @qp.set_shots(shots)
        @qp.qnode(dev)
        def qnode(grouping_type):
            H = qp.dot(
                [1.0, 2.0, 3.0],
                [qp.X(0), qp.prod(qp.X(0), qp.X(1)), qp.prod(qp.X(0), qp.X(1), qp.X(2))],
                grouping_type=grouping_type,
            )
            for i in range(3):
                qp.Hadamard(i)
            return qp.expval(H)

        assert np.allclose(qnode("qwc"), 6.0)
        assert np.allclose(qnode(None), 6.0)


class TestSupportsBroadcasting:
    """Test that the Sum operator supports broadcasting if its operands support broadcasting."""

    def test_batch_size_all_batched(self):
        """Test that the batch_size is correct when all operands are batched."""
        base = qp.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = Sum(base, base, base)
        assert op.batch_size == 3

    def test_batch_size_not_all_batched(self):
        """Test that the batch_size is correct when all operands are not batched."""
        base = qp.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = Sum(base, qp.RY(1, 0), qp.RZ(np.array([1, 2, 3]), wires=2))
        assert op.batch_size == 3

    def test_batch_size_None(self):
        """Test that the batch size is none if no operands have batching."""
        prod_op = Sum(qp.PauliX(0), qp.RX(1.0, wires=0))
        assert prod_op.batch_size is None

    def test_matrix_all_batched(self):
        """Test that Sum matrix has batching support when all operands are batched."""
        x = qp.numpy.array([0.1, 0.2, 0.3])
        y = qp.numpy.array([0.4, 0.5, 0.6])
        op = Sum(qp.RX(x, wires=0), qp.RY(y, wires=2), qp.PauliZ(1))
        mat = op.matrix()
        sum_list = [
            Sum(qp.RX(i, wires=0), qp.RY(j, wires=2), qp.PauliZ(1)) for i, j in zip(x, y)
        ]
        compare = qp.math.stack([s.matrix() for s in sum_list])
        assert qp.math.allclose(mat, compare)
        assert mat.shape == (3, 8, 8)

    def test_matrix_not_all_batched(self):
        """Test that Sum matrix has batching support when all operands are not batched."""
        x = qp.numpy.array([0.1, 0.2, 0.3])
        y = 0.5
        z = qp.numpy.array([0.4, 0.5, 0.6])
        op = Sum(qp.RX(x, wires=0), qp.RY(y, wires=2), qp.RZ(z, wires=1))
        mat = op.matrix()
        batched_y = [y for _ in x]
        sum_list = [
            Sum(qp.RX(i, wires=0), qp.RY(j, wires=2), qp.RZ(k, wires=1))
            for i, j, k in zip(x, batched_y, z)
        ]
        compare = qp.math.stack([s.matrix() for s in sum_list])
        assert qp.math.allclose(mat, compare)
        assert mat.shape == (3, 8, 8)
