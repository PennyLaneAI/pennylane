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

import pennylane as qml
import pennylane.numpy as qnp
from pennylane import X, Y, Z, math
from pennylane.exceptions import MatrixUndefinedError
from pennylane.operation import Operator
from pennylane.ops.op_math import Prod, Sum
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


def _get_pw(w, pauli_op):
    return qml.pauli.PauliWord({w: pauli_op})


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

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, qml.sum])
    @pytest.mark.parametrize("id", ("foo", "bar"))
    def test_init_sum_op(self, id, sum_method):
        """Test the initialization of a Sum operator."""
        sum_op = sum_method(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"), id=id)

        assert sum_op.wires == Wires((0, "a"))
        assert sum_op.num_wires == 2
        assert sum_op.name == "Sum"
        if sum_method.__name__ == sum.__name__:
            assert sum_op.id == id

        assert sum_op.data == (0.23,)
        assert sum_op.parameters == [0.23]
        assert sum_op.num_params == 1

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, qml.sum])
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

        assert sum_op.data == (0.23, 9.87)
        assert sum_op.parameters == [0.23, 9.87]
        assert sum_op.num_params == 2

    coeffs_ = [1.0, 1.0, 1.0, 3.0, 4.0, 4.0, 5.0]
    h6 = qml.sum(
        qml.s_prod(2.0, qml.prod(qml.PauliX(0), qml.PauliZ(10))),
        qml.s_prod(3.0, qml.prod(qml.PauliX(1), qml.PauliZ(11))),
    )
    ops_ = [
        qml.s_prod(1.0, qml.PauliX(0)),
        qml.s_prod(1.0, qml.prod(qml.PauliX(0), qml.PauliX(1))),
        qml.s_prod(2.0, qml.prod(qml.PauliZ(0), qml.PauliZ(1))),
        qml.s_prod(2.0, qml.PauliY(0)),
        qml.s_prod(2.0, qml.prod(qml.PauliY(0), qml.PauliY(1))),
        qml.s_prod(2.0, qml.prod(qml.s_prod(2.0, qml.PauliY(2)), qml.PauliY(3))),
        h6,
    ]
    Hamiltonian_with_Paulis = qml.dot(coeffs_, ops_)

    SUM_TERMS_OP_PAIRS_PAULI = (  # all operands have pauli representation
        (
            qml.sum(*(i * X(i) for i in range(1, 5))),
            [float(i) for i in range(1, 5)],
            [X(i) for i in range(1, 5)],
        ),
        (
            qml.sum(*(qml.s_prod(i, qml.prod(X(i), X(i + 1))) for i in range(1, 5))),
            [float(i) for i in range(1, 5)],
            [qml.prod(X(i), X(i + 1)) for i in range(1, 5)],
        ),
        (
            Hamiltonian_with_Paulis,
            [1.0, 1.0, 2.0, 6.0, 8.0, 16.0, 10.0, 15.0],
            [
                X(0),
                qml.prod(X(1), X(0)),
                qml.prod(Z(1), Z(0)),
                Y(0),
                qml.prod(Y(1), Y(0)),
                qml.prod(Y(3), Y(2)),
                qml.prod(Z(10), X(0)),
                qml.prod(Z(11), X(1)),
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
            qml.X(w0) @ qml.Y(w1) @ qml.X(w2) @ qml.Z(w3),
            qml.X(w0) @ qml.X(w1) @ qml.Y(w2) @ qml.Z(w3),
        ]

        H = qml.dot(coeffs, obs)
        _, H_ops = H.terms()

        assert all(o1.wires == o2.wires for o1, o2 in zip(obs, H_ops))
        assert H_ops[0] == qml.prod(qml.X(w0), qml.Y(w1), qml.X(w2), qml.Z(w3))
        assert H_ops[1] == qml.prod(qml.X(w0), qml.X(w1), qml.Y(w2), qml.Z(w3))

    coeffs_ = [1.0, 1.0, 1.0, 3.0, 4.0, 4.0, 5.0]
    h6 = qml.sum(
        qml.s_prod(2.0, qml.prod(qml.Hadamard(0), qml.PauliZ(10))),
        qml.s_prod(3.0, qml.prod(qml.PauliX(1), qml.PauliZ(11))),
    )
    ops_ = [
        qml.s_prod(1.0, qml.Hadamard(0)),
        qml.s_prod(1.0, qml.prod(qml.Hadamard(0), qml.PauliX(1))),
        qml.s_prod(2.0, qml.prod(qml.PauliZ(0), qml.PauliZ(1))),
        qml.s_prod(2.0, qml.PauliY(0)),
        qml.s_prod(2.0, qml.prod(qml.PauliY(0), qml.PauliY(1))),
        qml.s_prod(2.0, qml.prod(qml.s_prod(2.0, qml.PauliY(2)), qml.PauliY(3))),
        h6,
    ]
    Hamiltonian_mixed = qml.dot(coeffs_, ops_)

    SUM_TERMS_OP_PAIRS_MIXEDPAULI = (  # not all operands have pauli representation
        (
            qml.sum(*(i * qml.Hadamard(i) for i in range(1, 5))),
            [float(i) for i in range(1, 5)],
            [qml.Hadamard(i) for i in range(1, 5)],
        ),
        (
            qml.sum(qml.sum(*(i * qml.Hadamard(i) for i in range(1, 5))), 0.0 * qml.Identity(0)),
            [float(i) for i in range(1, 5)],
            [qml.Hadamard(i) for i in range(1, 5)],
        ),
        (
            qml.sum(qml.sum(*(i * qml.Hadamard(i) for i in range(1, 5))), qml.Identity(0)),
            [float(i) for i in range(1, 5)] + [1.0],
            [qml.Hadamard(i) for i in range(1, 5)] + [qml.Identity(0)],
        ),
        (
            qml.sum(*(qml.s_prod(i, qml.prod(X(i), qml.Hadamard(i + 1))) for i in range(1, 5))),
            [float(i) for i in range(1, 5)],
            [qml.prod(X(i), qml.Hadamard(i + 1)) for i in range(1, 5)],
        ),
        (
            Hamiltonian_mixed,
            [1.0, 1.0, 2.0, 6.0, 8.0, 16.0, 10.0, 15.0],
            [
                qml.Hadamard(0),
                qml.prod(X(1), qml.Hadamard(0)),
                qml.prod(Z(1), Z(0)),
                Y(0),
                qml.prod(Y(1), Y(0)),
                qml.prod(Y(3), Y(2)),
                qml.prod(Z(10), qml.Hadamard(0)),
                qml.prod(Z(11), X(1)),
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
        with qml.queuing.AnnotatedQueue() as q:
            qml.apply(op)
            _, _ = op.terms()

        assert q.queue == [op]

    def test_eigen_caching(self):
        """Test that the eigendecomposition is stored in cache."""
        diag_sum_op = Sum(qml.PauliZ(wires=0), qml.Identity(wires=1))
        eig_decomp = diag_sum_op.eigendecomposition

        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        eigs_cache = diag_sum_op._eigs[diag_sum_op.hash]  # pylint: disable=protected-access
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)

    SUM_REPR = (
        (qml.sum(), "Sum()"),
        (qml.sum(X(0), Y(1), Z(2)), "X(0) + Y(1) + Z(2)"),
        (X(0) + X(1) + X(2), "X(0) + X(1) + X(2)"),
        (0.5 * X(0) + 0.7 * X(1), "0.5 * X(0) + 0.7 * X(1)"),
        (0.5 * (X(0) @ X(1)) + 0.7 * X(1), "0.5 * (X(0) @ X(1)) + 0.7 * X(1)"),
        (
            0.5 * (X(0) @ (0.5 * X(1))) + 0.7 * X(1) + 0.8 * qml.CNOT((0, 1)),
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
        # qml.sum(*[0.5 * X(i) for i in range(10)]) # multiline output needs fixing of https://github.com/PennyLaneAI/pennylane/issues/5162 before working
    )

    @pytest.mark.parametrize("op", SUM_REPR_EVAL)
    def test_eval_sum(self, op):
        """Test that string representations of Sum can be evaluated and yield the same operator"""
        qml.assert_equal(eval(repr(op)), op)


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
            qml.Projector(state=qnp.array([0, 1]), wires=wires),
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
        sum_op = qml.sum(op1(wires=0), op2(wires=1))
        true_mat = math.kron(mat1, np.eye(2)) + math.kron(np.eye(2), mat2)
        sum_mat = sum_op.sparse_matrix().todense()

        assert np.allclose(true_mat, sum_mat)

    @pytest.mark.parametrize("op1, mat1", non_param_ops[:5])
    @pytest.mark.parametrize("op2, mat2", non_param_ops[:5])
    def test_sparse_matrix_format(self, op1, mat1, op2, mat2):
        """Test that the sparse matrix accepts the format parameter."""
        from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix

        sum_op = qml.sum(op1(wires=0), op2(wires=1))
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

        sum_op = qml.sum(op1(wires=2), op2(wires=0))
        sum_mat = sum_op.sparse_matrix(wire_order=[0, 1, 2]).todense()

        assert np.allclose(true_mat, sum_mat)

    def test_sparse_matrix_undefined_error(self):
        """Test that an error is raised when the sparse matrix method
        is undefined for any of the factors."""

        # pylint: disable=too-few-public-methods
        class DummyOp(qml.operation.Operation):
            num_wires = 1

            def sparse_matrix(self, wire_order=None):
                raise qml.operation.SparseMatrixUndefinedError

        sum_op = qml.sum(qml.PauliX(wires=0), DummyOp(wires=1))

        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            sum_op.sparse_matrix()


class TestProperties:
    """Test class properties."""

    def test_hash(self):
        """Test the hash property is independent of order."""
        op1 = Sum(qml.PauliX("a"), qml.PauliY("b"))
        op2 = Sum(qml.PauliY("b"), qml.PauliX("a"))
        assert op1.hash == op2.hash

        op3 = Sum(qml.PauliX("a"), qml.PauliY("b"), qml.PauliZ(-1))
        assert op3.hash != op1.hash

        op4 = Sum(qml.X("a"), qml.X("a"), qml.Y("b"))
        assert op4.hash != op1.hash

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, qml.sum])
    @pytest.mark.parametrize("ops_lst", ops)
    def test_is_hermitian(self, ops_lst, sum_method):
        """Test is_hermitian property updates correctly."""
        sum_op = sum_method(*ops_lst)
        true_hermitian_state = True

        for op in ops_lst:
            true_hermitian_state = true_hermitian_state and op.is_hermitian

        assert sum_op.is_hermitian == true_hermitian_state

    @pytest.mark.parametrize("sum_method", [sum_using_dunder_method, qml.sum])
    @pytest.mark.parametrize("ops_lst", ops)
    def test_queue_category(self, ops_lst, sum_method):
        """Test queue_category property is always None."""  # currently not supporting queuing Sum
        sum_op = sum_method(*ops_lst)
        assert sum_op._queue_category is None  # pylint: disable=protected-access

    def test_eigvals_Identity_no_wires(self):
        """Test that eigenvalues can be computed for a sum containing identity with no wires."""

        op1 = qml.X(0) + 2 * qml.I()
        op2 = qml.X(0) + 2 * qml.I(0)
        assert qml.math.allclose(sorted(op1.eigvals()), sorted(op2.eigvals()))

    def test_eigendecomposition(self):
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

    def test_eigendecomposition_repeat_operations(self):
        """Test that the eigendecomposition works with a repeated operation."""
        op1 = qml.X(0) + qml.X(0) + qml.X(0)
        op2 = qml.X(0) + qml.X(0)

        assert op1.diagonalizing_gates() == op2.diagonalizing_gates()
        assert qml.math.allclose(op1.eigvals(), np.array([-3.0, 3.0]))
        assert qml.math.allclose(op2.eigvals(), np.array([-2.0, 2.0]))

    op_pauli_reps = (
        (
            qml.sum(qml.PauliX(wires=0), qml.PauliY(wires=0), qml.PauliZ(wires=0)),
            qml.pauli.PauliSentence({_get_pw(0, "X"): 1, _get_pw(0, "Y"): 1, _get_pw(0, "Z"): 1}),
        ),
        (
            qml.sum(qml.PauliX(wires=0), qml.PauliX(wires=0), qml.PauliZ(wires=0)),
            qml.pauli.PauliSentence({_get_pw(0, "X"): 2, _get_pw(0, "Z"): 1}),
        ),
        (
            qml.sum(
                qml.PauliX(wires=0),
                qml.PauliY(wires=1),
                qml.PauliZ(wires="a"),
                qml.PauliZ(wires="a"),
            ),
            qml.pauli.PauliSentence({_get_pw(0, "X"): 1, _get_pw(1, "Y"): 1, _get_pw("a", "Z"): 2}),
        ),
    )

    @pytest.mark.parametrize("op, rep", op_pauli_reps)
    def test_pauli_rep(self, op, rep):
        """Test that the pauli rep gives the expected result."""
        assert op.pauli_rep == rep

    def test_pauli_rep_none(self):
        """Test that None is produced if any of the summands don't have a _pauli_rep."""
        op = qml.sum(qml.PauliX(wires=0), qml.RX(1.23, wires=1))
        assert op.pauli_rep is None

    op_pauli_reps_nested = (
        (
            qml.sum(
                qml.pow(
                    qml.sum(
                        qml.pow(qml.PauliZ(wires=0), z=3),
                        qml.pow(qml.PauliX(wires=1), z=2),
                        qml.pow(qml.PauliY(wires=2), z=1),
                    ),
                    z=3,
                ),
                qml.PauliY(wires=2),
            ),
            qml.pauli.PauliSentence(
                {
                    qml.pauli.PauliWord({0: "Z"}): 7,
                    qml.pauli.PauliWord({2: "Y"}): 8,
                    qml.pauli.PauliWord({0: "Z", 2: "Y"}): 6,
                    qml.pauli.PauliWord({}): 7,  # identity
                }
            ),
        ),  # sum + pow
        (
            qml.prod(
                qml.sum(
                    qml.prod(
                        qml.sum(qml.PauliX(wires=0), qml.PauliY(wires=1), qml.PauliZ(wires=2)),
                        qml.sum(qml.PauliZ(wires=0), qml.PauliZ(wires=1), qml.PauliZ(wires=2)),
                    ),
                    qml.Identity(wires=1),
                ),
                qml.PauliY(wires=3),
            ),
            qml.pauli.PauliSentence(
                {
                    qml.pauli.PauliWord({0: "Y", 3: "Y"}): -1j,
                    qml.pauli.PauliWord({0: "X", 1: "Z", 3: "Y"}): 1,
                    qml.pauli.PauliWord({0: "X", 2: "Z", 3: "Y"}): 1,
                    qml.pauli.PauliWord({0: "Z", 1: "Y", 3: "Y"}): 1,
                    qml.pauli.PauliWord({1: "X", 3: "Y"}): 1j,
                    qml.pauli.PauliWord({1: "Y", 2: "Z", 3: "Y"}): 1,
                    qml.pauli.PauliWord({0: "Z", 2: "Z", 3: "Y"}): 1,
                    qml.pauli.PauliWord({1: "Z", 2: "Z", 3: "Y"}): 1,
                    qml.pauli.PauliWord({3: "Y"}): 2,
                }
            ),
        ),  # sum + prod
        (
            qml.sum(
                qml.s_prod(
                    0.5,
                    qml.sum(
                        qml.s_prod(2j, qml.PauliX(wires=0)),
                        qml.s_prod(-4, qml.PauliY(wires=1)),
                    ),
                ),
                qml.s_prod(1.23 - 0.4j, qml.PauliZ(wires=2)),
            ),
            qml.pauli.PauliSentence(
                {_get_pw(0, "X"): 1.0j, _get_pw(1, "Y"): -2.0, _get_pw(2, "Z"): 1.23 - 0.4j}
            ),
        ),  # sum + s_prod
        (
            qml.prod(
                qml.s_prod(
                    -2,
                    qml.sum(
                        qml.s_prod(1j, qml.PauliX(wires=0)),
                        qml.PauliY(wires=1),
                    ),
                ),
                qml.pow(
                    qml.sum(
                        qml.s_prod(3, qml.PauliZ(wires=0)),
                        qml.PauliZ(wires=1),
                    ),
                    z=2,
                ),
            ),
            qml.pauli.PauliSentence(
                {
                    qml.pauli.PauliWord({0: "X"}): -20j,
                    qml.pauli.PauliWord({1: "Y"}): -20,
                    qml.pauli.PauliWord({0: "Y", 1: "Z"}): -12,
                    qml.pauli.PauliWord({0: "Z", 1: "X"}): -12j,
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
        op = Sum(qml.PauliX(0), qml.PauliZ(0))
        new_op = Sum._unflatten(*op._flatten())

        assert op == new_op
        assert new_op.grouping_indices is None

    @pytest.mark.parametrize("grouping_type", ("qwc", "commuting", "anticommuting"))
    @pytest.mark.parametrize("method", ("lf", "rlf"))
    def test_flatten_unflatten_with_groups(self, grouping_type, method):
        """Test that we can flatten/unflatten a Sum correctly when grouping indices are available."""
        # pylint: disable=protected-access
        op = Sum(
            qml.PauliX(0),
            qml.s_prod(2.0, qml.PauliX(1)),
            qml.s_prod(3.0, qml.PauliZ(0)),
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
        H = qml.sum(*[qml.X("a"), qml.X("b"), qml.Y("b")])

        H.grouping_indices = [[0, 1], [2]]

        assert isinstance(H.grouping_indices, tuple)
        assert H.grouping_indices == ((0, 1), (2,))

    def test_grouping_indices_setter_error(self):
        """Test that setting incompatible indices raises an error"""
        H = qml.sum(*[qml.X("a"), qml.X("b"), qml.Y("b")])

        with pytest.raises(
            ValueError,
            match="The grouped index value needs to be a tuple of tuples of integers between 0",
        ):
            H.grouping_indices = [[0, 1, 3], [2]]

    def test_label(self):
        """Tests the label method of Sum when <=3 coefficients."""
        H = qml.ops.Sum(-0.8 * Z(0))
        assert H.label() == "𝓗"
        assert H.label(decimals=2) == "𝓗\n(-0.80)"

    def test_label_many_coefficients(self):
        """Tests the label method of Sum when >3 coefficients."""
        H = qml.ops.Sum(*(0.1 * qml.Z(0) for _ in range(5)))
        assert H.label() == "𝓗"
        assert H.label(decimals=2) == "𝓗"


class TestSimplify:
    """Test Sum simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        ops_to_sum = (
            qml.RZ(1.32, wires=0),
            qml.Identity(wires=0),
            qml.RX(1.9, wires=1),
            qml.PauliX(0),
        )
        s1 = qml.sum(ops_to_sum[0], ops_to_sum[1])
        s2 = qml.sum(s1, ops_to_sum[2])
        nested_sum = qml.sum(s2, ops_to_sum[3])
        class_sum_op = Sum(*ops_to_sum)
        assert nested_sum.arithmetic_depth == 3
        assert class_sum_op.arithmetic_depth == 1

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        sum_op = qml.RZ(1.32, wires=0) + qml.Identity(wires=0) + qml.RX(1.9, wires=1)
        final_op = Sum(qml.RZ(1.32, wires=0), qml.Identity(wires=0), qml.RX(1.9, wires=1))
        simplified_op = sum_op.simplify()
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_grouping(self):
        """Test that the simplify method groups equal terms."""
        sum_op = qml.sum(
            qml.prod(qml.RX(1, 0), qml.PauliX(0), qml.PauliZ(1)),
            qml.prod(qml.RX(1.0, 0), qml.PauliX(0), qml.PauliZ(1)),
            qml.adjoint(qml.sum(qml.RY(1, 0), qml.PauliZ(1))),
            qml.adjoint(qml.RY(1, 0)),
            qml.adjoint(qml.PauliZ(1)),
        )
        mod_angle = -1 % (4 * np.pi)
        final_op = qml.sum(
            qml.s_prod(2, qml.prod(qml.RX(1, 0), qml.PauliX(0), qml.PauliZ(1))),
            qml.s_prod(2, qml.RY(mod_angle, 0)),
            qml.s_prod(2, qml.PauliZ(1)),
        )
        simplified_op = sum_op.simplify()
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_grouping_delete_terms(self):
        """Test that the simplify method deletes all terms with coefficient equal to 0."""
        sum_op = qml.sum(
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
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_grouping_with_tolerance(self):
        """Test the simplify method with a specific tolerance."""
        sum_op = qml.sum(-0.9 * qml.RX(1, 0), qml.RX(1, 0))
        final_op = qml.s_prod(0, qml.Identity(0))
        simplified_op = sum_op.simplify(cutoff=0.1)
        qml.assert_equal(simplified_op, final_op)

    @pytest.mark.jax
    def test_simplify_pauli_rep_jax(self):
        """Test that simplifying operators with a valid pauli representation works with jax interface."""
        import jax.numpy as jnp

        c1, c2, c3 = jnp.array(1.23), jnp.array(-1.23), jnp.array(0.5)

        op = qml.sum(
            qml.s_prod(c1, qml.PauliX(0)),
            qml.s_prod(c2, qml.PauliX(0)),
            qml.s_prod(c3, qml.PauliZ(1)),
        )
        result = qml.s_prod(c3, qml.PauliZ(1))
        simplified_op = op.simplify()

        qml.assert_equal(simplified_op, result)

    @pytest.mark.tf
    def test_simplify_pauli_rep_tf(self):
        """Test that simplifying operators with a valid pauli representation works with tf interface."""
        import tensorflow as tf

        c1, c2, c3 = tf.Variable(1.23), tf.Variable(-1.23), tf.Variable(0.5)

        op = qml.sum(
            qml.s_prod(c1, qml.PauliX(0)),
            qml.s_prod(c2, qml.PauliX(0)),
            qml.s_prod(c3, qml.PauliZ(1)),
        )
        result = qml.s_prod(c3, qml.PauliZ(1))
        simplified_op = op.simplify()
        qml.assert_equal(simplified_op, result)

    @pytest.mark.torch
    def test_simplify_pauli_rep_torch(self):
        """Test that simplifying operators with a valid pauli representation works with torch interface."""
        import torch

        c1, c2, c3 = torch.tensor(1.23), torch.tensor(-1.23), torch.tensor(0.5)

        op = qml.sum(
            qml.s_prod(c1, qml.PauliX(0)),
            qml.s_prod(c2, qml.PauliX(0)),
            qml.s_prod(c3, qml.PauliZ(1)),
        )
        result = qml.s_prod(c3, qml.PauliZ(1))
        simplified_op = op.simplify()

        qml.assert_equal(simplified_op, result)


class TestSortWires:
    """Tests for the wire sorting algorithm."""

    def test_sorting_operators_with_one_wire(self):
        """Test that the sorting algorithm works for operators that act on one wire."""
        op_list = [
            qml.X(3),
            qml.Z(2),
            qml.Y("a"),
            qml.RX(1, 5),
            qml.Y(0),
            qml.Y(1),
            qml.Z("c"),
            qml.X(5),
            qml.Z("ba"),
        ]
        sorted_list = Sum._sort(op_list)  # pylint: disable=protected-access
        final_list = [
            qml.Y(0),
            qml.Y(1),
            qml.Z(2),
            qml.X(3),
            qml.RX(1, 5),
            qml.X(5),
            qml.Y("a"),
            qml.Z("ba"),
            qml.Z("c"),
        ]

        for op1, op2 in zip(final_list, sorted_list):
            qml.assert_equal(op1, op2)

    def test_sorting_operators_with_multiple_wires(self):
        """Test that the sorting algorithm works for operators that act on multiple wires."""
        op_tuple = (
            qml.X(3),
            qml.X(5),
            qml.Toffoli([2, 3, 4]),
            qml.CNOT([2, 5]),
            qml.Z("ba"),
            qml.RX(1, 5),
            qml.Y(0),
            qml.CRX(1, [0, 2]),
            qml.Z(3),
            qml.Toffoli([1, "c", "ab"]),
            qml.CRY(1, [1, 2]),
            qml.X("d"),
        )
        sorted_list = Sum._sort(op_tuple)  # pylint: disable=protected-access
        final_list = [
            qml.Y(0),
            qml.CRX(1, [0, 2]),
            qml.CRY(1, [1, 2]),
            qml.Toffoli([1, "c", "ab"]),
            qml.CNOT([2, 5]),
            qml.Toffoli([2, 3, 4]),
            qml.X(3),
            qml.Z(3),
            qml.RX(1, 5),
            qml.X(5),
            qml.Z("ba"),
            qml.X("d"),
        ]

        for op1, op2 in zip(final_list, sorted_list):
            qml.assert_equal(op1, op2)

    def test_sorting_operators_with_wire_map(self):
        """Test that the sorting algorithm works using a wire map."""
        op_list = [
            qml.X("three"),
            qml.X(5),
            qml.Toffoli([2, "three", 4]),
            qml.CNOT([2, 5]),
            qml.RX(1, 5),
            qml.Y(0),
            qml.CRX(1, ["test", 2]),
            qml.Z("three"),
            qml.CRY(1, ["test", 2]),
        ]
        sorted_list = Sum._sort(  # pylint: disable=protected-access
            op_list, wire_map={0: 0, "test": 1, 2: 2, "three": 3, 4: 4, 5: 5}
        )
        final_list = [
            qml.Y(0),
            qml.CRX(1, ["test", 2]),
            qml.CRY(1, ["test", 2]),
            qml.CNOT([2, 5]),
            qml.Toffoli([2, "three", 4]),
            qml.X("three"),
            qml.Z("three"),
            qml.RX(1, 5),
            qml.X(5),
        ]

        for op1, op2 in zip(final_list, sorted_list):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert op1.data == op2.data

    def test_sort_wires_alphabetically(self):
        """Test that the summands are sorted alphabetically."""
        mixed_list = [
            qml.PauliY(1),
            qml.PauliZ(0),
            qml.PauliX(1),
            qml.PauliY(0),
            qml.PauliX(0),
            qml.PauliZ(1),
        ]
        final_list = [
            qml.PauliX(0),
            qml.PauliY(0),
            qml.PauliZ(0),
            qml.PauliX(1),
            qml.PauliY(1),
            qml.PauliZ(1),
        ]
        sorted_list = Sum._sort(mixed_list)  # pylint: disable=protected-access
        for op1, op2 in zip(final_list, sorted_list):
            qml.assert_equal(op1, op2)

    def test_sorting_operators_with_no_wires(self):
        """Test that sorting can occur when an operator acts on no wires."""

        op_list = [qml.GlobalPhase(0.5), qml.X(0), qml.Y(1), qml.I(), qml.CNOT((1, 2)), qml.I()]

        sorted_list = Sum._sort(op_list)  # pylint: disable=protected-access

        expected = [qml.GlobalPhase(0.5), qml.I(), qml.I(), qml.X(0), qml.Y(1), qml.CNOT((1, 2))]
        assert sorted_list == expected


class TestWrapperFunc:
    """Test wrapper function."""

    def test_op_sum_top_level(self):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        summands = (qml.PauliX(wires=1), qml.RX(1.23, wires=0), qml.CNOT(wires=[0, 1]))
        op_id = "sum_op"

        sum_func_op = qml.sum(*summands, id=op_id)
        sum_class_op = Sum(*summands, id=op_id)
        qml.assert_equal(sum_func_op, sum_class_op)

    def test_lazy_mode(self):
        """Test that by default, the operator is simply wrapped in `Sum`, even if a simplification exists."""
        op = qml.sum(qml.S(0), Sum(qml.S(1), qml.T(1)))

        assert isinstance(op, Sum)
        assert len(op) == 2

    def test_non_lazy_mode(self):
        """Test the lazy=False keyword."""
        op = qml.sum(qml.S(0), Sum(qml.S(1), qml.T(1)), lazy=False)

        assert isinstance(op, Sum)
        assert len(op) == 3

    def test_non_lazy_mode_queueing(self):
        """Test that if a simpification is accomplished, the metadata for the original op
        and the new simplified op is updated."""
        with qml.queuing.AnnotatedQueue() as q:
            sum1 = qml.sum(qml.S(1), qml.T(1))
            sum2 = qml.sum(qml.S(0), sum1, lazy=False)

        assert len(q) == 1
        assert q.queue[0] is sum2


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

    def test_measurement_process_probs(self):
        """Test Sum class instance in probs measurement process raises error."""
        dev = qml.device("default.qubit", wires=2)
        sum_op = Sum(qml.PauliX(0), qml.Hadamard(1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.probs(op=sum_op)

        x_probs = np.array([0.5, 0.5])
        h_probs = np.array([np.cos(-np.pi / 4 / 2) ** 2, np.sin(-np.pi / 4 / 2) ** 2])
        expected = np.tensordot(x_probs, h_probs, axes=0).flatten()
        out = my_circ()
        assert qml.math.allclose(out, expected)

    def test_measurement_process_sample(self):
        """Test Sum class instance in sample measurement process."""
        dev = qml.device("default.qubit", wires=2)
        sum_op = Sum(qml.PauliX(0), qml.PauliX(0))

        @qml.set_shots(20)
        @qml.qnode(dev)
        def my_circ():
            qml.prod(qml.Hadamard(0), qml.Hadamard(1))
            return qml.sample(op=sum_op)

        results = my_circ()

        assert len(results) == 20
        assert np.allclose(results, qnp.tensor([2.0] * 20, requires_grad=True))

    def test_measurement_process_count(self):
        """Test Sum class instance in counts measurement process."""
        dev = qml.device("default.qubit", wires=2)
        sum_op = Sum(qml.PauliX(0), qml.PauliX(0))

        @qml.set_shots(20)
        @qml.qnode(dev)
        def my_circ():
            qml.prod(qml.Hadamard(0), qml.Hadamard(1))
            return qml.counts(op=sum_op)

        results = my_circ()

        assert sum(results.values()) == 20
        assert np.allclose(
            2, list(results.keys())[0]
        )  # rounding errors due to float type of measurement outcome

    def test_differentiable_measurement_process(self):
        """Test that the gradient can be computed with a Sum op in the measurement process."""
        sum_op = Sum(qml.PauliX(0), qml.PauliZ(1))
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="best")
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

    def test_params_can_be_considered_trainable(self):
        """Tests that the parameters of a Sum are considered trainable."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=None)
        def circuit():
            return qml.expval(
                Sum(qml.s_prod(1.1, qml.PauliX(0)), qml.s_prod(qnp.array(2.2), qml.PauliY(1)))
            )

        tape = qml.workflow.construct_tape(circuit)()
        assert tape.trainable_params == [1]


# pylint: disable=too-few-public-methods
class TestArithmetic:
    """Test arithmetic decomposition methods."""

    def test_adjoint(self):
        """Test the adjoint method for Sum Operators."""

        sum_op = Sum(qml.RX(1.23, wires=0), qml.Identity(wires=1))
        final_op = Sum(qml.adjoint(qml.RX(1.23, wires=0)), qml.adjoint(qml.Identity(wires=1)))
        adj_op = sum_op.adjoint()
        qml.assert_equal(adj_op, final_op)


class TestGrouping:
    """Test grouping functionality of Sum"""

    def test_set_on_initialization(self):
        """Test that grouping indices can be set on initialization."""

        op = qml.ops.Sum(qml.X(0), qml.Y(1), _grouping_indices=[[0, 1]])
        assert op.grouping_indices == [[0, 1]]
        op_ac = qml.ops.Sum(qml.X(0), qml.Y(1), grouping_type="anticommuting")
        assert op_ac.grouping_indices == ((0,), (1,))
        with pytest.raises(ValueError, match=r"cannot be specified at the same time."):
            qml.ops.Sum(
                qml.X(0), qml.Y(1), grouping_type="anticommuting", _grouping_indices=[[0, 1]]
            )

    def test_non_pauli_error(self):
        """Test that grouping non-Pauli observables is not supported."""
        op = Sum(qml.PauliX(0), Prod(qml.PauliZ(0), qml.PauliX(1)), qml.Hadamard(2))

        with pytest.raises(
            ValueError, match="Cannot compute grouping for Sums containing non-Pauli"
        ):
            op.compute_grouping()

    def test_grouping_is_correct_kwarg(self):
        """Basic test checking that grouping with a kwarg works as expected"""
        a = qml.PauliX(0)
        b = qml.PauliX(1)
        c = qml.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        # check with qml.dot
        op1 = qml.dot(coeffs, obs, grouping_type="qwc")
        assert op1.grouping_indices == ((0, 1), (2,))

        # check with qml.sum
        sprods = [qml.s_prod(c, o) for c, o in zip(coeffs, obs)]
        op2 = qml.sum(*sprods, grouping_type="qwc")
        assert op2.grouping_indices == ((0, 1), (2,))

        # check with Sum directly
        op3 = Sum(*sprods, grouping_type="qwc")
        assert op3.grouping_indices == ((0, 1), (2,))

    def test_grouping_is_correct_compute_grouping(self):
        """Basic test checking that grouping with compute_grouping works as expected"""
        a = qml.PauliX(0)
        b = qml.PauliX(1)
        c = qml.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        op = qml.dot(coeffs, obs, grouping_type=None)
        assert op.grouping_indices is None

        op.compute_grouping(grouping_type="qwc")
        assert op.grouping_indices == ((0, 1), (2,))

    def test_grouping_does_not_alter_queue(self):
        """Tests that grouping is invisible to the queue."""
        a = qml.PauliX(0)
        b = qml.PauliX(1)
        c = qml.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        with qml.queuing.AnnotatedQueue() as q:
            op = qml.dot(coeffs, obs, grouping_type="qwc")

        assert q.queue == [op]

    def test_grouping_for_non_groupable_sums(self):
        """Test that grouping is computed correctly, even if no observables commute"""
        a = qml.PauliX(0)
        b = qml.PauliY(0)
        c = qml.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        op = qml.dot(coeffs, obs, grouping_type="qwc")
        assert op.grouping_indices == ((0,), (1,), (2,))

    def test_grouping_method_can_be_set(self):
        """Tests that the grouping method can be controlled by kwargs.
        This is done by changing from default to 'lf' and checking the result."""
        a = qml.PauliX(0)
        b = qml.PauliX(1)
        c = qml.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        # compute grouping during construction with qml.dot
        op1 = qml.dot(coeffs, obs, grouping_type="qwc", method="lf")
        assert set(op1.grouping_indices) == {(0, 1), (2,)}

        # compute grouping during construction with qml.sum
        sprods = [qml.s_prod(c, o) for c, o in zip(coeffs, obs)]
        op2 = qml.sum(*sprods, grouping_type="qwc", method="lf")
        assert set(op2.grouping_indices) == {(0, 1), (2,)}

        # compute grouping during construction with Sum
        op3 = Sum(*sprods, grouping_type="qwc", method="lf")
        assert set(op3.grouping_indices) == {(0, 1), (2,)}

        # compute grouping separately
        op4 = qml.dot(coeffs, obs, grouping_type=None)
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
        a = qml.PauliX(0)
        b = qml.PauliX(1)
        c = qml.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        # compute grouping during construction with qml.dot
        op1 = qml.dot(coeffs, obs, grouping_type=grouping_type)
        assert set(op1.grouping_indices) == grouping_indices

        # compute grouping during construction with qml.sum
        sprods = [qml.s_prod(c, o) for c, o in zip(coeffs, obs)]
        op2 = qml.sum(*sprods, grouping_type=grouping_type)
        assert set(op2.grouping_indices) == grouping_indices

        # compute grouping during construction with Sum
        op3 = Sum(*sprods, grouping_type=grouping_type)
        assert set(op3.grouping_indices) == grouping_indices

        # compute grouping separately
        op4 = qml.dot(coeffs, obs, grouping_type=None)
        op4.compute_grouping(grouping_type=grouping_type)
        assert set(op4.grouping_indices) == grouping_indices

    @pytest.mark.parametrize("shots", [None, 1000])
    def test_grouping_integration(self, shots):
        """Test that grouping does not impact the results of a circuit."""
        dev = qml.device("default.qubit")

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def qnode(grouping_type):
            H = qml.dot(
                [1.0, 2.0, 3.0],
                [qml.X(0), qml.prod(qml.X(0), qml.X(1)), qml.prod(qml.X(0), qml.X(1), qml.X(2))],
                grouping_type=grouping_type,
            )
            for i in range(3):
                qml.Hadamard(i)
            return qml.expval(H)

        assert np.allclose(qnode("qwc"), 6.0)
        assert np.allclose(qnode(None), 6.0)


class TestSupportsBroadcasting:
    """Test that the Sum operator supports broadcasting if its operands support broadcasting."""

    def test_batch_size_all_batched(self):
        """Test that the batch_size is correct when all operands are batched."""
        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = Sum(base, base, base)
        assert op.batch_size == 3

    def test_batch_size_not_all_batched(self):
        """Test that the batch_size is correct when all operands are not batched."""
        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = Sum(base, qml.RY(1, 0), qml.RZ(np.array([1, 2, 3]), wires=2))
        assert op.batch_size == 3

    def test_batch_size_None(self):
        """Test that the batch size is none if no operands have batching."""
        prod_op = Sum(qml.PauliX(0), qml.RX(1.0, wires=0))
        assert prod_op.batch_size is None

    def test_matrix_all_batched(self):
        """Test that Sum matrix has batching support when all operands are batched."""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        y = qml.numpy.array([0.4, 0.5, 0.6])
        op = Sum(qml.RX(x, wires=0), qml.RY(y, wires=2), qml.PauliZ(1))
        mat = op.matrix()
        sum_list = [
            Sum(qml.RX(i, wires=0), qml.RY(j, wires=2), qml.PauliZ(1)) for i, j in zip(x, y)
        ]
        compare = qml.math.stack([s.matrix() for s in sum_list])
        assert qml.math.allclose(mat, compare)
        assert mat.shape == (3, 8, 8)

    def test_matrix_not_all_batched(self):
        """Test that Sum matrix has batching support when all operands are not batched."""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        y = 0.5
        z = qml.numpy.array([0.4, 0.5, 0.6])
        op = Sum(qml.RX(x, wires=0), qml.RY(y, wires=2), qml.RZ(z, wires=1))
        mat = op.matrix()
        batched_y = [y for _ in x]
        sum_list = [
            Sum(qml.RX(i, wires=0), qml.RY(j, wires=2), qml.RZ(k, wires=1))
            for i, j, k in zip(x, batched_y, z)
        ]
        compare = qml.math.stack([s.matrix() for s in sum_list])
        assert qml.math.allclose(mat, compare)
        assert mat.shape == (3, 8, 8)
