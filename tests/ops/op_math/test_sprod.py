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

import pennylane as qml
import pennylane.numpy as qnp
from pennylane import QuantumFunctionError, math
from pennylane.operation import AnyWires, DecompositionUndefinedError, MatrixUndefinedError
from pennylane.ops.op_math.sprod import SProd, s_prod
from pennylane.wires import Wires

scalars = (1, 1.23, 0.0, 1 + 2j)  # int, float, zero, and complex cases accounted for

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
    (1.0, qml.PauliX(wires=0)),
    (0.0, qml.PauliZ(wires=0)),
    (1j, qml.Hadamard(wires=0)),
    (1.23, qml.CNOT(wires=[0, 1])),
    (4.56, qml.RX(1.23, wires=1)),
    (1.0 + 2.0j, qml.Identity(wires=0)),
    (10, qml.IsingXX(4.56, wires=[2, 3])),
    (0j, qml.Toffoli(wires=[1, 2, 3])),
    (42, qml.Rot(0.34, 1.0, 0, wires=0)),
)

ops_rep = (
    "1.0*(PauliX(wires=[0]))",
    "0.0*(PauliZ(wires=[0]))",
    "1j*(Hadamard(wires=[0]))",
    "1.23*(CNOT(wires=[0, 1]))",
    "4.56*(RX(1.23, wires=[1]))",
    "(1+2j)*(Identity(wires=[0]))",
    "10*(IsingXX(4.56, wires=[2, 3]))",
    "0j*(Toffoli(wires=[1, 2, 3]))",
    "42*(Rot(0.34, 1.0, 0, wires=[0]))",
)


class TestInitialization:
    """Test initialization of the SProd Class."""

    @pytest.mark.parametrize("test_id", ("foo", "bar"))
    def test_init_sprod_op(self, test_id):
        sprod_op = s_prod(3.14, qml.RX(0.23, wires="a"), do_queue=True, id=test_id)

        # no need to test if op.base == RX since this is covered in SymbolicOp tests
        assert sprod_op.scalar == 3.14
        assert sprod_op.wires == Wires(("a",))
        assert sprod_op.num_wires == 1
        assert sprod_op.name == "SProd"
        assert sprod_op.id == test_id
        assert sprod_op.queue_idx is None

        assert sprod_op.data == [[3.14], [0.23]]
        assert sprod_op.parameters == [[3.14], [0.23]]
        assert sprod_op.num_params == 2

    def test_parameters(self):
        sprod_op = s_prod(9.87, qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert sprod_op.parameters == [[9.87], [1.23, 4.0, 5.67]]

    def test_data(self):
        sprod_op = s_prod(9.87, qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert sprod_op.data == [[9.87], [1.23, 4.0, 5.67]]

    def test_data_setter(self):
        """Test the setter method for data"""
        scalar, angles = (9.87, (1.23, 4.0, 5.67))
        old_data = [[9.87], [1.23, 4.0, 5.67]]

        sprod_op = s_prod(scalar, qml.Rot(*angles, wires=1))
        assert sprod_op.data == old_data

        new_data = [[1.23], [0.0, -1.0, -2.0]]
        sprod_op.data = new_data
        assert sprod_op.data == new_data
        assert sprod_op.scalar == new_data[0][0]
        assert sprod_op.base.data == new_data[1]

    @pytest.mark.parametrize("scalar, op", ops)
    def test_terms(self, op, scalar):
        sprod_op = SProd(scalar, op)
        coeff, ops = sprod_op.terms()

        assert coeff == [scalar]
        for op1, op2 in zip(ops, [op]):
            assert qml.equal(op1, op2)

    def test_decomposition_raises_error(self):
        sprod_op = s_prod(3.14, qml.Identity(wires=1))

        with pytest.raises(DecompositionUndefinedError):
            sprod_op.decomposition()

    def test_diagonalizing_gates(self):
        """Test that the diagonalizing gates are correct."""
        diag_sprod_op = SProd(1.23, qml.PauliX(wires=0))
        diagonalizing_gates = diag_sprod_op.diagonalizing_gates()[0].matrix()
        true_diagonalizing_gates = (
            qml.PauliX(wires=0).diagonalizing_gates()[0].matrix()
        )  # scaling doesn't change diagonalizing gates

        assert np.allclose(diagonalizing_gates, true_diagonalizing_gates)


class TestMscMethods:
    """Test miscellaneous methods of the SProd class."""

    @pytest.mark.parametrize("op_scalar_tup, op_rep", tuple((i, j) for i, j in zip(ops, ops_rep)))
    def test_repr(self, op_scalar_tup, op_rep):
        """Test the repr dunder method."""
        scalar, op = op_scalar_tup
        sprod_op = SProd(scalar, op)
        assert op_rep == sprod_op.__repr__()

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
        assert (
            sprod_op.base.data is not copied_op.base.data
        )  # we want different object with same content


class TestMatrix:
    """Tests of the matrix of a SProd class."""

    @pytest.mark.parametrize("scalar", scalars)
    @pytest.mark.parametrize("op, mat", param_ops + non_param_ops)
    def test_various_ops(self, scalar, op, mat):
        """Test matrix method for a scalar product of parametric ops"""
        params = range(op.num_params)

        sprod_op = SProd(
            scalar, op(*params, wires=0 if op.num_wires is AnyWires else range(op.num_wires))
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
        sprod_op = SProd(scalar, qml.Toffoli(wires=[2, 0, 1]))
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
        (qml.QFT(wires=[0, 1, 2]), qml.QFT(wires=[0, 1, 2]).compute_matrix(3)),
        (
            qml.GroverOperator(wires=[0, 1, 2]),
            qml.GroverOperator(wires=[0, 1, 2]).compute_matrix(3, range(3)),
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
        sprod_op1 = SProd(1.23, qml.OrbitalRotation(4.56, wires=wires))
        sprod_op2 = SProd(3.45, qml.SingleExcitation(1.23, wires=[0, 1]))
        mat1 = sprod_op1.matrix()
        mat2 = sprod_op2.matrix()

        or_mat = 1.23 * gd.OrbitalRotation(4.56)
        se_mat = 3.45 * gd.SingleExcitation(1.23)

        assert np.allclose(mat1, or_mat)
        assert np.allclose(mat2, se_mat)

    def test_sprod_observables(self):
        """Test that observable objects can also be scaled with correct matrix representation."""
        wires = [0, 1]
        sprod_op1 = SProd(1.23, qml.Projector(basis_state=qnp.array([0, 1]), wires=wires))
        sprod_op2 = SProd(3.45, qml.Hermitian(qnp.array([[0.0, 1.0], [1.0, 0.0]]), wires=0))
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
        U_op = qml.QubitUnitary(U, wires=0)

        sprod_op = SProd(42, U_op)
        mat = sprod_op.matrix()

        true_mat = 42 * U
        assert np.allclose(mat, true_mat)

    sparse_ops = (
        qml.Identity(wires=0),
        qml.PauliX(wires=0),
        qml.PauliY(wires=0),
        qml.PauliZ(wires=0),
        qml.Hadamard(wires=0),
    )

    @pytest.mark.parametrize("op", sparse_ops)
    def test_sparse_matrix(self, op):
        """Test the sparse_matrix representation of scaled ops."""
        scalar = 1 + 2j
        sprod_op = SProd(scalar, op)
        sparse_matrix = sprod_op.sparse_matrix()
        sparse_matrix.sort_indices()

        expected_sparse_matrix = scalar * op.matrix()
        expected_sparse_matrix = csr_matrix(expected_sparse_matrix)
        expected_sparse_matrix.sort_indices()

        assert type(sparse_matrix) == type(expected_sparse_matrix)
        assert all(sparse_matrix.data == expected_sparse_matrix.data)
        assert all(sparse_matrix.indices == expected_sparse_matrix.indices)

    def test_sparse_matrix_sparse_hamiltonian(self):
        """Test the sparse_matrix representation of scaled ops."""
        scalar = 1.23
        op = qml.Hadamard(wires=0)
        sparse_ham = qml.SparseHamiltonian(csr_matrix(op.matrix()), wires=0)

        sprod_op = SProd(scalar, sparse_ham)
        sparse_matrix = sprod_op.sparse_matrix()

        expected_sparse_matrix = scalar * op.matrix()
        expected_sparse_matrix = csr_matrix(expected_sparse_matrix)

        assert np.allclose(sparse_matrix.todense(), expected_sparse_matrix.todense())

    def test_sprod_hamiltonian(self):
        """Test that a hamiltonian object can be scaled."""
        U = qml.Hamiltonian([0.5], [qml.PauliX(wires=0)])
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

        sprod_op = SProd(coeff, qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0))
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

        sprod_op = SProd(coeff, qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0))
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

        sprod_op = SProd(coeff, qml.Rot(rot_params[0], rot_params[1], rot_params[2], wires=0))
        mat = sprod_op.matrix()

        true_mat = 1.23 * gd.Rot3(raw_rot_params[0], raw_rot_params[1], raw_rot_params[2])
        true_mat = tf.Variable(true_mat, dtype=tf.complex128)

        assert isinstance(mat, tf.Tensor)
        assert mat.dtype == true_mat.dtype
        assert np.allclose(mat, true_mat)


class TestProperties:
    @pytest.mark.parametrize("op_scalar_tup", ops)
    def test_queue_catagory(self, op_scalar_tup):
        """Test queue_catagory property is always None."""  # currently not supporting queuing SProd
        scalar, op = op_scalar_tup
        sprod_op = SProd(scalar, op)
        assert sprod_op._queue_category is None

    def test_eigvals(self):
        """Test that the eigvals of the scalar product op are correct."""
        coeff, op = (1.0 + 2j, qml.PauliX(wires=0))
        sprod_op = SProd(coeff, op)
        sprod_op_eigvals = sprod_op.eigvals()

        x_eigvals = np.array([1.0, -1.0])
        true_eigvals = coeff * x_eigvals  # the true eigvals
        assert np.allclose(sprod_op_eigvals, true_eigvals)

    ops_are_hermitian = (
        (qml.PauliX(wires=0), 1.23 + 0.0j, True),  # Op is hermitian, scalar is real
        (qml.RX(1.23, wires=0), 1.0 + 0.0j, False),  # Op not hermitian
        (qml.PauliZ(wires=0), 2.0 + 1.0j, False),  # Scalar not real
    )

    @pytest.mark.parametrize("op, scalar, hermitian_status", ops_are_hermitian)
    def test_is_hermitian(self, op, scalar, hermitian_status):
        """Test that scalar product ops are correctly classified as hermitian or not."""
        sprod_op = s_prod(scalar, op)
        assert sprod_op.is_hermitian == hermitian_status

    @pytest.mark.tf
    def test_is_hermitian_tf(self):
        """Test that is_hermitian works when a tf type scalar is provided."""
        import tensorflow as tf

        coeffs = (tf.Variable(1.23), tf.Variable(1.23 + 1.2j))
        true_hermitian_states = (True, False)

        for scalar, hermitian_state in zip(coeffs, true_hermitian_states):
            op = s_prod(scalar, qml.PauliX(wires=0))
            assert op.is_hermitian == hermitian_state

    @pytest.mark.jax
    def test_is_hermitian_jax(self):
        """Test that is_hermitian works when a jax type scalar is provided."""
        import jax.numpy as jnp

        coeffs = (jnp.array(1.23), jnp.array(1.23 + 1.2j))
        true_hermitian_states = (True, False)

        for scalar, hermitian_state in zip(coeffs, true_hermitian_states):
            op = s_prod(scalar, qml.PauliX(wires=0))
            assert op.is_hermitian == hermitian_state

    @pytest.mark.torch
    def test_is_hermitian_torch(self):
        """Test that is_hermitian works when a torch type scalar is provided."""
        import torch

        coeffs = (torch.tensor(1.23), torch.tensor(1.23 + 1.2j))
        true_hermitian_states = (True, False)

        for scalar, hermitian_state in zip(coeffs, true_hermitian_states):
            op = s_prod(scalar, qml.PauliX(wires=0))
            assert op.is_hermitian == hermitian_state

    ops_labels = (
        (qml.PauliX(wires=0), 1.23, 2, "1.23*X"),
        (qml.RX(1.23, wires=0), 4.56, 1, "4.6*RX\n(1.2)"),
        (qml.RY(1.234, wires=0), 4.56, 3, "4.560*RY\n(1.234)"),
        (qml.Rot(1.0, 2.12, 3.1416, wires=0), 1, 2, "1.00*Rot\n(1.00,\n2.12,\n3.14)"),
    )

    @pytest.mark.parametrize("op, scalar, decimal, label", ops_labels)
    def test_label(self, op, scalar, decimal, label):
        """Testing that the label method works well with SProd objects."""
        sprod_op = s_prod(scalar, op)
        op_label = sprod_op.label(decimals=decimal)
        assert label == op_label

    def test_label_cache(self):
        """Test label method with cache keyword arg."""
        base = qml.QubitUnitary(np.eye(2), wires=0)
        op = s_prod(-1.2, base)

        cache = {"matrices": []}
        assert op.label(decimals=2, cache=cache) == "-1.20*U(M0)"
        assert len(cache["matrices"]) == 1


class TestSimplify:
    """Test SProd simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        sprod_op = s_prod(5, s_prod(3, s_prod(-1, qml.RZ(1.32, wires=0))))
        assert sprod_op.arithmetic_depth == 3

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        sprod_op = SProd(
            2, SProd(2, qml.RZ(1.32, wires=0)) + qml.Identity(wires=0) + qml.RX(1.9, wires=1)
        )
        final_op = qml.ops.Sum(
            SProd(4, qml.RZ(1.32, wires=0)),
            SProd(2, qml.Identity(wires=0)),
            SProd(2, qml.RX(1.9, wires=1)),
        )
        simplified_op = sprod_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, qml.ops.Sum)
        for s1, s2 in zip(final_op.operands, simplified_op.operands):
            assert isinstance(s2, SProd)
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_scalar_equal_to_1(self):
        """Test the simplify method when the scalar is 1."""
        sprod_op = s_prod(1, qml.PauliX(0))
        final_op = qml.PauliX(0)
        simplified_op = sprod_op.simplify()

        assert isinstance(simplified_op, qml.PauliX)
        assert simplified_op.name == final_op.name
        assert simplified_op.wires == final_op.wires
        assert simplified_op.data == final_op.data
        assert simplified_op.arithmetic_depth == final_op.arithmetic_depth

    def test_simplify_nested_sprod_scalar_equal_to_1(self):
        """Test the simplify method with nested SProd where the global scalar is 1."""
        sprod_op = s_prod(3, s_prod(1 / 3, qml.PauliX(0)))
        final_op = qml.PauliX(0)
        simplified_op = sprod_op.simplify()

        assert isinstance(simplified_op, qml.PauliX)
        assert simplified_op.name == final_op.name
        assert simplified_op.wires == final_op.wires
        assert simplified_op.data == final_op.data
        assert simplified_op.arithmetic_depth == final_op.arithmetic_depth

    def test_simplify_with_sum_operator(self):
        """Test the simplify method a scalar product of a Sum operator."""
        sprod_op = s_prod(0 - 3j, qml.op_sum(qml.PauliX(0), qml.PauliX(0)))
        final_op = s_prod(0 - 6j, qml.PauliX(0))
        simplified_op = sprod_op.simplify()

        assert isinstance(simplified_op, qml.ops.SProd)
        assert simplified_op.name == final_op.name
        assert repr(simplified_op) == repr(final_op)
        assert simplified_op.wires == final_op.wires
        assert simplified_op.data == final_op.data
        assert simplified_op.arithmetic_depth == final_op.arithmetic_depth


class TestWrapperFunc:
    @pytest.mark.parametrize("op_scalar_tup", ops)
    def test_s_prod_top_level(self, op_scalar_tup):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        coeff, op = op_scalar_tup

        op_id = "sprod_op"
        do_queue = False

        sprod_func_op = s_prod(coeff, op, id=op_id, do_queue=do_queue)
        sprod_class_op = SProd(coeff, op, id=op_id, do_queue=do_queue)

        assert sprod_class_op.scalar == sprod_func_op.scalar
        assert sprod_class_op.base == sprod_func_op.base
        assert np.allclose(sprod_class_op.matrix(), sprod_func_op.matrix())
        assert sprod_class_op.id == sprod_func_op.id
        assert sprod_class_op.wires == sprod_func_op.wires
        assert sprod_class_op.parameters == sprod_func_op.parameters


class TestIntegration:
    """Integration tests for SProd with a QNode."""

    def test_measurement_process_expval(self):
        """Test SProd class instance in expval measurement process."""
        dev = qml.device("default.qubit", wires=2)
        sprod_op = SProd(1.23, qml.Hadamard(1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.expval(sprod_op)

        exp_val = my_circ()
        true_exp_val = qnp.array(1.23 / qnp.sqrt(2))
        assert qnp.allclose(exp_val, true_exp_val)

    def test_measurement_process_var(self):
        """Test SProd class instance in var measurement process."""
        dev = qml.device("default.qubit", wires=2)
        sprod_op = SProd(1.23, qml.Hadamard(1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.var(sprod_op)

        var = my_circ()
        true_var = qnp.array(1.23**2 / 2)
        assert qnp.allclose(var, true_var)

    def test_measurement_process_probs(self):
        """Test SProd class instance in probs measurement process raises error."""  # currently can't support due to bug
        dev = qml.device("default.qubit", wires=2)
        sprod_op = SProd(1.23, qml.Hadamard(1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.probs(op=sprod_op)

        with pytest.raises(
            QuantumFunctionError,
            match="Symbolic Operations are not supported for " "rotating probabilities yet.",
        ):
            my_circ()

    def test_measurement_process_sample(self):
        """Test SProd class instance in sample measurement process."""
        dev = qml.device("default.qubit", wires=2, shots=20)
        sprod_op = SProd(1.23, qml.PauliX(1))

        @qml.qnode(dev)
        def my_circ():
            qml.Hadamard(1)
            return qml.sample(op=sprod_op)

        results = my_circ()

        assert len(results) == 20
        assert (results == 1.23).all()

    def test_measurement_process_count(self):
        """Test SProd class instance in counts measurement process."""
        dev = qml.device("default.qubit", wires=2, shots=20)
        sprod_op = SProd(1.23, qml.PauliX(1))

        @qml.qnode(dev)
        def my_circ():
            qml.Hadamard(1)
            return qml.counts(op=sprod_op)

        results = my_circ()

        assert sum(results.values()) == 20
        assert 1.23 in results
        assert -1.23 not in results

    def test_differentiable_scalar(self):
        """Test that the gradient can be computed of the scalar when a SProd op
        is used in the measurement process."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, grad_method="best")
        def circuit(scalar):
            qml.PauliX(wires=0)
            return qml.expval(SProd(scalar, qml.Hadamard(wires=0)))

        scalar = qnp.array([1.23], requires_grad=True)
        grad = qml.grad(circuit)(scalar)

        true_grad = -1 / qnp.sqrt(2)
        assert qnp.allclose(grad, true_grad)

    def test_differentiable_measurement_process(self):
        """Test that the gradient can be computed with a SProd op in the measurement process."""
        sprod_op = SProd(100, qml.Hadamard(0))
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, grad_method="best")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            return qml.expval(sprod_op)

        weights = qnp.array([0.1], requires_grad=True)
        grad = qml.grad(circuit)(weights)

        true_grad = 100 * -qnp.sqrt(2) * qnp.cos(weights[0] / 2) * qnp.sin(weights[0] / 2)
        assert qnp.allclose(grad, true_grad)

    def test_non_hermitian_op_in_measurement_process(self):
        """Test that non-hermitian ops in a measurement process will raise a warning."""
        wires = [0, 1]
        dev = qml.device("default.qubit", wires=wires)
        sprod_op = SProd(1.0 + 2.0j, qml.RX(1.23, wires=0))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.expval(sprod_op)

        with pytest.warns(UserWarning, match="SProd might not be hermitian."):
            my_circ()

    @pytest.mark.torch
    @pytest.mark.parametrize("diff_method", ("parameter-shift", "backprop"))
    def test_torch(self, diff_method):
        """Test that interface parameters can be unwrapped to numpy. This will occur when parameter-shift
        is requested for a given interface."""

        import torch

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(s):
            return qml.expval(qml.s_prod(s, qml.PauliZ(0)))

        res = circuit(torch.tensor(2))

        assert qml.math.allclose(res, 2)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ("parameter-shift", "backprop"))
    def test_torch(self, diff_method):
        """Test that interface parameters can be unwrapped to numpy. This will occur when parameter-shift
        is requested for a given interface."""

        from jax import numpy as jnp

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(s):
            return qml.expval(qml.s_prod(s, qml.PauliZ(0)))

        res = circuit(jnp.array(2))

        assert qml.math.allclose(res, 2)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ("parameter-shift", "backprop"))
    def test_tensorflow_qnode(self, diff_method):

        import tensorflow as tf

        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev, interface="tensorflow", diff_method=diff_method)
        def circuit(s):
            return qml.expval(qml.s_prod(s, qml.PauliZ(0)))

        res = circuit(tf.Variable(2))

        assert qml.math.allclose(res, 2)


class TestArithmetic:
    """Test arithmetic decomposition methods."""

    def test_pow(self):
        """Test the pow method for SProd Operators."""

        sprod_op = SProd(3, qml.RX(1.23, wires=0))
        final_op = SProd(scalar=3**2, base=qml.pow(base=qml.RX(1.23, wires=0), z=2))
        pow_op = sprod_op.pow(z=2)[0]

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(pow_op, SProd)
        assert pow_op.name == final_op.name
        assert pow_op.wires == final_op.wires
        assert pow_op.data == final_op.data
        assert pow_op.arithmetic_depth == final_op.arithmetic_depth

    def test_adjoint(self):
        """Test the adjoint method for Sprod Operators."""

        sprod_op = SProd(3j, qml.RX(1.23, wires=0))
        final_op = SProd(scalar=-3j, base=qml.adjoint(qml.RX(1.23, wires=0)))
        adj_op = sprod_op.adjoint()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(adj_op, SProd)
        assert adj_op.name == final_op.name
        assert adj_op.wires == final_op.wires
        assert adj_op.data == final_op.data
        assert adj_op.arithmetic_depth == final_op.arithmetic_depth
