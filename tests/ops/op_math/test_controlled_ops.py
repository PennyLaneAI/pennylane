# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for Operators inheriting from Controlled.
"""

import copy

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import unitary_group

from gate_data import (
    CY,
    CZ,
)

import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import AnyWires
from pennylane.ops.qubit.matrix_ops import QubitUnitary

# Non-parametrized operations and their matrix representation
NON_PARAMETRIZED_OPERATIONS = [
    (qml.CY, CY),
    (qml.CZ, CZ),
]

SPARSE_MATRIX_SUPPORTED_OPERATIONS = (
    (qml.CY(wires=[0, 1]), CY),
    (qml.CZ(wires=[0, 1]), CZ),
)

X = np.array([[0, 1], [1, 0]])
X_broadcasted = np.array([X] * 3)


class TestControlledQubitUnitary:
    """Tests specific to the ControlledQubitUnitary operation"""

    def test_initialization_from_matrix_and_operator(self):
        base_op = QubitUnitary(X, wires=1)

        op1 = qml.ControlledQubitUnitary(X, control_wires=[0, 2], wires=1)
        op2 = qml.ControlledQubitUnitary(base_op, control_wires=[0, 2])

        assert qml.equal(op1, op2)

    def test_no_control(self):
        """Test if ControlledQubitUnitary raises an error if control wires are not specified"""
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'control_wires'"
        ):
            qml.ControlledQubitUnitary(X, wires=2)

    def test_shared_control(self):
        """Test if ControlledQubitUnitary raises an error if control wires are shared with wires"""
        with pytest.raises(
            ValueError, match="The control wires must be different from the base operation wires"
        ):
            qml.ControlledQubitUnitary(X, control_wires=[0, 2], wires=2)

    def test_wires_specified_twice_warning(self):
        base = qml.QubitUnitary(X, 0)
        with pytest.warns(
            UserWarning,
            match="base operator already has wires; values specified through wires kwarg will be ignored.",
        ):
            qml.ControlledQubitUnitary(base, control_wires=[1, 2], wires=3)

    def test_wrong_shape(self):
        """Test if ControlledQubitUnitary raises a ValueError if a unitary of shape inconsistent
        with wires is provided"""
        with pytest.raises(ValueError, match=r"Input unitary must be of shape \(2, 2\)"):
            qml.ControlledQubitUnitary(np.eye(4), control_wires=[0, 1], wires=2).matrix()

    @pytest.mark.parametrize("target_wire", range(3))
    def test_toffoli(self, target_wire):
        """Test if ControlledQubitUnitary acts like a Toffoli gate when the input unitary is a
        single-qubit X. This test allows the target wire to be any of the three wires."""
        control_wires = list(range(3))
        del control_wires[target_wire]

        # pick some random unitaries (with a fixed seed) to make the circuit less trivial
        U1 = unitary_group.rvs(8, random_state=1)
        U2 = unitary_group.rvs(8, random_state=2)

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def f1():
            qml.QubitUnitary(U1, wires=range(3))
            qml.ControlledQubitUnitary(X, control_wires=control_wires, wires=target_wire)
            qml.QubitUnitary(U2, wires=range(3))
            return qml.state()

        @qml.qnode(dev)
        def f2():
            qml.QubitUnitary(U1, wires=range(3))
            qml.Toffoli(wires=control_wires + [target_wire])
            qml.QubitUnitary(U2, wires=range(3))
            return qml.state()

        state_1 = f1()
        state_2 = f2()

        assert np.allclose(state_1, state_2)

    @pytest.mark.parametrize("target_wire", range(3))
    def test_toffoli_broadcasted(self, target_wire):
        """Test if ControlledQubitUnitary acts like a Toffoli gate when the input unitary is a
        broadcasted single-qubit X. Allows the target wire to be any of the three wires."""
        control_wires = list(range(3))
        del control_wires[target_wire]

        # pick some random unitaries (with a fixed seed) to make the circuit less trivial
        U1 = unitary_group.rvs(8, random_state=1)
        U2 = unitary_group.rvs(8, random_state=2)

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def f1():
            qml.QubitUnitary(U1, wires=range(3))
            qml.ControlledQubitUnitary(
                X_broadcasted, control_wires=control_wires, wires=target_wire
            )
            qml.QubitUnitary(U2, wires=range(3))
            return qml.state()

        @qml.qnode(dev)
        def f2():
            qml.QubitUnitary(U1, wires=range(3))
            qml.Toffoli(wires=control_wires + [target_wire])
            qml.QubitUnitary(U2, wires=range(3))
            return qml.state()

        state_1 = f1()
        state_2 = f2()

        assert np.shape(state_1) == (3, 8)
        assert np.allclose(state_1, state_1[0])  # Check that all broadcasted results are equal
        assert np.allclose(state_1, state_2)

    def test_arbitrary_multiqubit(self):
        """Test if ControlledQubitUnitary applies correctly for a 2-qubit unitary with 2-qubit
        control, where the control and target wires are not ordered."""
        control_wires = [1, 3]
        target_wires = [2, 0]

        # pick some random unitaries (with a fixed seed) to make the circuit less trivial
        U1 = unitary_group.rvs(16, random_state=1)
        U2 = unitary_group.rvs(16, random_state=2)

        # the two-qubit unitary
        U = unitary_group.rvs(4, random_state=3)

        # the 4-qubit representation of the unitary if the control wires were [0, 1] and the target
        # wires were [2, 3]
        U_matrix = np.eye(16, dtype=np.complex128)
        U_matrix[12:16, 12:16] = U

        # We now need to swap wires so that the control wires are [1, 3] and the target wires are
        # [2, 0]
        swap = qml.SWAP.compute_matrix()

        # initial wire permutation: 0123
        # target wire permutation: 1302
        swap1 = np.kron(swap, np.eye(4))  # -> 1023
        swap2 = np.kron(np.eye(4), swap)  # -> 1032
        swap3 = np.kron(np.kron(np.eye(2), swap), np.eye(2))  # -> 1302
        swap4 = np.kron(np.eye(4), swap)  # -> 1320

        all_swap = swap4 @ swap3 @ swap2 @ swap1
        U_matrix = all_swap.T @ U_matrix @ all_swap

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def f1():
            qml.QubitUnitary(U1, wires=range(4))
            qml.ControlledQubitUnitary(U, control_wires=control_wires, wires=target_wires)
            qml.QubitUnitary(U2, wires=range(4))
            return qml.state()

        @qml.qnode(dev)
        def f2():
            qml.QubitUnitary(U1, wires=range(4))
            qml.QubitUnitary(U_matrix, wires=range(4))
            qml.QubitUnitary(U2, wires=range(4))
            return qml.state()

        state_1 = f1()
        state_2 = f2()

        assert np.allclose(state_1, state_2)

    def test_mismatched_control_value_length(self):
        """Test if ControlledQubitUnitary properly handles invalid mixed-polarity
        control values."""
        control_wires = [0, 1]
        wires = 2
        control_values = "011"
        target_wires = Wires(wires)

        with pytest.raises(
            ValueError, match="control_values should be the same length as control_wires"
        ):
            qml.ControlledQubitUnitary(
                X, control_wires=control_wires, wires=target_wires, control_values=control_values
            )

    @pytest.mark.parametrize(
        "control_wires,wires,control_values",
        [
            ([0], 1, [0]),
            ([0, 1], 2, [0, 0]),
            ([0, 1], 2, [1, 0]),
            ([0, 1], 2, [1, 1]),
            ([1, 0], 2, [0, 1]),
            ([0, 1], [2, 3], [1, 1]),
            ([0, 2], [3, 1], [1, 0]),
            ([1, 2, 0], [3, 4], [1, 0, 0]),
            ([1, 0, 2], [4, 3], [1, 1, 0]),
        ],
    )
    def test_mixed_polarity_controls(self, control_wires, wires, control_values):
        """Test if ControlledQubitUnitary properly applies mixed-polarity
        control values."""
        target_wires = Wires(wires)

        dev = qml.device("default.qubit", wires=len(control_wires + target_wires))

        # Pick a random unitary
        U = unitary_group.rvs(2 ** len(target_wires), random_state=1967)

        # Pick random starting state for the control and target qubits
        control_state_weights = np.random.normal(size=2 ** (len(control_wires) + 1) - 2)
        target_state_weights = np.random.normal(size=2 ** (len(target_wires) + 1) - 2)

        @qml.qnode(dev)
        def circuit_mixed_polarity():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            qml.ControlledQubitUnitary(
                U, control_wires=control_wires, wires=target_wires, control_values=control_values
            )
            return qml.state()

        # The result of applying the mixed-polarity gate should be the same as
        # if we conjugated the specified control wires with Pauli X and applied the
        # "regular" ControlledQubitUnitary in between.

        x_locations = [x for x in range(len(control_values)) if control_values[x] == 0]

        @qml.qnode(dev)
        def circuit_pauli_x():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            qml.ControlledQubitUnitary(U, control_wires=control_wires, wires=wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            return qml.state()

        mixed_polarity_state = circuit_mixed_polarity()
        pauli_x_state = circuit_pauli_x()

        assert np.allclose(mixed_polarity_state, pauli_x_state)

    def test_same_as_Toffoli(self):
        """Test if ControlledQubitUnitary returns the correct matrix for a control-control-X
        (Toffoli) gate"""
        mat = qml.ControlledQubitUnitary(X, control_wires=[0, 1], wires=2).matrix()
        mat2 = qml.Toffoli(wires=[0, 1, 2]).matrix()
        assert np.allclose(mat, mat2)

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""
        U = np.array([[0.94877869, 0.31594146], [-0.31594146, 0.94877869]])
        res_dynamic = qml.ControlledQubitUnitary(U, control_wires=[1], wires=0).matrix()
        expected = np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.94877869 + 0.0j, 0.31594146 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, -0.31594146 + 0.0j, 0.94877869 + 0.0j],
            ]
        )
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_matrix_representation_broadcasted(self, tol):
        """Test that the matrix representation is defined correctly"""
        U = np.array(
            [
                [[0.94877869, 0.31594146], [-0.31594146, 0.94877869]],
                [[0.4125124, -0.91095199], [0.91095199, 0.4125124]],
                [[0.31594146, 0.94877869j], [0.94877869j, 0.31594146]],
            ]
        )

        res_dynamic = qml.ControlledQubitUnitary(U, control_wires=[1], wires=0).matrix()
        expected = np.array(
            [
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.94877869 + 0.0j, 0.31594146 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, -0.31594146 + 0.0j, 0.94877869 + 0.0j],
                ],
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.4125124 + 0.0j, -0.91095199 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.91095199 + 0.0j, 0.4125124 + 0.0j],
                ],
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.31594146 + 0.0j, 0.0 + 0.94877869j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.94877869j, 0.31594146 + 0.0j],
                ],
            ]
        )
        assert np.allclose(res_dynamic, expected, atol=tol)

    @pytest.mark.parametrize("n", (2, -1, -2))
    def test_pow(self, n):
        """Tests the metadata and unitary for a ControlledQubitUnitary raised to a power."""
        U1 = np.array(
            [
                [0.73708696 + 0.61324932j, 0.27034258 + 0.08685028j],
                [-0.24979544 - 0.1350197j, 0.95278437 + 0.1075819j],
            ]
        )

        op = qml.ControlledQubitUnitary(
            U1, control_wires=("b", "c"), wires="a", control_values="01"
        )

        pow_ops = op.pow(n)
        assert len(pow_ops) == 1

        assert pow_ops[0].target_wires == op.target_wires
        assert pow_ops[0].control_wires == op.control_wires
        assert pow_ops[0].control_values == op.control_values

        op_mat_to_pow = qml.math.linalg.matrix_power(op.data[0], n)
        assert qml.math.allclose(pow_ops[0].data[0], op_mat_to_pow)

    @pytest.mark.parametrize("n", (2, -1, -2))
    def test_pow_broadcasted(self, n):
        """Tests the metadata and unitary for a broadcasted
        ControlledQubitUnitary raised to a power."""
        U1 = np.tensordot(
            np.array([1j, -1.0, 1j]),
            np.array(
                [
                    [0.73708696 + 0.61324932j, 0.27034258 + 0.08685028j],
                    [-0.24979544 - 0.1350197j, 0.95278437 + 0.1075819j],
                ]
            ),
            axes=0,
        )

        op = qml.ControlledQubitUnitary(U1, control_wires=("b", "c"), wires="a")

        pow_ops = op.pow(n)
        assert len(pow_ops) == 1

        assert pow_ops[0].target_wires == op.target_wires
        assert pow_ops[0].control_wires == op.control_wires

        op_mat_to_pow = qml.math.linalg.matrix_power(op.data[0], n)
        assert qml.math.allclose(pow_ops[0].data[0], op_mat_to_pow)

    def test_noninteger_pow(self):
        """Test that a ControlledQubitUnitary raised to a non-integer power raises an error."""
        U1 = np.array(
            [
                [0.73708696 + 0.61324932j, 0.27034258 + 0.08685028j],
                [-0.24979544 - 0.1350197j, 0.95278437 + 0.1075819j],
            ]
        )

        op = qml.ControlledQubitUnitary(U1, control_wires=("b", "c"), wires="a")

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(0.12)

    def test_noninteger_pow_broadcasted(self):
        """Test that a ControlledQubitUnitary raised to a non-integer power raises an error."""
        U1 = np.tensordot(
            np.array([1j, -1.0, 1j]),
            np.array(
                [
                    [0.73708696 + 0.61324932j, 0.27034258 + 0.08685028j],
                    [-0.24979544 - 0.1350197j, 0.95278437 + 0.1075819j],
                ]
            ),
            axes=0,
        )

        op = qml.ControlledQubitUnitary(U1, control_wires=("b", "c"), wires="a")

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(0.12)

    def test_controlled(self):
        """Test the _controlled method for ControlledQubitUnitary."""

        U = qml.PauliX(0).compute_matrix()

        original = qml.ControlledQubitUnitary(U, control_wires=(0, 1), wires=4, control_values="01")
        expected = qml.ControlledQubitUnitary(
            U, control_wires=(0, 1, "a"), wires=4, control_values="011"
        )

        out = original._controlled("a")  # pylint: disable=protected-access
        assert qml.equal(out, expected)

    def test_unitary_check(self):
        unitary = np.array([[0.94877869j, 0.31594146], [-0.31594146, 0.94877869j]])
        not_unitary = np.array([[0.94877869j, 0.31594146], [-5, 0.94877869j]])

        qml.ControlledQubitUnitary(unitary, control_wires=[0, 2], wires=1, unitary_check=True)

        with pytest.warns(UserWarning, match="may not be unitary"):
            qml.ControlledQubitUnitary(
                not_unitary, control_wires=[0, 2], wires=1, unitary_check=True
            )


class TestOperations:
    @pytest.mark.parametrize("op_cls, _", NON_PARAMETRIZED_OPERATIONS)
    def test_nonparametrized_op_copy(self, op_cls, _, tol):
        """Tests that copied nonparametrized ops function as expected"""
        op = op_cls(wires=0 if op_cls.num_wires is AnyWires else range(op_cls.num_wires))
        copied_op = copy.copy(op)
        assert qml.equal(copied_op, op, atol=tol, rtol=0)
        assert copied_op is not op
        assert qml.equal(copied_op, op, atol=tol, rtol=0)

    @pytest.mark.parametrize("ops, mat", NON_PARAMETRIZED_OPERATIONS)
    def test_matrices(self, ops, mat, tol):
        """Test matrices of non-parametrized operations are correct"""
        op = ops(wires=0 if ops.num_wires is AnyWires else range(ops.num_wires))
        res_static = op.compute_matrix()
        res_dynamic = op.matrix()
        assert np.allclose(res_static, mat, atol=tol, rtol=0)
        assert np.allclose(res_dynamic, mat, atol=tol, rtol=0)


class TestDecompositions:  # pylint: disable=too-few-public-methods
    def test_CY_decomposition(self, tol):
        """Tests that the decomposition of the CY gate is correct"""
        op = qml.CY(wires=[0, 1])
        gate1, gate2 = op.decomposition()
        decomposed_matrix = qml.matrix(op.decomposition)()

        assert qml.equal(gate1, qml.CRY(np.pi, wires=[0, 1]))
        assert qml.equal(gate2, qml.S(0))
        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_CZ_decomposition(self, tol):
        """Tests that the decomposition of the CZ gate is correct"""
        op = qml.CZ(wires=[0, 1])
        res = op.decomposition()

        assert op.has_decomposition
        assert len(res) == 1
        assert qml.equal(qml.ctrl(qml.PhaseShift(np.pi, wires=1), 0), res[0], atol=tol, rtol=0)

        decomposed_matrix = res[0].matrix()
        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)


class TestEigenval:  # pylint: disable=too-few-public-methods
    def test_CZ_eigenval(self):
        """Tests that the CZ eigenvalue matches the numpy eigenvalues of the CZ matrix"""
        op = qml.CZ(wires=[0, 1])
        exp = np.linalg.eigvals(op.matrix())
        res = op.eigvals()
        assert np.allclose(res, exp)


period_two_ops = (
    qml.CY(wires=(0, 1)),
    qml.CZ(wires=(0, 1)),
)


class TestPowMethod:
    @pytest.mark.parametrize("op", period_two_ops)
    @pytest.mark.parametrize("n", (1, 5, -1, -5))
    def test_period_two_pow_odd(self, op, n):
        """Test that ops with a period of 2 raised to an odd power are the same as the original op."""
        assert qml.equal(op.pow(n)[0], op)
        assert np.allclose(op.pow(n)[0].matrix(), op.matrix())
        assert op.pow(n)[0].name == op.name

    @pytest.mark.parametrize("op", period_two_ops)
    @pytest.mark.parametrize("n", (2, 6, 0, -2))
    def test_period_two_pow_even(self, op, n):
        """Test that ops with a period of 2 raised to an even power are empty lists."""
        assert len(op.pow(n)) == 0

    @pytest.mark.parametrize("op", period_two_ops)
    def test_period_two_noninteger_power(self, op):
        """Test that ops with a period of 2 raised to a non-integer power raise an error."""
        if isinstance(op, (qml.PauliZ, qml.CZ)):
            pytest.skip("PauliZ can be raised to any power.")
        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(1.234)

        if op.__class__ is qml.CZ:
            pytest.skip("CZ can be raised to any power.")
        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(1.234)

    @pytest.mark.parametrize("n", (0.12, -3.462, 3.693))
    def test_cz_general_power(self, n):
        """Check that CZ raised to an non-integer power that's not the square root
        results in a controlled PhaseShift."""
        op_pow = qml.CZ(wires=[0, 1]).pow(n)

        assert len(op_pow) == 1
        assert isinstance(op_pow[0], qml.ops.ControlledOp)
        assert isinstance(op_pow[0].base, qml.PhaseShift)
        assert qml.math.allclose(op_pow[0].data[0], np.pi * (n % 2))


class TestControlledMethod:  # pylint: disable=too-few-public-methods
    """Tests for the _controlled method of non-parametric operations."""

    def test_CZ(self):
        """Test the PauliZ _controlled method."""
        out = qml.CZ(wires=[0, 1])._controlled("a")  # pylint: disable=protected-access
        assert qml.equal(out, qml.CCZ(("a", 0, 1)))


class TestSparseMatrix:  # pylint: disable=too-few-public-methods
    @pytest.mark.parametrize("op, mat", SPARSE_MATRIX_SUPPORTED_OPERATIONS)
    def test_sparse_matrix(self, op, mat):
        """Tests the sparse matrix method for operations which support it."""
        expected_sparse_mat = csr_matrix(mat)
        sparse_mat = op.sparse_matrix()

        assert isinstance(sparse_mat, csr_matrix)
        assert isinstance(expected_sparse_mat, csr_matrix)
        assert all(sparse_mat.data == expected_sparse_mat.data)
        assert all(sparse_mat.indices == expected_sparse_mat.indices)


label_data = [
    (qml.CY(wires=(0, 1)), "Y"),
    (qml.CZ(wires=(0, 1)), "Z"),
]


@pytest.mark.parametrize("op, label", label_data)
def test_label_method(op, label):
    """Tests that the label method gives the expected result."""
    assert op.label() == label
    assert op.label(decimals=2) == label


control_data = [
    (qml.CY(wires=(0, 1)), Wires(0)),
    (qml.CZ(wires=(0, 1)), Wires(0)),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test ``control_wires`` attribute for non-parametrized operations."""

    assert op.control_wires == control_wires


involution_ops = [  # ops who are their own inverses
    qml.CY((0, 1)),
    qml.CZ(wires=(0, 1)),
]


@pytest.mark.parametrize("op", involution_ops)
def test_adjoint_method(op):
    """Tests the adjoint method for operations that are their own adjoint."""
    adj_op = copy.copy(op)
    for _ in range(4):
        adj_op = adj_op.adjoint()

        assert qml.equal(adj_op, op)


@pytest.mark.parametrize("op_cls, _", NON_PARAMETRIZED_OPERATIONS)
def test_map_wires(op_cls, _):
    """Test that we can get and set private wires in all operations."""

    op = op_cls(wires=[0, 1])
    assert op.wires == Wires((0, 1))

    op = op.map_wires(wire_map={0: "a", 1: "b"})
    assert op.base.wires == Wires(("b"))
    assert op.control_wires == Wires(("a"))
