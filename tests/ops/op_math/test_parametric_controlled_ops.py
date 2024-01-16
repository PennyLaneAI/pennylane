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
Unit tests for parametric Operators inheriting from ControlledOp.
"""

from functools import reduce

import numpy as np
import pytest
from gate_data import ControlledPhaseShift
from scipy.linalg import fractional_matrix_power
from scipy.stats import unitary_group

import pennylane as qml
from pennylane.wires import Wires
from pennylane.ops.qubit.matrix_ops import QubitUnitary

X = np.array([[0, 1], [1, 0]])
X_broadcasted = np.array([X] * 3)


def dot_broadcasted(a, b):
    return np.einsum("...ij,...jk->...ik", a, b)


def multi_dot_broadcasted(matrices):
    return reduce(dot_broadcasted, matrices)


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
        """Test that a ControlledQubitUnitary raised to a non-integer power evalutes."""
        U1 = np.array(
            [
                [0.73708696 + 0.61324932j, 0.27034258 + 0.08685028j],
                [-0.24979544 - 0.1350197j, 0.95278437 + 0.1075819j],
            ]
        )

        op = qml.ControlledQubitUnitary(U1, control_wires=("b", "c"), wires="a")

        z = 0.12
        [pow_op] = op.pow(z)
        expected = np.eye(8, dtype=complex)
        expected[-2:, -2:] = fractional_matrix_power(U1, z)
        assert qml.math.allequal(pow_op.matrix(), expected)

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


class TestDecompositions:
    def test_CRX_decomposition(self):
        """Test the decomposition for CRX."""
        phi = 0.432

        ops1 = qml.CRX.compute_decomposition(phi, wires=[0, 1])
        ops2 = qml.CRX(phi, wires=(0, 1)).decomposition()

        classes = [qml.RZ, qml.RY, qml.CNOT, qml.RY, qml.CNOT, qml.RZ]
        params = [[np.pi / 2], [phi / 2], [], [-phi / 2], [], [-np.pi / 2]]
        wires = [Wires(1), Wires(1), Wires((0, 1)), Wires(1), Wires((0, 1)), Wires(1)]

        for ops in [ops1, ops2]:
            for op, c, p, w in zip(ops, classes, params, wires):
                assert isinstance(op, c)
                assert op.parameters == p
                assert op.wires == w

    def test_CRX_decomposition_broadcasted(self):
        """Test the decomposition for broadcasted CRX."""
        phi = np.array([0.1, 2.1])

        ops1 = qml.CRX.compute_decomposition(phi, wires=[0, 1])
        ops2 = qml.CRX(phi, wires=(0, 1)).decomposition()

        classes = [qml.RZ, qml.RY, qml.CNOT, qml.RY, qml.CNOT, qml.RZ]
        params = [[np.pi / 2], [phi / 2], [], [-phi / 2], [], [-np.pi / 2]]
        wires = [Wires(1), Wires(1), Wires((0, 1)), Wires(1), Wires((0, 1)), Wires(1)]

        for ops in [ops1, ops2]:
            for op, c, p, w in zip(ops, classes, params, wires):
                assert isinstance(op, c)
                assert qml.math.allclose(op.parameters, p)
                assert op.wires == w

    def test_CRY_decomposition(self):
        """Test the decomposition for CRY."""
        phi = 0.432

        ops1 = qml.CRY.compute_decomposition(phi, wires=[0, 1])
        ops2 = qml.CRY(phi, wires=(0, 1)).decomposition()

        classes = [qml.RY, qml.CNOT, qml.RY, qml.CNOT]
        params = [[phi / 2], [], [-phi / 2], []]
        wires = [Wires(1), Wires((0, 1)), Wires(1), Wires((0, 1))]

        for ops in [ops1, ops2]:
            for op, c, p, w in zip(ops, classes, params, wires):
                assert isinstance(op, c)
                assert np.allclose(op.parameters, p)
                assert op.wires == w

    def test_CRY_decomposition_broadcasted(self):
        """Test the decomposition for broadcastedCRY."""
        phi = np.array([2.1, 0.2])

        ops1 = qml.CRY.compute_decomposition(phi, wires=[0, 1])
        ops2 = qml.CRY(phi, wires=(0, 1)).decomposition()

        classes = [qml.RY, qml.CNOT, qml.RY, qml.CNOT]
        params = [[phi / 2], [], [-phi / 2], []]
        wires = [Wires(1), Wires((0, 1)), Wires(1), Wires((0, 1))]

        for ops in [ops1, ops2]:
            for op, c, p, w in zip(ops, classes, params, wires):
                assert isinstance(op, c)
                assert np.allclose(op.parameters, p)
                assert op.wires == w

    def test_CRZ_decomposition(self):
        """Test the decomposition for CRZ."""
        phi = 0.321

        ops1 = qml.CRZ.compute_decomposition(phi, wires=[0, 1])
        ops2 = qml.CRZ(phi, wires=(0, 1)).decomposition()

        classes = [qml.PhaseShift, qml.CNOT, qml.PhaseShift, qml.CNOT]
        params = [[phi / 2], [], [-phi / 2], []]
        wires = [Wires(1), Wires((0, 1)), Wires(1), Wires((0, 1))]

        for ops in [ops1, ops2]:
            for op, c, p, w in zip(ops, classes, params, wires):
                assert isinstance(op, c)
                assert np.allclose(op.parameters, p)
                assert op.wires == w

    def test_CRZ_decomposition_broadcasted(self):
        """Test the decomposition for broadcasted CRZ."""
        phi = np.array([0.6, 2.1])

        ops1 = qml.CRZ.compute_decomposition(phi, wires=[0, 1])
        ops2 = qml.CRZ(phi, wires=(0, 1)).decomposition()

        classes = [qml.PhaseShift, qml.CNOT, qml.PhaseShift, qml.CNOT]
        params = [[phi / 2], [], [-phi / 2], []]
        wires = [Wires(1), Wires((0, 1)), Wires(1), Wires((0, 1))]

        for ops in [ops1, ops2]:
            for op, c, p, w in zip(ops, classes, params, wires):
                assert isinstance(op, c)
                assert np.allclose(op.parameters, p)
                assert op.wires == w

    @pytest.mark.parametrize("phi, theta, omega", [[0.5, 0.6, 0.7], [0.1, -0.4, 0.7], [-10, 5, -1]])
    def test_CRot_decomposition(self, tol, phi, theta, omega):
        """Tests that the decomposition of the CRot gate is correct"""
        op = qml.CRot(phi, theta, omega, wires=[0, 1])
        res = op.decomposition()

        mats = []
        for i in reversed(res):
            if len(i.wires) == 1:
                mats.append(np.kron(np.eye(2), i.matrix()))
            else:
                mats.append(i.matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "phi, theta, omega",
        [
            [np.array([0.1, 0.2]), np.array([-0.4, 2.19]), np.array([0.7, -0.7])],
            [np.array([0.1, 0.2, 0.9]), -0.4, np.array([0.7, 0.0, -0.7])],
        ],
    )
    def test_CRot_decomposition_broadcasted(self, tol, phi, theta, omega):
        """Tests that the decomposition of the broadcasted CRot gate is correct"""
        op = qml.CRot(phi, theta, omega, wires=[0, 1])
        res = op.decomposition()

        mats = []
        for i in reversed(res):
            mat = i.matrix()
            if len(i.wires) == 1:
                I = np.eye(2)[np.newaxis] if qml.math.ndim(mat) == 3 else np.eye(2)
                mats.append(np.kron(I, mat))
            else:
                mats.append(mat)

        decomposed_matrix = multi_dot_broadcasted(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    @pytest.mark.parametrize("cphase_op", [qml.ControlledPhaseShift, qml.CPhase])
    def test_controlled_phase_shift_decomp(self, phi, cphase_op):
        """Tests that the ControlledPhaseShift and CPhase operation
        calculates the correct decomposition"""
        op = cphase_op(phi, wires=[0, 2])
        decomp = op.decomposition()

        mats = []
        for i in reversed(decomp):
            if i.wires.tolist() == [0]:
                mats.append(np.kron(i.matrix(), np.eye(4)))
            elif i.wires.tolist() == [1]:
                mats.append(np.kron(np.eye(2), np.kron(i.matrix(), np.eye(2))))
            elif i.wires.tolist() == [2]:
                mats.append(np.kron(np.eye(4), i.matrix()))
            elif isinstance(i, qml.CNOT) and i.wires.tolist() == [0, 1]:
                mats.append(np.kron(i.matrix(), np.eye(2)))
            elif isinstance(i, qml.CNOT) and i.wires.tolist() == [0, 2]:
                mats.append(
                    np.array(
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                        ]
                    )
                )

        decomposed_matrix = np.linalg.multi_dot(mats)
        lam = np.exp(1j * phi)
        exp = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, lam, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, lam],
            ]
        )

        assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("cphase_op", [qml.ControlledPhaseShift, qml.CPhase])
    def test_controlled_phase_shift_decomp_broadcasted(self, cphase_op):
        """Tests that the ControlledPhaseShift and CPhase operation
        calculates the correct decomposition"""
        phi = np.array([-0.2, 4.2, 1.8])
        op = cphase_op(phi, wires=[0, 2])
        decomp = op.decomposition()

        mats = []
        for i in reversed(decomp):
            mat = i.matrix()
            eye = np.eye(2)[np.newaxis] if np.ndim(mat) == 3 else np.eye(2)
            if i.wires.tolist() == [0]:
                mats.append(np.kron(mat, np.kron(eye, eye)))
            elif i.wires.tolist() == [1]:
                mats.append(np.kron(eye, np.kron(mat, eye)))
            elif i.wires.tolist() == [2]:
                mats.append(np.kron(np.kron(eye, eye), mat))
            elif isinstance(i, qml.CNOT) and i.wires.tolist() == [0, 1]:
                mats.append(np.kron(mat, eye))
            elif isinstance(i, qml.CNOT) and i.wires.tolist() == [0, 2]:
                mats.append(
                    np.array(
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                        ]
                    )
                )

        decomposed_matrix = multi_dot_broadcasted(mats)
        lam = np.exp(1j * phi)
        exp = np.array([np.diag([1, 1, 1, 1, 1, el, 1, el]) for el in lam])

        assert np.allclose(decomposed_matrix, exp)


class TestMatrix:
    def test_CRX(self, tol):
        """Test controlled x rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.CRX.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.CRX(0, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected_pi_half = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), -1j / np.sqrt(2)],
                [0, 0, -1j / np.sqrt(2), 1 / np.sqrt(2)],
            ]
        )
        assert np.allclose(qml.CRX.compute_matrix(np.pi / 2), expected_pi_half, atol=tol, rtol=0)
        assert np.allclose(
            qml.CRX(np.pi / 2, wires=[0, 1]).matrix(), expected_pi_half, atol=tol, rtol=0
        )

        # test identity for theta=pi
        expected_pi = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]])
        assert np.allclose(qml.CRX.compute_matrix(np.pi), expected_pi, atol=tol, rtol=0)
        assert np.allclose(qml.CRX(np.pi, wires=[0, 1]).matrix(), expected_pi, atol=tol, rtol=0)

        # test broadcasting
        param = np.array([np.pi / 2, np.pi])
        expected = [expected_pi_half, expected_pi]
        assert np.allclose(qml.CRX.compute_matrix(param), expected, atol=tol, rtol=0)
        assert np.allclose(qml.CRX(param, wires=[0, 1]).matrix(), expected, atol=tol, rtol=0)

    def test_CRY(self, tol):
        """Test controlled y rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.CRY.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.CRY(0, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected_pi_half = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            ]
        )
        assert np.allclose(qml.CRY.compute_matrix(np.pi / 2), expected_pi_half, atol=tol, rtol=0)
        assert np.allclose(
            qml.CRY(np.pi / 2, wires=[0, 1]).matrix(), expected_pi_half, atol=tol, rtol=0
        )

        # test identity for theta=pi
        expected_pi = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        assert np.allclose(qml.CRY.compute_matrix(np.pi), expected_pi, atol=tol, rtol=0)
        assert np.allclose(qml.CRY(np.pi, wires=[0, 1]).matrix(), expected_pi, atol=tol, rtol=0)

        # test broadcasting
        param = np.array([np.pi / 2, np.pi])
        expected = [expected_pi_half, expected_pi]
        assert np.allclose(qml.CRY.compute_matrix(param), expected, atol=tol, rtol=0)
        assert np.allclose(qml.CRY(param, wires=[0, 1]).matrix(), expected, atol=tol, rtol=0)

    def test_CRZ(self, tol):
        """Test controlled z rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.CRZ.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(0, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected_pi_half = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * np.pi / 4), 0],
                [0, 0, 0, np.exp(1j * np.pi / 4)],
            ]
        )
        assert np.allclose(qml.CRZ.compute_matrix(np.pi / 2), expected_pi_half, atol=tol, rtol=0)
        assert np.allclose(
            qml.CRZ(np.pi / 2, wires=[0, 1]).matrix(), expected_pi_half, atol=tol, rtol=0
        )

        # test identity for theta=pi
        expected_pi = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1j]])
        assert np.allclose(qml.CRZ.compute_matrix(np.pi), expected_pi, atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(np.pi, wires=[0, 1]).matrix(), expected_pi, atol=tol, rtol=0)

        # test broadcasting
        param = np.array([np.pi / 2, np.pi])
        expected = [expected_pi_half, expected_pi]
        assert np.allclose(qml.CRZ.compute_matrix(param), expected, atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(param, wires=[0, 1]).matrix(), expected, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_CRZ_tf(self, tol):
        """Test controlled z rotation is correct when used with Tensorflow,
        because the code differs in that case."""
        import tensorflow as tf

        # test identity for theta=0
        z = tf.Variable(0.0)
        assert np.allclose(qml.CRZ.compute_matrix(z), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(z, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected_pi_half = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * np.pi / 4), 0],
                [0, 0, 0, np.exp(1j * np.pi / 4)],
            ]
        )
        phi = tf.Variable(np.pi / 2)
        assert np.allclose(qml.CRZ.compute_matrix(phi), expected_pi_half, atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(phi, wires=[0, 1]).matrix(), expected_pi_half, atol=tol, rtol=0)

        # test identity for theta=pi
        expected_pi = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1j]])
        phi = tf.Variable(np.pi)
        assert np.allclose(qml.CRZ.compute_matrix(phi), expected_pi, atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(phi, wires=[0, 1]).matrix(), expected_pi, atol=tol, rtol=0)

        # test broadcasting
        param = np.array([np.pi / 2, np.pi])
        expected = [expected_pi_half, expected_pi]
        param_tf = tf.Variable(param)
        assert np.allclose(qml.CRZ.compute_matrix(param_tf), expected, atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(param_tf, wires=[0, 1]).matrix(), expected, atol=tol, rtol=0)

    def test_CRot(self, tol):
        """Test controlled arbitrary rotation is correct"""

        # test identity for phi,theta,omega=0
        assert np.allclose(qml.CRot.compute_matrix(0, 0, 0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(
            qml.CRot(0, 0, 0, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0
        )

        # test identity for phi,theta,omega=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        assert np.allclose(qml.CRot.compute_matrix(np.pi, np.pi, np.pi), expected, atol=tol, rtol=0)
        assert np.allclose(
            qml.CRot(np.pi, np.pi, np.pi, wires=[0, 1]).matrix(), expected, atol=tol, rtol=0
        )

        def arbitrary_Crotation(x, y, z):
            """controlled arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, np.exp(-0.5j * (x + z)) * c, -np.exp(0.5j * (x - z)) * s],
                    [0, 0, np.exp(-0.5j * (x - z)) * s, np.exp(0.5j * (x + z)) * c],
                ]
            )

        a, b, c = 0.432, -0.152, 0.9234
        assert np.allclose(
            qml.CRot.compute_matrix(a, b, c), arbitrary_Crotation(a, b, c), atol=tol, rtol=0
        )
        assert np.allclose(
            qml.CRot(a, b, c, wires=[0, 1]).matrix(),
            arbitrary_Crotation(a, b, c),
            atol=tol,
            rtol=0,
        )

    def test_CRot_broadcasted(self, tol):
        """Test broadcasted controlled arbitrary rotation is correct"""

        # test identity for phi,theta,omega=0
        z = np.zeros(5)
        assert np.allclose(qml.CRot.compute_matrix(z, z, z), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(
            qml.CRot(z, z, z, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0
        )

        # test -i*CY for phi,theta,omega=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        pi = np.ones(3) * np.pi
        assert np.allclose(qml.CRot.compute_matrix(pi, pi, pi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.CRot(pi, pi, pi, wires=[0, 1]).matrix(), expected, atol=tol, rtol=0)

        def arbitrary_Crotation(x, y, z):
            """controlled arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, np.exp(-0.5j * (_x + _z)) * _c, -np.exp(0.5j * (_x - _z)) * _s],
                        [0, 0, np.exp(-0.5j * (_x - _z)) * _s, np.exp(0.5j * (_x + _z)) * _c],
                    ]
                    for _x, _z, _c, _s in zip(x, z, c, s)
                ]
            )

        a, b, c = np.array([0.432, -0.124]), np.array([-0.152, 2.912]), np.array([0.9234, -9.2])
        assert np.allclose(
            qml.CRot.compute_matrix(a, b, c), arbitrary_Crotation(a, b, c), atol=tol, rtol=0
        )
        assert np.allclose(
            qml.CRot(a, b, c, wires=[0, 1]).matrix(),
            arbitrary_Crotation(a, b, c),
            atol=tol,
            rtol=0,
        )

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    @pytest.mark.parametrize("cphase_op", [qml.ControlledPhaseShift, qml.CPhase])
    def test_controlled_phase_shift_matrix_and_eigvals(self, phi, cphase_op):
        """Tests that the ControlledPhaseShift and CPhase operation calculate the correct
        matrix and eigenvalues"""
        op = cphase_op(phi, wires=[0, 1])
        res = op.matrix()
        exp = ControlledPhaseShift(phi)
        assert np.allclose(res, exp)

        res = op.eigvals()
        assert np.allclose(np.diag(res), exp)

    @pytest.mark.tf
    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    @pytest.mark.parametrize("cphase_op", [qml.ControlledPhaseShift, qml.CPhase])
    def test_controlled_phase_shift_matrix_and_eigvals_tf(self, phi, cphase_op):
        """Tests that the ControlledPhaseShift and CPhase operation calculate the correct
        matrix and eigenvalues for the Tensorflow interface, because the code differs
        in that case."""
        import tensorflow as tf

        op = cphase_op(tf.Variable(phi), wires=[0, 1])
        res = op.matrix()
        exp = ControlledPhaseShift(phi)
        assert np.allclose(res, exp)

        res = op.eigvals()
        assert np.allclose(np.diag(res), exp)

    @pytest.mark.parametrize("cphase_op", [qml.ControlledPhaseShift, qml.CPhase])
    def test_controlled_phase_shift_matrix_and_eigvals_broadcasted(self, cphase_op):
        """Tests that the ControlledPhaseShift and CPhase operation calculate the
        correct matrix and eigenvalues for broadcasted parameters"""
        phi = np.array([0.2, np.pi / 2, -0.1])
        op = cphase_op(phi, wires=[0, 1])
        res = op.matrix()
        expected = np.array([np.eye(4, dtype=complex)] * 3)
        expected[..., 3, 3] = np.exp(1j * phi)
        assert np.allclose(res, expected)

        res = op.eigvals()
        exp_eigvals = np.ones((3, 4), dtype=complex)
        exp_eigvals[..., 3] = np.exp(1j * phi)
        assert np.allclose(res, exp_eigvals)

    @pytest.mark.tf
    @pytest.mark.parametrize("cphase_op", [qml.ControlledPhaseShift, qml.CPhase])
    def test_controlled_phase_shift_matrix_and_eigvals_broadcasted_tf(self, cphase_op):
        """Tests that the ControlledPhaseShift and CPhase operation calculate the
        correct matrix and eigenvalues for broadcasted parameters and Tensorflow,
        because the code differs for that interface."""
        import tensorflow as tf

        phi = np.array([0.2, np.pi / 2, -0.1])
        phi_tf = tf.Variable(phi)
        op = cphase_op(phi_tf, wires=[0, 1])
        res = op.matrix()
        expected = np.array([np.eye(4, dtype=complex)] * 3)
        expected[..., 3, 3] = np.exp(1j * phi)
        assert np.allclose(res, expected)

        res = op.eigvals()
        exp_eigvals = np.ones((3, 4), dtype=complex)
        exp_eigvals[..., 3] = np.exp(1j * phi)
        assert np.allclose(res, exp_eigvals)


class TestEigvals:  # pylint: disable=too-few-public-methods
    """Tests for the Eigvals operation"""

    def test_crz_eigvals(self, tol):
        """Test controlled z rotation eigvals are correct"""

        # test identity for theta=0
        assert np.allclose(qml.CRZ.compute_eigvals(0), np.ones(4), atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(0, wires=[0, 1]).eigvals(), np.ones(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected_pi_half = np.array([1, 1, np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)])
        assert np.allclose(qml.CRZ.compute_eigvals(np.pi / 2), expected_pi_half, atol=tol, rtol=0)
        assert np.allclose(
            qml.CRZ(np.pi / 2, wires=[0, 1]).eigvals(), expected_pi_half, atol=tol, rtol=0
        )

        # test identity for theta=pi
        expected_pi = np.array([1, 1, -1j, 1j])
        assert np.allclose(qml.CRZ.compute_eigvals(np.pi), expected_pi, atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(np.pi, wires=[0, 1]).eigvals(), expected_pi, atol=tol, rtol=0)

        # test broadcasting
        param = np.array([np.pi / 2, np.pi])
        expected = [expected_pi_half, expected_pi]
        assert np.allclose(qml.CRZ.compute_eigvals(param), expected, atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(param, wires=[0, 1]).eigvals(), expected, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_crz_eigvals_tf(self, tol):
        """Test controlled z rotation eigvals are correct with Tensorflow, because the
        code differs for that interface."""
        import tensorflow as tf

        # test identity for theta=0
        z = tf.Variable(0)
        assert np.allclose(qml.CRZ.compute_eigvals(z), np.ones(4), atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(z, wires=[0, 1]).eigvals(), np.ones(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected_pi_half = np.array([1, 1, np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)])
        phi = tf.Variable(np.pi / 2)
        assert np.allclose(qml.CRZ.compute_eigvals(phi), expected_pi_half, atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(phi, wires=[0, 1]).eigvals(), expected_pi_half, atol=tol, rtol=0)

        # test identity for theta=pi
        expected_pi = np.array([1, 1, -1j, 1j])
        phi = tf.Variable(np.pi)
        assert np.allclose(qml.CRZ.compute_eigvals(phi), expected_pi, atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(phi, wires=[0, 1]).eigvals(), expected_pi, atol=tol, rtol=0)

        # test broadcasting
        param = np.array([np.pi / 2, np.pi])
        param_tf = tf.Variable(param)
        expected = [expected_pi_half, expected_pi]
        assert np.allclose(qml.CRZ.compute_eigvals(param_tf), expected, atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(param_tf, wires=[0, 1]).eigvals(), expected, atol=tol, rtol=0)


def test_simplify_crot():
    """Simplify CRot operations with different parameters."""

    crot_x = qml.CRot(np.pi / 2, 0.1, -np.pi / 2, wires=[0, 1])
    simplify_crot_x = crot_x.simplify()

    assert simplify_crot_x.name == "CRX"
    assert simplify_crot_x.data == (0.1,)
    assert np.allclose(simplify_crot_x.matrix(), crot_x.matrix())

    crot_y = qml.CRot(0, 0.1, 0, wires=[0, 1])
    simplify_crot_y = crot_y.simplify()

    assert simplify_crot_y.name == "CRY"
    assert simplify_crot_y.data == (0.1,)
    assert np.allclose(simplify_crot_y.matrix(), crot_y.matrix())

    crot_z = qml.CRot(0.1, 0, 0.2, wires=[0, 1])
    simplify_crot_z = crot_z.simplify()

    assert simplify_crot_z.name == "CRZ"
    assert np.allclose(simplify_crot_z.data, [0.3])
    assert np.allclose(simplify_crot_z.matrix(), crot_z.matrix())

    crot = qml.CRot(0.1, 0.2, 0.3, wires=[0, 1])
    not_simplified_crot = crot.simplify()

    assert not_simplified_crot.name == "CRot"
    assert np.allclose(not_simplified_crot.matrix(), crot.matrix())


controlled_data = [
    (qml.RX(1.234, wires=0), qml.CRX(1.234, wires=("a", 0))),
    (qml.RY(1.234, wires=0), qml.CRY(1.234, wires=("a", 0))),
    (qml.RZ(1.234, wires=0), qml.CRZ(1.234, wires=("a", 0))),
    (qml.PhaseShift(1.234, wires=0), qml.ControlledPhaseShift(1.234, wires=("a", 0))),
    (qml.Rot(1.2, 2.3, 3.4, wires=0), qml.CRot(1.2, 2.3, 3.4, wires=("a", 0))),
]


@pytest.mark.parametrize("base, cbase", controlled_data)
def test_controlled_method(base, cbase):
    """Tests the _controlled method for parametric ops."""
    # pylint: disable=protected-access
    assert qml.equal(base._controlled("a"), cbase)
