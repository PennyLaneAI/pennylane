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
Unit tests for Operators inheriting from ControlledOp.
"""

import numpy as np
import pytest
from gate_data import CY, CZ, ControlledPhaseShift, CRot3, CRotx, CRoty, CRotz
from scipy import sparse
from scipy.linalg import fractional_matrix_power
from scipy.stats import unitary_group

import pennylane as qml
from pennylane.wires import Wires

NON_PARAMETRIZED_OPERATIONS = [
    (qml.CY, CY),
    (qml.CZ, CZ),
]

PARAMETRIZED_OPERATIONS = [
    (qml.CRX, CRotx),
    (qml.CRY, CRoty),
    (qml.CRZ, CRotz),
    (qml.CRot, CRot3),
    (qml.ControlledPhaseShift, ControlledPhaseShift),
]

ALL_OPERATIONS = NON_PARAMETRIZED_OPERATIONS + PARAMETRIZED_OPERATIONS

NON_PARAMETRIC_OPS_DECOMPOSITIONS = (
    (qml.CY, [qml.CRY(np.pi, wires=[0, 1]), qml.S(0)]),
    (qml.CZ, [qml.ControlledPhaseShift(np.pi, wires=[0, 1])]),
)

X = np.array([[0, 1], [1, 0]])
X_broadcasted = np.array([X] * 3)


# pylint: disable=too-many-public-methods
class TestControlledQubitUnitary:
    """Tests specific to the ControlledQubitUnitary operation"""

    def test_has_decomposition_sparse_edge_case(self):
        """Test that has_decomposition doesn't error out with sparse matrices."""
        U = np.random.rand(4, 4) + 1.0j * np.random.rand(4, 4)
        U, _ = np.linalg.qr(U)
        U = sparse.csr_matrix(U)
        op = qml.ControlledQubitUnitary(U, wires=(0, 1, 2))
        assert not op.has_decomposition
        with pytest.raises(qml.operation.DecompositionUndefinedError):
            op.decomposition()

    @pytest.mark.parametrize(
        "op",
        [
            qml.ControlledQubitUnitary(np.eye(2), wires=(1, 2, 3)),
            qml.ControlledQubitUnitary(np.eye(2), wires=(1, 2, 3, 4)),
        ],
    )
    def test_standard_validity(self, op):
        """Test that the operation is valid."""
        qml.ops.functions.assert_valid(op, skip_differentiation=True)

    def test_noniterable_base(self):
        """Test that an error is raised if the user provides a non-iterable base operator"""

        # make a non-iterable, non-QubitUnitary base operator
        def base_op(x):
            return 1 if x == 0 else 0  # pauliX as a mapping

        with pytest.raises(ValueError, match="Base must be a matrix"):
            qml.ControlledQubitUnitary(base_op, wires=[0, 1])

    def test_wires_is_none(self):
        """Test that an error is raised if the user provides no target wires for an iterable base operator"""
        base_op = [[0, 1], [1, 0]]

        with pytest.raises(TypeError, match="Must specify a set of wires"):
            qml.ControlledQubitUnitary(base_op, wires=None)

    @pytest.mark.capture
    @pytest.mark.parametrize(
        "control_wires, wires",
        [(0, 1), ([0, 1], [2])],
    )
    def test_consistency_with_capture(self, control_wires, wires):
        """Test that the operator wires are as expected with capture enabled"""
        base_op = [[0, 1], [1, 0]]

        op_kwarg = qml.ControlledQubitUnitary(base_op, wires=Wires(control_wires) + Wires(wires))
        assert op_kwarg.base.wires == Wires(wires)
        assert op_kwarg.control_wires == Wires(control_wires)
        op = qml.ControlledQubitUnitary(base_op, wires=Wires(control_wires) + Wires(wires))
        assert op.base.wires == Wires(wires)
        assert op.control_wires == Wires(control_wires)

    @pytest.mark.capture
    def test_pairwise_consistency_with_capture(self):
        """Test that both combinations of control and target wires lead to the same operator"""
        base_op = [[0, 1], [1, 0]]

        control_wires_1, target_wires_1 = [0, 1], [2]
        op_1 = qml.ControlledQubitUnitary(base_op, wires=control_wires_1 + target_wires_1)

        assert op_1.base.wires == Wires(2)
        assert op_1.control_wires == Wires([0, 1])

        control_wires_2, target_wires_2 = [0, 1, 2], ()
        op_2 = qml.ControlledQubitUnitary(
            base_op, wires=Wires(control_wires_2) + Wires(target_wires_2)
        )

        assert op_2.base.wires == Wires(2)
        assert op_2.control_wires == Wires([0, 1])

    @pytest.mark.parametrize("target_wire", list(range(3)))
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
            qml.ControlledQubitUnitary(X, wires=control_wires + [target_wire])
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
            qml.ControlledQubitUnitary(X_broadcasted, wires=control_wires + [target_wire])
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
            qml.ControlledQubitUnitary(U, wires=control_wires + target_wires)
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
                X, wires=control_wires + target_wires, control_values=control_values
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
                U, wires=control_wires + target_wires, control_values=control_values
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

            qml.ControlledQubitUnitary(U, wires=control_wires + target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            return qml.state()

        mixed_polarity_state = circuit_mixed_polarity()
        pauli_x_state = circuit_pauli_x()

        assert np.allclose(mixed_polarity_state, pauli_x_state)

    def test_same_as_Toffoli(self):
        """Test if ControlledQubitUnitary returns the correct matrix for a control-control-X
        (Toffoli) gate"""
        mat = qml.ControlledQubitUnitary(X, wires=[0, 1, 2]).matrix()
        mat2 = qml.Toffoli(wires=[0, 1, 2]).matrix()
        assert np.allclose(mat, mat2)

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""
        U = np.array([[0.94877869, 0.31594146], [-0.31594146, 0.94877869]])
        res_dynamic = qml.ControlledQubitUnitary(U, wires=[1, 0]).matrix()
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

        res_dynamic = qml.ControlledQubitUnitary(U, wires=[1, 0]).matrix()
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

        op = qml.ControlledQubitUnitary(U1, wires=("b", "c", "a"), control_values="01")

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

        op = qml.ControlledQubitUnitary(U1, wires=("b", "c", "a"))

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

        op = qml.ControlledQubitUnitary(U1, wires=("b", "c", "a"))

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

        op = qml.ControlledQubitUnitary(U1, wires=("b", "c", "a"))

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(0.12)

    def test_controlled(self):
        """Test the _controlled method for ControlledQubitUnitary."""

        U = qml.PauliX(0).compute_matrix()

        original = qml.ControlledQubitUnitary(U, wires=(0, 1, 4), control_values="01")
        expected = qml.ControlledQubitUnitary(U, wires=(0, 1, "a", 4), control_values="011")

        out = original._controlled("a")  # pylint: disable=protected-access
        qml.assert_equal(out, expected)

    def test_unitary_check(self):
        unitary = np.array([[0.94877869j, 0.31594146], [-0.31594146, 0.94877869j]])
        not_unitary = np.array([[0.94877869j, 0.31594146], [-5, 0.94877869j]])

        qml.ControlledQubitUnitary(unitary, wires=[0, 2, 1], unitary_check=True)

        with pytest.warns(UserWarning, match="may not be unitary"):
            qml.ControlledQubitUnitary(not_unitary, wires=[0, 2, 1], unitary_check=True)


@pytest.mark.parametrize("op_cls, _", NON_PARAMETRIZED_OPERATIONS)
def test_map_wires_non_parametric(op_cls, _):
    """Test get and set private wires in non-parametric controlled operations."""

    op = op_cls(wires=[0, 1])
    assert op.wires == Wires((0, 1))

    op = op.map_wires(wire_map={0: "a", 1: "b"})
    assert op.base.wires == Wires("b")
    assert op.control_wires == Wires("a")


def test_controlled_phase_shift_alias():
    """Tests that the alias for ControlledPhaseShift works"""
    qml.assert_equal(qml.ControlledPhaseShift(0.123, wires=[0, 1]), qml.CPhase(0.123, wires=[0, 1]))


def _arbitrary_crot(x, y, z):
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


def _arbitrary_crot_broadcasted(x, y, z):
    """controlled arbitrary single qubit rotation broadcasted"""

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


def _phase_shift_matrix_broadcasted(phi):
    """Phase shift matrix broadcasted"""

    expected = np.array([np.eye(4, dtype=complex)] * 3)
    expected[..., 3, 3] = np.exp(1j * phi)
    return expected


EXPECTED_MATRICES = [
    (qml.CRX, [0], np.identity(4)),
    (
        qml.CRX,
        [np.pi / 2],
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), -1j / np.sqrt(2)],
                [0, 0, -1j / np.sqrt(2), 1 / np.sqrt(2)],
            ]
        ),
    ),
    (qml.CRX, [np.pi], np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]])),
    (
        qml.CRX,
        [np.array([np.pi / 2, np.pi])],
        np.array(
            [
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1 / np.sqrt(2), -1j / np.sqrt(2)],
                    [0, 0, -1j / np.sqrt(2), 1 / np.sqrt(2)],
                ],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]],
            ]
        ),
    ),
    (qml.CRY, [0], np.identity(4)),
    (
        qml.CRY,
        [np.pi / 2],
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            ]
        ),
    ),
    (qml.CRY, [np.pi], np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])),
    (
        qml.CRY,
        [np.array([np.pi / 2, np.pi])],
        np.array(
            [
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
                    [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                ],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],
            ]
        ),
    ),
    (qml.CRZ, [0], np.identity(4)),
    (
        qml.CRZ,
        [np.pi / 2],
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * np.pi / 4), 0],
                [0, 0, 0, np.exp(1j * np.pi / 4)],
            ]
        ),
    ),
    (qml.CRZ, [np.pi], np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1j]])),
    (
        qml.CRZ,
        [np.array([np.pi / 2, np.pi])],
        np.array(
            [
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, np.exp(-1j * np.pi / 4), 0],
                    [0, 0, 0, np.exp(1j * np.pi / 4)],
                ],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1j]],
            ]
        ),
    ),
    (qml.CRot, [0, 0, 0], np.identity(4)),
    (
        qml.CRot,
        [np.pi, np.pi, np.pi],
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]),
    ),
    (qml.CRot, [0.432, -0.152, 0.9234], _arbitrary_crot(0.432, -0.152, 0.9234)),
    (qml.CRot, [np.zeros(5), np.zeros(5), np.zeros(5)], np.identity(4)),
    (
        qml.CRot,
        [np.ones(3) * np.pi, np.ones(3) * np.pi, np.ones(3) * np.pi],
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]),
    ),
    (
        qml.CRot,
        [np.array([0.432, -0.124]), np.array([-0.152, 2.912]), np.array([0.9234, -9.2])],
        _arbitrary_crot_broadcasted(
            np.array([0.432, -0.124]), np.array([-0.152, 2.912]), np.array([0.9234, -9.2])
        ),
    ),
    (qml.ControlledPhaseShift, [0.123], ControlledPhaseShift(0.123)),
    (
        qml.ControlledPhaseShift,
        [np.array([0.2, np.pi / 2, -0.1])],
        _phase_shift_matrix_broadcasted(np.array([0.2, np.pi / 2, -0.1])),
    ),
]

EXPECTED_EIGVALS = [
    (qml.CRZ, [0], np.ones(4)),
    (qml.CRZ, [np.pi / 2], np.array([1, 1, np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)])),
    (qml.CRZ, [np.pi], np.array([1, 1, -1j, 1j])),
    (
        qml.CRZ,
        [np.array([np.pi / 2, np.pi])],
        [
            np.array([1, 1, np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)]),
            np.array([1, 1, -1j, 1j]),
        ],
    ),
    (qml.ControlledPhaseShift, [0.123], np.linalg.eigvals(ControlledPhaseShift(0.123))),
    (
        qml.ControlledPhaseShift,
        [np.array([0.2, np.pi / 2, -0.1])],
        np.linalg.eigvals(_phase_shift_matrix_broadcasted(np.array([0.2, np.pi / 2, -0.1]))),
    ),
]


class TestComputations:
    """Tests computing the matrix and eigenvalues of a controlled operation"""

    @pytest.mark.parametrize("op, params, expected_matrix", EXPECTED_MATRICES)
    def test_matrix(self, tol, op, params, expected_matrix):
        """Tests that the correct matrix is returned"""

        assert np.allclose(op.compute_matrix(*params), expected_matrix, atol=tol, rtol=0)
        assert np.allclose(op(*params, wires=[0, 1]).matrix(), expected_matrix, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("op, params, expected_matrix", EXPECTED_MATRICES)
    def test_matrix_tf(self, tol, op, params, expected_matrix):
        """Tests that the correct matrix is returned when using TensorFlow"""

        import tensorflow as tf

        variables = [tf.Variable(param) for param in params]

        assert np.allclose(op.compute_matrix(*variables), expected_matrix, atol=tol, rtol=0)
        assert np.allclose(op(*variables, wires=[0, 1]).matrix(), expected_matrix, atol=tol, rtol=0)

    @pytest.mark.parametrize("op, params, expected_eigvals", EXPECTED_EIGVALS)
    def test_eigvals(self, tol, op, params, expected_eigvals):
        """Tests that the correct eigenvalues are returned"""

        assert np.allclose(op.compute_eigvals(*params), expected_eigvals, atol=tol, rtol=0)
        assert np.allclose(op(*params, wires=[0, 1]).eigvals(), expected_eigvals, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("op, params, expected_eigvals", EXPECTED_EIGVALS)
    def test_eigvals_tf(self, tol, op, params, expected_eigvals):
        """Tests that the correct eigenvalues are returned when using TensorFlow"""

        import tensorflow as tf

        variables = [tf.Variable(param) for param in params]

        assert np.allclose(op.compute_eigvals(*variables), expected_eigvals, atol=tol, rtol=0)
        assert np.allclose(
            op(*variables, wires=[0, 1]).eigvals(), expected_eigvals, atol=tol, rtol=0
        )


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
    qml.assert_equal(base._controlled("a"), cbase)


@pytest.mark.parametrize(
    "control, control_values",
    [([0, 1], [True, False]), ([10, "a"], (0, 0)), ([2], 1), (2, (True,))],
)
@pytest.mark.parametrize(
    "base_op", [qml.CRX(0.2, [21, 22]), qml.CNOT([21, 22]), qml.CPhase(0.6, [21, 22])]
)
def test_controlling_a_controlled_operation(control, control_values, base_op):
    """Test that a controlled op can be controlled again."""
    qml.ctrl(base_op, control=control, control_values=control_values)


@pytest.mark.parametrize("op_type", (qml.CH, qml.CY, qml.CZ, qml.CNOT))
def test_tuple_control_wires_non_parametric_ops(op_type):
    """Test that tuples can be provided as control wire labels."""

    assert op_type([(0, 1), 2]).wires == qml.wires.Wires([(0, 1), 2])


@pytest.mark.parametrize("op_type", (qml.CRX, qml.CRY, qml.CRZ, qml.CPhase))
def test_tuple_control_wires_parametric_ops(op_type):
    """Test that tuples can be provided as control wire labels."""

    assert op_type(0.123, [(0, 1), 2]).wires == qml.wires.Wires([(0, 1), 2])


def test_CNOT_decomposition():
    """Test that CNOT raises a DecompositionUndefinedError instead of using the
    controlled_op decomposition functions"""
    assert not qml.CNOT((0, 1)).has_decomposition

    with pytest.raises(qml.operation.DecompositionUndefinedError):
        qml.CNOT.compute_decomposition()

    with pytest.raises(qml.operation.DecompositionUndefinedError):
        qml.CNOT([0, 1]).decomposition()
