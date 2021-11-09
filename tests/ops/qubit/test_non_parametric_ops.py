# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the available non-parametric qubit operations
"""
import itertools
import re
import pytest
import copy
import numpy as np
from scipy.stats import unitary_group

import pennylane as qml
from pennylane.wires import Wires

from gate_data import (
    X,
    Y,
    Z,
    H,
    CNOT,
    SWAP,
    ISWAP,
    SISWAP,
    CZ,
    S,
    T,
    CSWAP,
    Toffoli,
)


# Non-parametrized operations and their matrix representation
NON_PARAMETRIZED_OPERATIONS = [
    (qml.CNOT, CNOT),
    (qml.SWAP, SWAP),
    (qml.ISWAP, ISWAP),
    (qml.SISWAP, SISWAP),
    (qml.CZ, CZ),
    (qml.S, S),
    (qml.T, T),
    (qml.CSWAP, CSWAP),
    (qml.Toffoli, Toffoli),
]


class TestOperations:
    @pytest.mark.parametrize("op_cls, mat", NON_PARAMETRIZED_OPERATIONS)
    def test_nonparametrized_op_copy(self, op_cls, mat, tol):
        """Tests that copied nonparametrized ops function as expected"""
        op = op_cls(wires=range(op_cls.num_wires))
        copied_op = copy.copy(op)
        np.testing.assert_allclose(op.matrix, copied_op.matrix, atol=tol)

        op._inverse = True
        copied_op2 = copy.copy(op)
        np.testing.assert_allclose(op.matrix, copied_op2.matrix, atol=tol)

    @pytest.mark.parametrize("ops, mat", NON_PARAMETRIZED_OPERATIONS)
    def test_matrices(self, ops, mat, tol):
        """Test matrices of non-parametrized operations are correct"""
        op = ops(wires=range(ops.num_wires))
        res = op.matrix
        assert np.allclose(res, mat, atol=tol, rtol=0)


class TestDecompositions:
    def test_x_decomposition(self, tol):
        """Tests that the decomposition of the PauliX is correct"""
        op = qml.PauliX(wires=0)
        res = op.decomposition(0)

        assert len(res) == 3

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[1].name == "RX"
        assert res[1].wires == Wires([0])
        assert res[1].data[0] == np.pi

        assert res[2].name == "PhaseShift"
        assert res[2].wires == Wires([0])
        assert res[2].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_y_decomposition(self, tol):
        """Tests that the decomposition of the PauliY is correct"""
        op = qml.PauliY(wires=0)
        res = op.decomposition(0)

        assert len(res) == 3

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[1].name == "RY"
        assert res[1].wires == Wires([0])
        assert res[1].data[0] == np.pi

        assert res[2].name == "PhaseShift"
        assert res[2].wires == Wires([0])
        assert res[2].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_z_decomposition(self, tol):
        """Tests that the decomposition of the PauliZ is correct"""
        op = qml.PauliZ(wires=0)
        res = op.decomposition(0)

        assert len(res) == 1

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi

        decomposed_matrix = res[0].matrix
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_s_decomposition(self, tol):
        """Tests that the decomposition of the S gate is correct"""
        op = qml.S(wires=0)
        res = op.decomposition(0)

        assert len(res) == 1

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        decomposed_matrix = res[0].matrix
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_t_decomposition(self, tol):
        """Tests that the decomposition of the T gate is correct"""
        op = qml.T(wires=0)
        res = op.decomposition(0)

        assert len(res) == 1

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 4

        decomposed_matrix = res[0].matrix
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_sx_decomposition(self, tol):
        """Tests that the decomposition of the SX gate is correct"""
        op = qml.SX(wires=0)
        res = op.decomposition(0)

        assert len(res) == 4

        assert all([res[i].wires == Wires([0]) for i in range(4)])

        assert res[0].name == "RZ"
        assert res[1].name == "RY"
        assert res[2].name == "RZ"
        assert res[3].name == "PhaseShift"

        assert res[0].data[0] == np.pi / 2
        assert res[1].data[0] == np.pi / 2
        assert res[2].data[0] == -np.pi
        assert res[3].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_hadamard_decomposition(self, tol):
        """Tests that the decomposition of the Hadamard gate is correct"""
        op = qml.Hadamard(wires=0)
        res = op.decomposition(0)

        assert len(res) == 3

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[1].name == "RX"
        assert res[1].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[2].name == "PhaseShift"
        assert res[2].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_CY_decomposition(self, tol):
        """Tests that the decomposition of the CY gate is correct"""
        op = qml.CY(wires=[0, 1])
        res = op.decomposition(op.wires)

        mats = []
        for i in reversed(res):
            if len(i.wires) == 1:
                mats.append(np.kron(i.matrix, np.eye(2)))
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_ISWAP_decomposition(self, tol):
        """Tests that the decomposition of the ISWAP gate is correct"""
        op = qml.ISWAP(wires=[0, 1])
        res = op.decomposition(op.wires)

        assert len(res) == 6

        assert res[0].wires == Wires([0])
        assert res[1].wires == Wires([1])
        assert res[2].wires == Wires([0])
        assert res[3].wires == Wires([0, 1])
        assert res[4].wires == Wires([1, 0])
        assert res[5].wires == Wires([1])

        assert res[0].name == "S"
        assert res[1].name == "S"
        assert res[2].name == "Hadamard"
        assert res[3].name == "CNOT"
        assert res[4].name == "CNOT"
        assert res[5].name == "Hadamard"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([1]):
                mats.append(np.kron(np.eye(2), i.matrix))
            elif i.wires == Wires([0]):
                mats.append(np.kron(i.matrix, np.eye(2)))
            elif i.wires == Wires([1, 0]) and i.name == "CNOT":
                mats.append(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    @pytest.mark.parametrize("siswap_op", [qml.SISWAP, qml.SQISW])
    def test_SISWAP_decomposition(self, siswap_op, tol):
        """Tests that the decomposition of the SISWAP gate and its SQISW alias gate is correct"""
        op = siswap_op(wires=[0, 1])
        res = op.decomposition(op.wires)

        assert len(res) == 12

        assert res[0].wires == Wires([0])
        assert res[1].wires == Wires([0])
        assert res[2].wires == Wires([0, 1])
        assert res[3].wires == Wires([0])
        assert res[4].wires == Wires([0])
        assert res[5].wires == Wires([0])
        assert res[6].wires == Wires([0])
        assert res[7].wires == Wires([1])
        assert res[8].wires == Wires([1])
        assert res[9].wires == Wires([0, 1])
        assert res[10].wires == Wires([0])
        assert res[11].wires == Wires([1])

        assert res[0].name == "SX"
        assert res[1].name == "RZ"
        assert res[2].name == "CNOT"
        assert res[3].name == "SX"
        assert res[4].name == "RZ"
        assert res[5].name == "SX"
        assert res[6].name == "RZ"
        assert res[7].name == "SX"
        assert res[8].name == "RZ"
        assert res[9].name == "CNOT"
        assert res[10].name == "SX"
        assert res[11].name == "SX"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([1]):
                mats.append(np.kron(np.eye(2), i.matrix))
            elif i.wires == Wires([0]):
                mats.append(np.kron(i.matrix, np.eye(2)))
            elif i.wires == Wires([1, 0]) and i.name == "CNOT":
                mats.append(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_toffoli_decomposition(self, tol):
        """Tests that the decomposition of the Toffoli gate is correct"""
        op = qml.Toffoli(wires=[0, 1, 2])
        res = op.decomposition(op.wires)

        assert len(res) == 15

        mats = []

        for i in reversed(res):
            if i.wires == Wires([2]):
                mats.append(np.kron(np.eye(4), i.matrix))
            elif i.wires == Wires([1]):
                mats.append(np.kron(np.eye(2), np.kron(i.matrix, np.eye(2))))
            elif i.wires == Wires([0]):
                mats.append(np.kron(i.matrix, np.eye(4)))
            elif i.wires == Wires([0, 1]) and i.name == "CNOT":
                mats.append(np.kron(i.matrix, np.eye(2)))
            elif i.wires == Wires([1, 2]) and i.name == "CNOT":
                mats.append(np.kron(np.eye(2), i.matrix))
            elif i.wires == Wires([0, 2]) and i.name == "CNOT":
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

        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_CSWAP_decomposition(self, tol):
        """Tests that the decomposition of the CSWAP gate is correct"""
        op = qml.CSWAP(wires=[0, 1, 2])
        res = op.decomposition(op.wires)

        assert len(res) == 3

        mats = []

        for i in reversed(res):  # only use 3 toffoli gates
            if i.wires == Wires([0, 2, 1]) and i.name == "Toffoli":
                mats.append(
                    np.array(
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
                )
            elif i.wires == Wires([0, 1, 2]) and i.name == "Toffoli":
                mats.append(
                    np.array(
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                        ]
                    )
                )

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_swap_decomposition(self):
        """Tests the swap operator produces the correct output"""
        opr = qml.SWAP(wires=[0, 1])
        decomp = opr.decomposition([0, 1])

        mat = []
        for op in reversed(decomp):
            if isinstance(op, qml.CNOT) and op.wires.tolist() == [0, 1]:
                mat.append(CNOT)
            elif isinstance(op, qml.CNOT) and op.wires.tolist() == [1, 0]:
                mat.append(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))

        decomposed_matrix = np.linalg.multi_dot(mat)

        assert np.allclose(decomposed_matrix, opr.matrix)


class TestEigenval:
    def test_iswap_eigenval(self):
        """Tests that the ISWAP eigenvalue matches the numpy eigenvalues of the ISWAP matrix"""
        op = qml.ISWAP(wires=[0, 1])
        exp = np.linalg.eigvals(op.matrix)
        res = op.eigvals
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("siswap_op", [qml.SISWAP, qml.SQISW])
    def test_siswap_eigenval(self, siswap_op):
        """Tests that the ISWAP eigenvalue matches the numpy eigenvalues of the ISWAP matrix"""
        op = siswap_op(wires=[0, 1])
        exp = np.linalg.eigvals(op.matrix)
        res = op.eigvals
        assert np.allclose(res, exp)


class TestMultiControlledX:
    """Tests for the MultiControlledX"""

    X = np.array([[0, 1], [1, 0]])

    @pytest.mark.parametrize(
        "control_wires,wires,control_values,expected_error_message",
        [
            ([0, 1], 2, "ab", "String of control values can contain only '0' or '1'."),
            ([0, 1], 2, "011", "Length of control bit string must equal number of control wires."),
            ([0, 1], 2, [0, 1], "Alternative control values must be passed as a binary string."),
            (
                [0, 1],
                [2, 3],
                "10",
                "MultiControlledX accepts a single target wire.",
            ),
        ],
    )
    def test_invalid_mixed_polarity_controls(
        self, control_wires, wires, control_values, expected_error_message
    ):
        """Test if MultiControlledX properly handles invalid mixed-polarity
        control values."""
        target_wires = Wires(wires)

        with pytest.raises(ValueError, match=expected_error_message):
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wires, control_values=control_values
            )

    @pytest.mark.parametrize(
        "control_wires,wires,control_values",
        [
            ([0], 1, "0"),
            ([0, 1], 2, "00"),
            ([0, 1], 2, "10"),
            ([1, 0], 2, "10"),
            ([0, 1], 2, "11"),
            ([0, 2], 1, "10"),
            ([1, 2, 0], 3, "100"),
            ([1, 0, 2, 4], 3, "1001"),
            ([0, 1, 2, 5, 3, 6], 4, "100001"),
        ],
    )
    def test_mixed_polarity_controls(self, control_wires, wires, control_values):
        """Test if MultiControlledX properly applies mixed-polarity
        control values."""
        target_wires = Wires(wires)

        dev = qml.device("default.qubit", wires=len(control_wires + target_wires))

        # Pick random starting state for the control and target qubits
        control_state_weights = np.random.normal(size=(2 ** (len(control_wires) + 1) - 2))
        target_state_weights = np.random.normal(size=(2 ** (len(target_wires) + 1) - 2))

        @qml.qnode(dev)
        def circuit_mpmct():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wires, control_values=control_values
            )
            return qml.state()

        # The result of applying the mixed-polarity gate should be the same as
        # if we conjugated the specified control wires with Pauli X and applied the
        # "regular" ControlledQubitUnitary in between.

        x_locations = [x for x in range(len(control_values)) if control_values[x] == "0"]

        @qml.qnode(dev)
        def circuit_pauli_x():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            qml.ControlledQubitUnitary(X, control_wires=control_wires, wires=target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            return qml.state()

        mpmct_state = circuit_mpmct()
        pauli_x_state = circuit_pauli_x()

        assert np.allclose(mpmct_state, pauli_x_state)

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 6))
    def test_decomposition_with_many_workers(self, n_ctrl_wires):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there are many work wires."""
        control_wires = range(n_ctrl_wires)
        target_wire = n_ctrl_wires
        work_wires = range(n_ctrl_wires + 1, 2 * n_ctrl_wires + 1)

        dev = qml.device("default.qubit", wires=2 * n_ctrl_wires + 1)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX._decomposition_with_many_workers(
                control_wires, target_wire, work_wires
            )
        assert all(isinstance(op, qml.Toffoli) for op in tape.operations)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(control_wires=control_wires, wires=target_wire).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array([f(b) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 6))
    def test_decomposition_with_one_worker(self, n_ctrl_wires):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there is one work wire."""
        control_wires = Wires(range(n_ctrl_wires))
        target_wire = n_ctrl_wires
        work_wires = n_ctrl_wires + 1

        dev = qml.device("default.qubit", wires=n_ctrl_wires + 2)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX._decomposition_with_one_worker(
                control_wires, target_wire, work_wires
            )
        tape = tape.expand(depth=1)
        assert all(
            isinstance(op, qml.Toffoli) or isinstance(op, qml.CNOT) for op in tape.operations
        )

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(control_wires=control_wires, wires=target_wire).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array([f(b) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    def test_not_enough_workers(self):
        """Test that a ValueError is raised when more than 2 control wires are to be decomposed with
        no work wires supplied"""
        control_wires = range(3)
        target_wire = 4
        op = qml.MultiControlledX(control_wires=control_wires, wires=target_wire)

        match = (
            f"At least one work wire is required to decompose operation: {re.escape(op.__repr__())}"
        )
        with pytest.raises(ValueError, match=match):
            op.decomposition()

    def test_not_unique_wires(self):
        """Test that a ValueError is raised when work_wires is not complementary to control_wires"""
        control_wires = range(3)
        target_wire = 4
        work_wires = range(2)
        with pytest.raises(
            ValueError, match="The work wires must be different from the control and target wires"
        ):
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wire, work_wires=work_wires
            )

    @pytest.mark.parametrize("control_val", ["0", "1"])
    @pytest.mark.parametrize("n_ctrl_wires", range(1, 6))
    def test_decomposition_with_flips(self, n_ctrl_wires, control_val, mocker):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on varying the control values."""
        control_values = control_val * n_ctrl_wires
        control_wires = range(n_ctrl_wires)
        target_wire = n_ctrl_wires
        work_wires = range(n_ctrl_wires + 1, 2 * n_ctrl_wires + 1)

        spy = mocker.spy(qml.MultiControlledX, "decomposition")
        dev = qml.device("default.qubit", wires=2 * n_ctrl_wires + 1)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX(
                control_wires=control_wires,
                wires=target_wire,
                work_wires=work_wires,
                control_values=control_values,
            )
        tape = tape.expand(depth=1)
        assert all(not isinstance(op, qml.MultiControlledX) for op in tape.operations)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wire, control_values=control_values
            ).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array([f(b) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]).T
        spy.assert_called()
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    def test_decomposition_with_custom_wire_labels(self, mocker):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on using custom wire labels."""
        n_ctrl_wires = 4
        control_wires = [-1, "alice", 42, 3.14]
        target_wire = ["bob"]
        work_wires = ["charlie"]
        all_wires = control_wires + target_wire + work_wires

        spy = mocker.spy(qml.MultiControlledX, "decomposition")
        dev = qml.device("default.qubit", wires=all_wires)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wire, work_wires=work_wires
            )
        tape = tape.expand(depth=2)
        assert all(not isinstance(op, qml.MultiControlledX) for op in tape.operations)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=control_wires + target_wire)
            qml.MultiControlledX(control_wires=control_wires, wires=target_wire).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=control_wires + target_wire)

        u = np.array([f(b) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]).T
        spy.assert_called()
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    def test_worker_state_unperturbed(self, mocker):
        """Test that the state of the worker wires is unperturbed after the decomposition has used
        them. To do this, a random state over all the qubits (control, target and workers) is
        loaded and U^dagger U(decomposed) is applied. If the workers are uncomputed, the output
        state will be the same as the input."""
        control_wires = range(4)
        target_wire = 4
        worker_wires = [5, 6]
        n_all_wires = 7

        rnd_state = unitary_group.rvs(2 ** n_all_wires, random_state=1)[0]
        spy = mocker.spy(qml.MultiControlledX, "decomposition")
        dev = qml.device("default.qubit", wires=n_all_wires)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wire, work_wires=worker_wires
            )
        tape = tape.expand(depth=1)
        assert all(not isinstance(op, qml.MultiControlledX) for op in tape.operations)

        @qml.qnode(dev)
        def f():
            qml.QubitStateVector(rnd_state, wires=range(n_all_wires))
            qml.MultiControlledX(control_wires=control_wires, wires=target_wire).inv()
            for op in tape.operations:
                op.queue()
            return qml.state()

        assert np.allclose(f(), rnd_state)
        spy.assert_called()


label_data = [
    (qml.Hadamard(0), "H", "H"),
    (qml.PauliX(0), "X", "X"),
    (qml.PauliY(0), "Y", "Y"),
    (qml.PauliZ(0), "Z", "Z"),
    (qml.S(wires=0), "S", "S⁻¹"),
    (qml.T(wires=0), "T", "T⁻¹"),
    (qml.SX(wires=0), "SX", "SX⁻¹"),
    (qml.CNOT(wires=(0, 1)), "⊕", "⊕"),
    (qml.CZ(wires=(0, 1)), "Z", "Z"),
    (qml.CY(wires=(0, 1)), "Y", "Y"),
    (qml.SWAP(wires=(0, 1)), "SWAP", "SWAP⁻¹"),
    (qml.ISWAP(wires=(0, 1)), "ISWAP", "ISWAP⁻¹"),
    (qml.SISWAP(wires=(0, 1)), "SISWAP", "SISWAP⁻¹"),
    (qml.SQISW(wires=(0, 1)), "SISWAP", "SISWAP⁻¹"),
    (qml.CSWAP(wires=(0, 1, 2)), "SWAP", "SWAP"),
    (qml.Toffoli(wires=(0, 1, 2)), "⊕", "⊕"),
    (qml.MultiControlledX(control_wires=(0, 1, 2), wires=(3)), "⊕", "⊕"),
]


@pytest.mark.parametrize("op, label1, label2", label_data)
def test_label_method(op, label1, label2):
    assert op.label() == label1
    assert op.label(decimals=2) == label1

    op.inv()
    assert op.label() == label2


control_data = [
    (qml.Hadamard(0), Wires([])),
    (qml.PauliX(0), Wires([])),
    (qml.PauliY(0), Wires([])),
    (qml.S(wires=0), Wires([])),
    (qml.T(wires=0), Wires([])),
    (qml.SX(wires=0), Wires([])),
    (qml.SWAP(wires=(0, 1)), Wires([])),
    (qml.ISWAP(wires=(0, 1)), Wires([])),
    (qml.SISWAP(wires=(0, 1)), Wires([])),
    (qml.CNOT(wires=(0, 1)), Wires(0)),
    (qml.CZ(wires=(0, 1)), Wires(0)),
    (qml.CY(wires=(0, 1)), Wires(0)),
    (qml.CSWAP(wires=(0, 1, 2)), Wires([0])),
    (qml.Toffoli(wires=(0, 1, 2)), Wires([0, 1])),
    (qml.MultiControlledX(control_wires=[0, 1, 2, 3], wires=4), Wires([0, 1, 2, 3])),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test ``control_wires`` attribute for non-parametrized operations."""

    assert op.control_wires == control_wires
