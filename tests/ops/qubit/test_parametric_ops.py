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
Unit tests for the available built-in parametric qubit operations.
"""
import pytest
import copy
import numpy as np
from pennylane import numpy as npp

import pennylane as qml
from pennylane.wires import Wires

from gate_data import ControlledPhaseShift, Z

PARAMETRIZED_OPERATIONS = [
    qml.RX(0.123, wires=0),
    qml.RY(1.434, wires=0),
    qml.RZ(2.774, wires=0),
    qml.PauliRot(0.123, "Y", wires=0),
    qml.IsingXX(0.123, wires=[0, 1]),
    qml.IsingYY(0.123, wires=[0, 1]),
    qml.IsingZZ(0.123, wires=[0, 1]),
    qml.Rot(0.123, 0.456, 0.789, wires=0),
    qml.PhaseShift(2.133, wires=0),
    qml.ControlledPhaseShift(1.777, wires=[0, 2]),
    qml.CPhase(1.777, wires=[0, 2]),
    qml.MultiRZ(0.112, wires=[1, 2, 3]),
    qml.CRX(0.836, wires=[2, 3]),
    qml.CRY(0.721, wires=[2, 3]),
    qml.CRZ(0.554, wires=[2, 3]),
    qml.U1(0.123, wires=0),
    qml.U2(3.556, 2.134, wires=0),
    qml.U3(2.009, 1.894, 0.7789, wires=0),
    qml.CRot(0.123, 0.456, 0.789, wires=[0, 1]),
    qml.QubitUnitary(np.eye(2) * 1j, wires=0),
    qml.DiagonalQubitUnitary(np.array([1.0, 1.0j]), wires=1),
    qml.ControlledQubitUnitary(np.eye(2) * 1j, wires=[0], control_wires=[2]),
    qml.MultiControlledX(control_wires=[0, 1], wires=2, control_values="01"),
    qml.MultiControlledX(wires=[0, 1, 2], control_values="01"),
    qml.SingleExcitation(0.123, wires=[0, 3]),
    qml.SingleExcitationPlus(0.123, wires=[0, 3]),
    qml.SingleExcitationMinus(0.123, wires=[0, 3]),
    qml.DoubleExcitation(0.123, wires=[0, 1, 2, 3]),
    qml.DoubleExcitationPlus(0.123, wires=[0, 1, 2, 3]),
    qml.DoubleExcitationMinus(0.123, wires=[0, 1, 2, 3]),
]

NON_PARAMETRIZED_OPERATIONS = [
    qml.S(wires=0),
    qml.SX(wires=0),
    qml.T(wires=0),
    qml.CNOT(wires=[0, 1]),
    qml.CZ(wires=[0, 1]),
    qml.CY(wires=[0, 1]),
    qml.SWAP(wires=[0, 1]),
    qml.ISWAP(wires=[0, 1]),
    qml.SISWAP(wires=[0, 1]),
    qml.SQISW(wires=[0, 1]),
    qml.CSWAP(wires=[0, 1, 2]),
    qml.Toffoli(wires=[0, 1, 2]),
    qml.Hadamard(wires=0),
    qml.PauliX(wires=0),
    qml.PauliZ(wires=0),
    qml.PauliY(wires=0),
    qml.MultiControlledX(control_wires=[0, 1], wires=2, control_values="01"),
    qml.QubitSum(wires=[0, 1, 2]),
]

ALL_OPERATIONS = NON_PARAMETRIZED_OPERATIONS + PARAMETRIZED_OPERATIONS


class TestOperations:
    @pytest.mark.parametrize("op", ALL_OPERATIONS)
    def test_parametrized_op_copy(self, op, tol):
        """Tests that copied parametrized ops function as expected"""
        copied_op = copy.copy(op)
        np.testing.assert_allclose(op.get_matrix(), copied_op.get_matrix(), atol=tol)

        op.inv()
        copied_op2 = copy.copy(op)
        np.testing.assert_allclose(op.get_matrix(), copied_op2.get_matrix(), atol=tol)
        op.inv()

    @pytest.mark.parametrize("op", ALL_OPERATIONS)
    def test_adjoint_unitaries(self, op, tol):
        op_d = op.adjoint()
        res1 = np.dot(op.get_matrix(), op_d.get_matrix())
        res2 = np.dot(op_d.get_matrix(), op.get_matrix())
        np.testing.assert_allclose(res1, np.eye(2 ** len(op.wires)), atol=tol)
        np.testing.assert_allclose(res2, np.eye(2 ** len(op.wires)), atol=tol)
        assert op.wires == op_d.wires


class TestParameterFrequencies:
    @pytest.mark.parametrize("op", PARAMETRIZED_OPERATIONS)
    def test_parameter_frequencies_match_generator(self, op, tol):
        if not qml.operation.has_gen(op):
            pytest.skip(f"Operation {op.name} does not have a generator defined to test against.")

        gen = op.generator()

        try:
            mat = gen.get_matrix()
        except (AttributeError, qml.operation.MatrixUndefinedError):

            if isinstance(gen, qml.Hamiltonian):
                mat = qml.utils.sparse_hamiltonian(gen).toarray()
            elif isinstance(gen, qml.SparseHamiltonian):
                mat = gen.sparse_matrix().toarray()
            else:
                pytest.skip(f"Operation {op.name}'s generator does not define a matrix.")

        gen_eigvals = np.round(np.linalg.eigvalsh(mat), 8)
        freqs_from_gen = qml.gradients.eigvals_to_frequencies(tuple(gen_eigvals))

        freqs = op.parameter_frequencies
        assert np.allclose(freqs, freqs_from_gen, atol=tol)


class TestDecompositions:
    def test_phase_decomposition(self, tol):
        """Tests that the decomposition of the Phase gate is correct"""
        phi = 0.3
        op = qml.PhaseShift(phi, wires=0)
        res = op.decomposition()

        assert len(res) == 1

        assert res[0].name == "RZ"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == 0.3

        decomposed_matrix = res[0].get_matrix()
        global_phase = (
            decomposed_matrix[op.get_matrix() != 0] / op.get_matrix()[op.get_matrix() != 0]
        )[0]

        assert np.allclose(decomposed_matrix, global_phase * op.get_matrix(), atol=tol, rtol=0)

    def test_Rot_decomposition(self):
        """Test the decomposition of Rot."""
        phi = 0.432
        theta = 0.654
        omega = -5.43

        ops1 = qml.Rot.compute_decomposition(phi, theta, omega, wires=0)
        ops2 = qml.Rot(phi, theta, omega, wires=0).decomposition()

        assert len(ops1) == len(ops2) == 3

        classes = [qml.RZ, qml.RY, qml.RZ]
        params = [[phi], [theta], [omega]]

        for ops in [ops1, ops2]:
            for c, p, op in zip(classes, params, ops):
                assert isinstance(op, c)
                assert op.parameters == p

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
                assert op.parameters == p
                assert op.wires == w

    def test_CRZ_decomposition(self):
        """Test the decomposition for CRZ."""
        phi = 0.432

        ops1 = qml.CRZ.compute_decomposition(phi, wires=[0, 1])
        ops2 = qml.CRZ(phi, wires=(0, 1)).decomposition()

        classes = [qml.PhaseShift, qml.CNOT, qml.PhaseShift, qml.CNOT]
        params = [[phi / 2], [], [-phi / 2], []]
        wires = [Wires(1), Wires((0, 1)), Wires(1), Wires((0, 1))]

        for ops in [ops1, ops2]:
            for op, c, p, w in zip(ops, classes, params, wires):
                assert isinstance(op, c)
                assert op.parameters == p
                assert op.wires == w

    @pytest.mark.parametrize("phi, theta, omega", [[0.5, 0.6, 0.7], [0.1, -0.4, 0.7], [-10, 5, -1]])
    def test_CRot_decomposition(self, tol, phi, theta, omega, monkeypatch):
        """Tests that the decomposition of the CRot gate is correct"""
        op = qml.CRot(phi, theta, omega, wires=[0, 1])
        res = op.decomposition()

        mats = []
        for i in reversed(res):
            if len(i.wires) == 1:
                mats.append(np.kron(np.eye(2), i.get_matrix()))
            else:
                mats.append(i.get_matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.get_matrix(), atol=tol, rtol=0)

    def test_U1_decomposition(self):
        """Test the decomposition for U1."""
        phi = 0.432
        res = qml.U1(phi, wires=0).decomposition()
        res2 = qml.U1.compute_decomposition(phi, wires=0)

        assert len(res) == len(res2) == 1
        assert res[0].name == res2[0].name == "PhaseShift"
        assert res[0].parameters == res2[0].parameters == [phi]

    def test_U2_decomposition(self):
        """Test the decomposition for U2."""
        phi = 0.432
        lam = 0.654

        ops1 = qml.U2.compute_decomposition(phi, lam, wires=0)
        ops2 = qml.U2(phi, lam, wires=0).decomposition()

        classes = [qml.Rot, qml.PhaseShift, qml.PhaseShift]
        params = [[lam, np.pi / 2, -lam], [lam], [phi]]

        for ops in [ops1, ops2]:
            for op, c, p in zip(ops, classes, params):
                assert isinstance(op, c)
                assert op.parameters == p

    def test_U3_decomposition(self):
        """Test the decomposition for U3."""
        theta = 0.654
        phi = 0.432
        lam = 0.654

        ops1 = qml.U3.compute_decomposition(theta, phi, lam, wires=0)
        ops2 = qml.U3(theta, phi, lam, wires=0).decomposition()

        classes = [qml.Rot, qml.PhaseShift, qml.PhaseShift]
        params = [[lam, theta, -lam], [lam], [phi]]

        for ops in [ops1, ops2]:
            for op, c, p in zip(ops, classes, params):
                assert isinstance(op, c)
                assert op.parameters == p

    def test_isingxx_decomposition(self, tol):
        """Tests that the decomposition of the IsingXX gate is correct"""
        param = 0.1234
        op = qml.IsingXX(param, wires=[3, 2])
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].wires == Wires([3, 2])
        assert res[1].wires == Wires([3])
        assert res[2].wires == Wires([3, 2])

        assert res[0].name == "CNOT"
        assert res[1].name == "RX"
        assert res[2].name == "CNOT"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([3]):
                # RX gate
                mats.append(np.kron(i.get_matrix(), np.eye(2)))
            else:
                mats.append(i.get_matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.get_matrix(), atol=tol, rtol=0)

    def test_isingyy_decomposition(self, tol):
        """Tests that the decomposition of the IsingYY gate is correct"""
        param = 0.1234
        op = qml.IsingYY(param, wires=[3, 2])
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].wires == Wires([3, 2])
        assert res[1].wires == Wires([3])
        assert res[2].wires == Wires([3, 2])

        assert res[0].name == "CY"
        assert res[1].name == "RY"
        assert res[2].name == "CY"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([3]):
                # RY gate
                mats.append(np.kron(i.get_matrix(), np.eye(2)))
            else:
                mats.append(i.get_matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.get_matrix(), atol=tol, rtol=0)

    def test_isingzz_decomposition(self, tol):
        """Tests that the decomposition of the IsingZZ gate is correct"""
        param = 0.1234
        op = qml.IsingZZ(param, wires=[3, 2])
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].wires == Wires([3, 2])
        assert res[1].wires == Wires([2])
        assert res[2].wires == Wires([3, 2])

        assert res[0].name == "CNOT"
        assert res[1].name == "RZ"
        assert res[2].name == "CNOT"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([2]):
                # RZ gate
                mats.append(np.kron(np.eye(2), i.get_matrix()))
            else:
                mats.append(i.get_matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.get_matrix(), atol=tol, rtol=0)

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
                mats.append(np.kron(i.get_matrix(), np.eye(4)))
            elif i.wires.tolist() == [1]:
                mats.append(np.kron(np.eye(2), np.kron(i.get_matrix(), np.eye(2))))
            elif i.wires.tolist() == [2]:
                mats.append(np.kron(np.eye(4), i.get_matrix()))
            elif isinstance(i, qml.CNOT) and i.wires.tolist() == [0, 1]:
                mats.append(np.kron(i.get_matrix(), np.eye(2)))
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


class TestMatrix:
    def test_phase_shift(self, tol):
        """Test phase shift is correct"""

        # test identity for theta=0
        assert np.allclose(qml.PhaseShift.compute_matrix(0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.U1.compute_matrix(0), np.identity(2), atol=tol, rtol=0)

        # test arbitrary phase shift
        phi = 0.5432
        expected = np.array([[1, 0], [0, np.exp(1j * phi)]])
        assert np.allclose(qml.PhaseShift.compute_matrix(phi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.U1.compute_matrix(phi), expected, atol=tol, rtol=0)

    def test_rx(self, tol):
        """Test x rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.RX.compute_matrix(0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.RX(0, wires=0).get_matrix(), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
        assert np.allclose(qml.RX.compute_matrix(np.pi / 2), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RX(np.pi / 2, wires=0).get_matrix(), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = -1j * np.array([[0, 1], [1, 0]])
        assert np.allclose(qml.RX.compute_matrix(np.pi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RX(np.pi, wires=0).get_matrix(), expected, atol=tol, rtol=0)

    def test_ry(self, tol):
        """Test y rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.RY.compute_matrix(0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.RY(0, wires=0).get_matrix(), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        assert np.allclose(qml.RY.compute_matrix(np.pi / 2), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RY(np.pi / 2, wires=0).get_matrix(), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[0, -1], [1, 0]])
        assert np.allclose(qml.RY.compute_matrix(np.pi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RY(np.pi, wires=0).get_matrix(), expected, atol=tol, rtol=0)

    def test_rz(self, tol):
        """Test z rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.RZ.compute_matrix(0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.RZ(0, wires=0).get_matrix(), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.diag(np.exp([-1j * np.pi / 4, 1j * np.pi / 4]))
        assert np.allclose(qml.RZ.compute_matrix(np.pi / 2), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RZ(np.pi / 2, wires=0).get_matrix(), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        assert np.allclose(qml.RZ.compute_matrix(np.pi), -1j * Z, atol=tol, rtol=0)
        assert np.allclose(qml.RZ(np.pi, wires=0).get_matrix(), -1j * Z, atol=tol, rtol=0)

    def test_isingxx(self, tol):
        """Test that the IsingXX operation is correct"""
        assert np.allclose(qml.IsingXX.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingXX(0, wires=[0, 1]).get_matrix(), np.identity(4), atol=tol, rtol=0
        )

        def get_expected(theta):
            expected = np.array(np.diag([np.cos(theta / 2)] * 4), dtype=np.complex128)
            sin_coeff = -1j * np.sin(theta / 2)
            expected[3, 0] = sin_coeff
            expected[2, 1] = sin_coeff
            expected[1, 2] = sin_coeff
            expected[0, 3] = sin_coeff
            return expected

        param = np.pi / 2
        assert np.allclose(qml.IsingXX.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingXX(param, wires=[0, 1]).get_matrix(), get_expected(param), atol=tol, rtol=0
        )

        param = np.pi
        assert np.allclose(qml.IsingXX.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingXX(param, wires=[0, 1]).get_matrix(), get_expected(param), atol=tol, rtol=0
        )

    def test_isingzz(self, tol):
        """Test that the IsingZZ operation is correct"""
        assert np.allclose(qml.IsingZZ.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingZZ(0, wires=[0, 1]).get_matrix(), np.identity(4), atol=tol, rtol=0
        )
        assert np.allclose(
            qml.IsingZZ.compute_eigvals(0), np.diagonal(np.identity(4)), atol=tol, rtol=0
        )

        def get_expected(theta):
            neg_imag = np.exp(-1j * theta / 2)
            plus_imag = np.exp(1j * theta / 2)
            expected = np.array(
                np.diag([neg_imag, plus_imag, plus_imag, neg_imag]), dtype=np.complex128
            )
            return expected

        param = np.pi / 2
        assert np.allclose(qml.IsingZZ.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingZZ(param, wires=[0, 1]).get_matrix(), get_expected(param), atol=tol, rtol=0
        )
        assert np.allclose(
            qml.IsingZZ.compute_eigvals(param), np.diagonal(get_expected(param)), atol=tol, rtol=0
        )

        param = np.pi
        assert np.allclose(qml.IsingZZ.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingZZ(param, wires=[0, 1]).get_matrix(), get_expected(param), atol=tol, rtol=0
        )
        assert np.allclose(
            qml.IsingZZ.compute_eigvals(param), np.diagonal(get_expected(param)), atol=tol, rtol=0
        )

    def test_isingzz_matrix_tf(self, tol):
        """Tests the matrix representation for IsingZZ for tensorflow, since the method contains
        different logic for this framework"""
        tf = pytest.importorskip("tensorflow")

        def get_expected(theta):
            neg_imag = np.exp(-1j * theta / 2)
            plus_imag = np.exp(1j * theta / 2)
            expected = np.array(
                np.diag([neg_imag, plus_imag, plus_imag, neg_imag]), dtype=np.complex128
            )
            return expected

        param = tf.Variable(np.pi)
        assert np.allclose(qml.IsingZZ.compute_matrix(param), get_expected(np.pi), atol=tol, rtol=0)

    def test_Rot(self, tol):
        """Test arbitrary single qubit rotation is correct"""

        # test identity for phi,theta,omega=0
        assert np.allclose(qml.Rot.compute_matrix(0, 0, 0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.Rot(0, 0, 0, wires=0).get_matrix(), np.identity(2), atol=tol, rtol=0)

        # expected result
        def arbitrary_rotation(x, y, z):
            """arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [np.exp(-0.5j * (x + z)) * c, -np.exp(0.5j * (x - z)) * s],
                    [np.exp(-0.5j * (x - z)) * s, np.exp(0.5j * (x + z)) * c],
                ]
            )

        a, b, c = 0.432, -0.152, 0.9234
        assert np.allclose(
            qml.Rot.compute_matrix(a, b, c), arbitrary_rotation(a, b, c), atol=tol, rtol=0
        )
        assert np.allclose(
            qml.Rot(a, b, c, wires=0).get_matrix(), arbitrary_rotation(a, b, c), atol=tol, rtol=0
        )

    def test_CRx(self, tol):
        """Test controlled x rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.CRX.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.CRX(0, wires=[0, 1]).get_matrix(), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), -1j / np.sqrt(2)],
                [0, 0, -1j / np.sqrt(2), 1 / np.sqrt(2)],
            ]
        )
        assert np.allclose(qml.CRX.compute_matrix(np.pi / 2), expected, atol=tol, rtol=0)
        assert np.allclose(
            qml.CRX(np.pi / 2, wires=[0, 1]).get_matrix(), expected, atol=tol, rtol=0
        )

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]])
        assert np.allclose(qml.CRX.compute_matrix(np.pi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.CRX(np.pi, wires=[0, 1]).get_matrix(), expected, atol=tol, rtol=0)

    def test_CRY(self, tol):
        """Test controlled y rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.CRY.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.CRY(0, wires=[0, 1]).get_matrix(), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            ]
        )
        assert np.allclose(qml.CRY.compute_matrix(np.pi / 2), expected, atol=tol, rtol=0)
        assert np.allclose(
            qml.CRY(np.pi / 2, wires=[0, 1]).get_matrix(), expected, atol=tol, rtol=0
        )

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        assert np.allclose(qml.CRY.compute_matrix(np.pi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.CRY(np.pi, wires=[0, 1]).get_matrix(), expected, atol=tol, rtol=0)

    def test_CRZ(self, tol):
        """Test controlled z rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.CRZ.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(0, wires=[0, 1]).get_matrix(), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * np.pi / 4), 0],
                [0, 0, 0, np.exp(1j * np.pi / 4)],
            ]
        )
        assert np.allclose(qml.CRZ.compute_matrix(np.pi / 2), expected, atol=tol, rtol=0)
        assert np.allclose(
            qml.CRZ(np.pi / 2, wires=[0, 1]).get_matrix(), expected, atol=tol, rtol=0
        )

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1j]])
        assert np.allclose(qml.CRZ.compute_matrix(np.pi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.CRZ(np.pi, wires=[0, 1]).get_matrix(), expected, atol=tol, rtol=0)

    def test_CRot(self, tol):
        """Test controlled arbitrary rotation is correct"""

        # test identity for phi,theta,omega=0
        assert np.allclose(qml.CRot.compute_matrix(0, 0, 0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(
            qml.CRot(0, 0, 0, wires=[0, 1]).get_matrix(), np.identity(4), atol=tol, rtol=0
        )

        # test identity for phi,theta,omega=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        assert np.allclose(qml.CRot.compute_matrix(np.pi, np.pi, np.pi), expected, atol=tol, rtol=0)
        assert np.allclose(
            qml.CRot(np.pi, np.pi, np.pi, wires=[0, 1]).get_matrix(), expected, atol=tol, rtol=0
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
            qml.CRot(a, b, c, wires=[0, 1]).get_matrix(),
            arbitrary_Crotation(a, b, c),
            atol=tol,
            rtol=0,
        )

    def test_U2_gate(self, tol):
        """Test U2 gate matrix matches the documentation"""
        phi = 0.432
        lam = -0.12
        expected = np.array(
            [[1, -np.exp(1j * lam)], [np.exp(1j * phi), np.exp(1j * (phi + lam))]]
        ) / np.sqrt(2)
        assert np.allclose(qml.U2.compute_matrix(phi, lam), expected, atol=tol, rtol=0)
        assert np.allclose(qml.U2(phi, lam, wires=[0]).get_matrix(), expected, atol=tol, rtol=0)

    def test_U3_gate(self, tol):
        """Test U3 gate matrix matches the documentation"""
        theta = 0.65
        phi = 0.432
        lam = -0.12

        expected = np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lam)) * np.cos(theta / 2),
                ],
            ]
        )

        assert np.allclose(qml.U3.compute_matrix(theta, phi, lam), expected, atol=tol, rtol=0)
        assert np.allclose(
            qml.U3(theta, phi, lam, wires=[0]).get_matrix(), expected, atol=tol, rtol=0
        )

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    @pytest.mark.parametrize("cphase_op", [qml.ControlledPhaseShift, qml.CPhase])
    def test_controlled_phase_shift_matrix_and_eigvals(self, phi, cphase_op):
        """Tests that the ControlledPhaseShift and CPhase operation calculate the correct matrix and
        eigenvalues"""
        op = cphase_op(phi, wires=[0, 1])
        res = op.get_matrix()
        exp = ControlledPhaseShift(phi)
        assert np.allclose(res, exp)

        res = op.get_eigvals()
        assert np.allclose(res, np.diag(exp))


class TestGrad:
    device_methods = [
        ["default.qubit", "finite-diff"],
        ["default.qubit", "parameter-shift"],
        ["default.qubit", "backprop"],
        ["default.qubit", "adjoint"],
    ]

    phis = [0.1, 0.2, 0.3]

    configuration = []

    for phi in phis:
        for device, method in device_methods:
            configuration.append([device, method, npp.array(phi, requires_grad=True)])

    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingxx_autograd_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingXX."""
        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = npp.array([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
        norm = np.linalg.norm(init_state)
        init_state /= norm

        @qml.qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(phi):
            qml.QubitStateVector(init_state, wires=[0, 1])
            qml.IsingXX(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        expected = (
            0.5
            * (1 / norm**2)
            * (
                -np.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * np.sin(phi / 2)
                * np.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        res = qml.grad(circuit)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingyy_autograd_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingYY."""
        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = npp.array([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
        norm = np.linalg.norm(init_state)
        init_state /= norm

        @qml.qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(phi):
            qml.QubitStateVector(init_state, wires=[0, 1])
            qml.IsingYY(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        expected = (
            0.5
            * (1 / norm**2)
            * (
                -np.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * np.sin(phi / 2)
                * np.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        res = qml.grad(circuit)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingzz_autograd_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingZZ."""
        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = npp.array([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
        norm = np.linalg.norm(init_state)
        init_state /= norm

        @qml.qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(phi):
            qml.QubitStateVector(init_state, wires=[0, 1])
            qml.IsingZZ(phi, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        phi = npp.array(0.1, requires_grad=True)

        expected = (1 / norm**2) * (-2 * (psi_0 * psi_2 + psi_1 * psi_3) * np.sin(phi))

        res = qml.grad(circuit)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingxx_jax_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingXX."""

        if diff_method in {"finite-diff"}:
            pytest.skip("Test does not support finite-diff")

        if diff_method in {"parameter-shift"}:
            pytest.skip("Test does not support parameter-shift")

        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = jnp.array([psi_0, psi_1, psi_2, psi_3])
        norm = jnp.linalg.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method, interface="jax")
        def circuit(phi):
            qml.QubitStateVector(init_state, wires=[0, 1])
            qml.IsingXX(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi = jnp.array(0.1)

        expected = (
            0.5
            * (1 / norm**2)
            * (
                -np.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * np.sin(phi / 2)
                * np.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        res = jax.grad(circuit, argnums=0)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingyy_jax_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingYY."""

        if diff_method in {"finite-diff"}:
            pytest.skip("Test does not support finite-diff")

        if diff_method in {"parameter-shift"}:
            pytest.skip("Test does not support parameter-shift")

        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = jnp.array([psi_0, psi_1, psi_2, psi_3])
        norm = jnp.linalg.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method, interface="jax")
        def circuit(phi):
            qml.QubitStateVector(init_state, wires=[0, 1])
            qml.IsingYY(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi = jnp.array(0.1)

        expected = (
            0.5
            * (1 / norm**2)
            * (
                -np.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * np.sin(phi / 2)
                * np.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        res = jax.grad(circuit, argnums=0)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingzz_jax_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingZZ."""

        if diff_method in {"finite-diff"}:
            pytest.skip("Test does not support finite-diff")

        if diff_method in {"parameter-shift"}:
            pytest.skip("Test does not support parameter-shift")

        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = jnp.array([psi_0, psi_1, psi_2, psi_3])
        norm = jnp.linalg.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method, interface="jax")
        def circuit(phi):
            qml.QubitStateVector(init_state, wires=[0, 1])
            qml.IsingZZ(phi, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        phi = jnp.array(0.1)

        expected = (1 / norm**2) * (-2 * (psi_0 * psi_2 + psi_1 * psi_3) * np.sin(phi))

        res = jax.grad(circuit, argnums=0)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingxx_tf_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingXX."""
        tf = pytest.importorskip("tensorflow", minversion="2.1")

        dev = qml.device(dev_name, wires=2)

        psi_0 = tf.Variable(0.1, dtype=tf.complex128)
        psi_1 = tf.Variable(0.2, dtype=tf.complex128)
        psi_2 = tf.Variable(0.3, dtype=tf.complex128)
        psi_3 = tf.Variable(0.4, dtype=tf.complex128)

        init_state = tf.Variable([psi_0, psi_1, psi_2, psi_3], dtype=tf.complex128)
        norm = tf.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, interface="tf", diff_method=diff_method)
        def circuit(phi):
            qml.QubitStateVector(init_state, wires=[0, 1])
            qml.IsingXX(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi = tf.Variable(0.1, dtype=tf.complex128)

        expected = (
            0.5
            * (1 / norm**2)
            * (
                -tf.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * tf.sin(phi / 2)
                * tf.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        with tf.GradientTape() as tape:
            result = circuit(phi)
        res = tape.gradient(result, phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingyy_tf_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingYY."""
        tf = pytest.importorskip("tensorflow", minversion="2.1")

        dev = qml.device(dev_name, wires=2)

        psi_0 = tf.Variable(0.1, dtype=tf.complex128)
        psi_1 = tf.Variable(0.2, dtype=tf.complex128)
        psi_2 = tf.Variable(0.3, dtype=tf.complex128)
        psi_3 = tf.Variable(0.4, dtype=tf.complex128)

        init_state = tf.Variable([psi_0, psi_1, psi_2, psi_3], dtype=tf.complex128)
        norm = tf.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, interface="tf", diff_method=diff_method)
        def circuit(phi):
            qml.QubitStateVector(init_state, wires=[0, 1])
            qml.IsingYY(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi = tf.Variable(0.1, dtype=tf.complex128)

        expected = (
            0.5
            * (1 / norm**2)
            * (
                -tf.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * tf.sin(phi / 2)
                * tf.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        with tf.GradientTape() as tape:
            result = circuit(phi)
        res = tape.gradient(result, phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingzz_tf_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingZZ."""
        tf = pytest.importorskip("tensorflow", minversion="2.1")

        dev = qml.device(dev_name, wires=2)

        psi_0 = tf.Variable(0.1, dtype=tf.complex128)
        psi_1 = tf.Variable(0.2, dtype=tf.complex128)
        psi_2 = tf.Variable(0.3, dtype=tf.complex128)
        psi_3 = tf.Variable(0.4, dtype=tf.complex128)

        init_state = tf.Variable([psi_0, psi_1, psi_2, psi_3], dtype=tf.complex128)
        norm = tf.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, interface="tf", diff_method=diff_method)
        def circuit(phi):
            qml.QubitStateVector(init_state, wires=[0, 1])
            qml.IsingZZ(phi, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        phi = tf.Variable(0.1, dtype=tf.complex128)

        expected = (1 / norm**2) * (-2 * (psi_0 * psi_2 + psi_1 * psi_3) * np.sin(phi))

        with tf.GradientTape() as tape:
            result = circuit(phi)
        res = tape.gradient(result, phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", np.linspace(0, 2 * np.pi, 3))
    def test_qnode_with_rx_and_state_jacobian_jax(self, par, tol):
        """Test the jacobian of a complex valued QNode that contains a rotation
        using the JAX interface."""
        jax = pytest.importorskip("jax")

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def test(x):
            qml.RX(x, wires=[0])
            return qml.state()

        res = jax.jacobian(test, holomorphic=True)(par + 0j)
        expected = -1 / 2 * np.sin(par / 2), -1 / 2 * 1j * np.cos(par / 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)


PAULI_ROT_PARAMETRIC_MATRIX_TEST_DATA = [
    (
        "XY",
        lambda theta: np.array(
            [
                [np.cos(theta / 2), 0, 0, -np.sin(theta / 2)],
                [0, np.cos(theta / 2), np.sin(theta / 2), 0],
                [0, -np.sin(theta / 2), np.cos(theta / 2), 0],
                [np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
            ],
            dtype=complex,
        ),
    ),
    (
        "ZZ",
        lambda theta: np.diag(
            [
                np.exp(-1j * theta / 2),
                np.exp(1j * theta / 2),
                np.exp(1j * theta / 2),
                np.exp(-1j * theta / 2),
            ],
        ),
    ),
    (
        "XI",
        lambda theta: np.array(
            [
                [np.cos(theta / 2), 0, -1j * np.sin(theta / 2), 0],
                [0, np.cos(theta / 2), 0, -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), 0, np.cos(theta / 2), 0],
                [0, -1j * np.sin(theta / 2), 0, np.cos(theta / 2)],
            ],
        ),
    ),
    ("X", qml.RX.compute_matrix),
    ("Y", qml.RY.compute_matrix),
    ("Z", qml.RZ.compute_matrix),
]

PAULI_ROT_MATRIX_TEST_DATA = [
    (
        np.pi,
        "XIZ",
        np.array(
            [
                [0, 0, 0, 0, -1j, 0, 0, 0],
                [0, 0, 0, 0, 0, 1j, 0, 0],
                [0, 0, 0, 0, 0, 0, -1j, 0],
                [0, 0, 0, 0, 0, 0, 0, 1j],
                [-1j, 0, 0, 0, 0, 0, 0, 0],
                [0, 1j, 0, 0, 0, 0, 0, 0],
                [0, 0, -1j, 0, 0, 0, 0, 0],
                [0, 0, 0, 1j, 0, 0, 0, 0],
            ]
        ),
    ),
    (
        np.pi / 3,
        "XYZ",
        np.array(
            [
                [np.sqrt(3) / 2, 0, 0, 0, 0, 0, -(1 / 2), 0],
                [0, np.sqrt(3) / 2, 0, 0, 0, 0, 0, 1 / 2],
                [0, 0, np.sqrt(3) / 2, 0, 1 / 2, 0, 0, 0],
                [0, 0, 0, np.sqrt(3) / 2, 0, -(1 / 2), 0, 0],
                [0, 0, -(1 / 2), 0, np.sqrt(3) / 2, 0, 0, 0],
                [0, 0, 0, 1 / 2, 0, np.sqrt(3) / 2, 0, 0],
                [1 / 2, 0, 0, 0, 0, 0, np.sqrt(3) / 2, 0],
                [0, -(1 / 2), 0, 0, 0, 0, 0, np.sqrt(3) / 2],
            ]
        ),
    ),
]


class TestPauliRot:
    """Test the PauliRot operation."""

    @pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize(
        "pauli_word,expected_matrix",
        PAULI_ROT_PARAMETRIC_MATRIX_TEST_DATA,
    )
    def test_PauliRot_matrix_parametric(self, theta, pauli_word, expected_matrix, tol):
        """Test parametrically that the PauliRot matrix is correct."""

        res = qml.PauliRot.compute_matrix(theta, pauli_word)
        expected = expected_matrix(theta)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "theta,pauli_word,expected_matrix",
        PAULI_ROT_MATRIX_TEST_DATA,
    )
    def test_PauliRot_matrix(self, theta, pauli_word, expected_matrix, tol):
        """Test non-parametrically that the PauliRot matrix is correct."""

        res = qml.PauliRot.compute_matrix(theta, pauli_word)
        expected = expected_matrix

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "theta,pauli_word,compressed_pauli_word,wires,compressed_wires",
        [
            (np.pi, "XIZ", "XZ", [0, 1, 2], [0, 2]),
            (np.pi / 3, "XIYIZI", "XYZ", [0, 1, 2, 3, 4, 5], [0, 2, 4]),
            (np.pi / 7, "IXI", "X", [0, 1, 2], [1]),
            (np.pi / 9, "IIIIIZI", "Z", [0, 1, 2, 3, 4, 5, 6], [5]),
            (np.pi / 11, "XYZIII", "XYZ", [0, 1, 2, 3, 4, 5], [0, 1, 2]),
            (np.pi / 11, "IIIXYZ", "XYZ", [0, 1, 2, 3, 4, 5], [3, 4, 5]),
        ],
    )
    def test_PauliRot_matrix_identity(
        self, theta, pauli_word, compressed_pauli_word, wires, compressed_wires, tol
    ):
        """Test PauliRot matrix correctly accounts for identities."""

        res = qml.PauliRot.compute_matrix(theta, pauli_word)
        expected = qml.utils.expand(
            qml.PauliRot.compute_matrix(theta, compressed_pauli_word), compressed_wires, wires
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_PauliRot_wire_as_int(self):
        """Test that passing a single wire as an integer works."""

        theta = 0.4
        op = qml.PauliRot(theta, "Z", wires=0)
        decomp_ops = qml.PauliRot.compute_decomposition(theta, "Z", wires=0)

        assert np.allclose(
            op.get_eigvals(), np.array([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
        )
        assert np.allclose(
            op.get_matrix(), np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
        )

        assert len(decomp_ops) == 1

        assert decomp_ops[0].name == "MultiRZ"

        assert decomp_ops[0].wires == Wires([0])
        assert decomp_ops[0].data[0] == theta

    def test_PauliRot_all_Identity(self):
        """Test handling of the all-identity Pauli."""

        theta = 0.4
        op = qml.PauliRot(theta, "II", wires=[0, 1])
        decomp_ops = op.decomposition()

        assert np.allclose(op.get_eigvals(), np.exp(-1j * theta / 2) * np.ones(4))
        assert np.allclose(op.get_matrix() / op.get_matrix()[0, 0], np.eye(4))

        assert len(decomp_ops) == 0

    def test_PauliRot_decomposition_ZZ(self):
        """Test that the decomposition for a ZZ rotation is correct."""

        theta = 0.4
        op = qml.PauliRot(theta, "ZZ", wires=[0, 1])
        decomp_ops = op.decomposition()

        assert len(decomp_ops) == 1

        assert decomp_ops[0].name == "MultiRZ"

        assert decomp_ops[0].wires == Wires([0, 1])
        assert decomp_ops[0].data[0] == theta

    def test_PauliRot_decomposition_XY(self):
        """Test that the decomposition for a XY rotation is correct."""

        theta = 0.4
        op = qml.PauliRot(theta, "XY", wires=[0, 1])
        decomp_ops = op.decomposition()

        assert len(decomp_ops) == 5

        assert decomp_ops[0].name == "Hadamard"
        assert decomp_ops[0].wires == Wires([0])

        assert decomp_ops[1].name == "RX"

        assert decomp_ops[1].wires == Wires([1])
        assert decomp_ops[1].data[0] == np.pi / 2

        assert decomp_ops[2].name == "MultiRZ"
        assert decomp_ops[2].wires == Wires([0, 1])
        assert decomp_ops[2].data[0] == theta

        assert decomp_ops[3].name == "Hadamard"
        assert decomp_ops[3].wires == Wires([0])

        assert decomp_ops[4].name == "RX"

        assert decomp_ops[4].wires == Wires([1])
        assert decomp_ops[4].data[0] == -np.pi / 2

    def test_PauliRot_decomposition_XIYZ(self):
        """Test that the decomposition for a XIYZ rotation is correct."""

        theta = 0.4
        op = qml.PauliRot(theta, "XIYZ", wires=[0, 1, 2, 3])
        decomp_ops = op.decomposition()

        assert len(decomp_ops) == 5

        assert decomp_ops[0].name == "Hadamard"
        assert decomp_ops[0].wires == Wires([0])

        assert decomp_ops[1].name == "RX"

        assert decomp_ops[1].wires == Wires([2])
        assert decomp_ops[1].data[0] == np.pi / 2

        assert decomp_ops[2].name == "MultiRZ"
        assert decomp_ops[2].wires == Wires([0, 2, 3])
        assert decomp_ops[2].data[0] == theta

        assert decomp_ops[3].name == "Hadamard"
        assert decomp_ops[3].wires == Wires([0])

        assert decomp_ops[4].name == "RX"

        assert decomp_ops[4].wires == Wires([2])
        assert decomp_ops[4].data[0] == -np.pi / 2

    @pytest.mark.parametrize("angle", npp.linspace(0, 2 * np.pi, 7, requires_grad=True))
    @pytest.mark.parametrize("pauli_word", ["XX", "YY", "ZZ"])
    def test_differentiability(self, angle, pauli_word, tol):
        """Test that differentiation of PauliRot works."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.PauliRot(theta, pauli_word, wires=[0, 1])

            return qml.expval(qml.PauliZ(0))

        res = circuit(angle)
        gradient = np.squeeze(qml.grad(circuit)(angle))

        assert gradient == pytest.approx(
            0.5 * (circuit(angle + np.pi / 2) - circuit(angle - np.pi / 2)), abs=tol
        )

    @pytest.mark.parametrize("angle", npp.linspace(0, 2 * np.pi, 7, requires_grad=True))
    def test_decomposition_integration(self, angle, tol):
        """Test that the decompositon of PauliRot yields the same results."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.PauliRot(theta, "XX", wires=[0, 1])

            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def decomp_circuit(theta):
            qml.PauliRot.compute_decomposition(theta, "XX", wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(angle), decomp_circuit(angle))
        assert np.allclose(qml.grad(circuit)(angle), qml.grad(decomp_circuit)(angle))

    def test_matrix_incorrect_pauli_word_error(self):
        """Test that _matrix throws an error if a wrong Pauli word is supplied."""

        with pytest.raises(
            ValueError,
            match='The given Pauli word ".*" contains characters that are not allowed.'
            " Allowed characters are I, X, Y and Z",
        ):
            qml.PauliRot.compute_matrix(0.3, "IXYZV")

    def test_init_incorrect_pauli_word_error(self):
        """Test that __init__ throws an error if a wrong Pauli word is supplied."""

        with pytest.raises(
            ValueError,
            match='The given Pauli word ".*" contains characters that are not allowed.'
            " Allowed characters are I, X, Y and Z",
        ):
            qml.PauliRot(0.3, "IXYZV", wires=[0, 1, 2, 3, 4])

    @pytest.mark.parametrize(
        "pauli_word,wires",
        [
            ("XYZ", [0, 1]),
            ("XYZ", [0, 1, 2, 3]),
        ],
    )
    def test_init_incorrect_pauli_word_length_error(self, pauli_word, wires):
        """Test that __init__ throws an error if a Pauli word of wrong length is supplied."""

        with pytest.raises(
            ValueError,
            match="The given Pauli word has length .*, length .* was expected for wires .*",
        ):
            qml.PauliRot(0.3, pauli_word, wires=wires)

    @pytest.mark.parametrize(
        "pauli_word",
        [
            ("XIZ"),
            ("IIII"),
            ("XIYIZI"),
            ("IXI"),
            ("IIIIIZI"),
            ("XYZIII"),
            ("IIIXYZ"),
        ],
    )
    def test_multirz_generator(self, pauli_word):
        """Test that the generator of the MultiRZ gate is correct."""
        op = qml.PauliRot(0.3, pauli_word, wires=range(len(pauli_word)))
        gen = op.generator()

        if pauli_word[0] == "I":
            # this is the identity
            expected_gen = qml.Identity(wires=0)
        else:
            expected_gen = getattr(qml, f"Pauli{pauli_word[0]}")(wires=0)

        for i, pauli in enumerate(pauli_word[1:]):
            i += 1
            if pauli == "I":
                expected_gen = expected_gen @ qml.Identity(wires=i)
            else:
                expected_gen = expected_gen @ getattr(qml, f"Pauli{pauli}")(wires=i)

        assert gen.compare(-0.5 * expected_gen)

    @pytest.mark.gpu
    @pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize("torch_device", [None, "cuda"])
    def test_pauli_rot_identity_torch(self, torch_device, theta):
        """Test that the PauliRot operation returns the correct matrix when
        providing a gate parameter on the GPU and only specifying the identity
        operation."""
        torch = pytest.importorskip("torch")

        if torch_device == "cuda" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        x = torch.tensor(theta, device=torch_device)
        mat = qml.PauliRot(x, "I", wires=[0]).get_matrix()

        val = np.cos(-theta / 2) + 1j * np.sin(-theta / 2)
        exp = torch.tensor(np.diag([val, val]), device=torch_device)
        assert torch.allclose(mat, exp)


class TestMultiRZ:
    """Test the MultiRZ operation."""

    @pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize(
        "wires,expected_matrix",
        [
            ([0], qml.RZ.compute_matrix),
            (
                [0, 1],
                lambda theta: np.diag(
                    np.exp(1j * np.array([-1, 1, 1, -1]) * theta / 2),
                ),
            ),
            (
                [0, 1, 2],
                lambda theta: np.diag(
                    np.exp(1j * np.array([-1, 1, 1, -1, 1, -1, -1, 1]) * theta / 2),
                ),
            ),
        ],
    )
    def test_MultiRZ_matrix_parametric(self, theta, wires, expected_matrix, tol):
        """Test parametrically that the MultiRZ matrix is correct."""

        res_static = qml.MultiRZ.compute_matrix(theta, len(wires))
        res_dynamic = qml.MultiRZ(theta, wires=wires).get_matrix()
        expected = expected_matrix(theta)

        assert np.allclose(res_static, expected, atol=tol, rtol=0)
        assert np.allclose(res_dynamic, expected, atol=tol, rtol=0)

    def test_MultiRZ_matrix_expand(self, tol):
        """Test that the MultiRZ matrix respects the wire order."""

        res = qml.MultiRZ(0.1, wires=[0, 1]).get_matrix(wire_order=[1, 0])
        expected = np.array(
            [
                [0.99875026 - 0.04997917j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.99875026 + 0.04997917j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.99875026 + 0.04997917j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99875026 - 0.04997917j],
            ]
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_MultiRZ_decomposition_ZZ(self):
        """Test that the decomposition for a ZZ rotation is correct."""

        theta = 0.4
        op = qml.MultiRZ(theta, wires=[0, 1])
        decomp_ops = op.decomposition()

        assert decomp_ops[0].name == "CNOT"
        assert decomp_ops[0].wires == Wires([1, 0])

        assert decomp_ops[1].name == "RZ"

        assert decomp_ops[1].wires == Wires([0])
        assert decomp_ops[1].data[0] == theta

        assert decomp_ops[2].name == "CNOT"
        assert decomp_ops[2].wires == Wires([1, 0])

    def test_MultiRZ_decomposition_ZZZ(self):
        """Test that the decomposition for a ZZZ rotation is correct."""

        theta = 0.4
        op = qml.MultiRZ(theta, wires=[0, 2, 3])
        decomp_ops = op.decomposition()

        assert decomp_ops[0].name == "CNOT"
        assert decomp_ops[0].wires == Wires([3, 2])

        assert decomp_ops[1].name == "CNOT"
        assert decomp_ops[1].wires == Wires([2, 0])

        assert decomp_ops[2].name == "RZ"

        assert decomp_ops[2].wires == Wires([0])
        assert decomp_ops[2].data[0] == theta

        assert decomp_ops[3].name == "CNOT"
        assert decomp_ops[3].wires == Wires([2, 0])

        assert decomp_ops[4].name == "CNOT"
        assert decomp_ops[4].wires == Wires([3, 2])

    @pytest.mark.parametrize("angle", npp.linspace(0, 2 * np.pi, 7, requires_grad=True))
    def test_differentiability(self, angle, tol):
        """Test that differentiation of MultiRZ works."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(0)
            qml.MultiRZ(theta, wires=[0, 1])

            return qml.expval(qml.PauliX(0))

        res = circuit(angle)
        gradient = np.squeeze(qml.grad(circuit)(angle))

        assert gradient == pytest.approx(
            0.5 * (circuit(angle + np.pi / 2) - circuit(angle - np.pi / 2)), abs=tol
        )

    @pytest.mark.parametrize("angle", npp.linspace(0, 2 * np.pi, 7, requires_grad=True))
    def test_decomposition_integration(self, angle, tol):
        """Test that the decompositon of MultiRZ yields the same results."""
        angle = qml.numpy.array(angle)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(0)
            qml.MultiRZ(theta, wires=[0, 1])

            return qml.expval(qml.PauliX(0))

        @qml.qnode(dev)
        def decomp_circuit(theta):
            qml.Hadamard(0)
            qml.MultiRZ.compute_decomposition(theta, wires=[0, 1])

            return qml.expval(qml.PauliX(0))

        assert np.allclose(circuit(angle), decomp_circuit(angle))
        assert np.allclose(qml.jacobian(circuit)(angle), qml.jacobian(decomp_circuit)(angle))

    @pytest.mark.parametrize("qubits", range(3, 6))
    def test_multirz_generator(self, qubits, mocker):
        """Test that the generator of the MultiRZ gate is correct."""
        op = qml.MultiRZ(0.3, wires=range(qubits))
        gen = op.generator()

        expected_gen = qml.PauliZ(wires=0)
        for i in range(1, qubits):
            expected_gen = expected_gen @ qml.PauliZ(wires=i)

        assert gen.compare(-0.5 * expected_gen)

        spy = mocker.spy(qml.utils, "pauli_eigs")

        op.generator()
        spy.assert_not_called()


label_data = [
    (
        qml.Rot(1.23456, 2.3456, 3.45678, wires=0),
        "Rot",
        "Rot\n(1.23,\n2.35,\n3.46)",
        "Rot\n(1,\n2,\n3)",
        "Rot⁻¹\n(1,\n2,\n3)",
    ),
    (qml.RX(1.23456, wires=0), "RX", "RX\n(1.23)", "RX\n(1)", "RX⁻¹\n(1)"),
    (qml.RY(1.23456, wires=0), "RY", "RY\n(1.23)", "RY\n(1)", "RY⁻¹\n(1)"),
    (qml.RZ(1.23456, wires=0), "RZ", "RZ\n(1.23)", "RZ\n(1)", "RZ⁻¹\n(1)"),
    (qml.MultiRZ(1.23456, wires=0), "MultiRZ", "MultiRZ\n(1.23)", "MultiRZ\n(1)", "MultiRZ⁻¹\n(1)"),
    (
        qml.PauliRot(1.2345, "XYZ", wires=(0, 1, 2)),
        "RXYZ",
        "RXYZ\n(1.23)",
        "RXYZ\n(1)",
        "RXYZ⁻¹\n(1)",
    ),
    (
        qml.PhaseShift(1.2345, wires=0),
        "Rϕ",
        "Rϕ\n(1.23)",
        "Rϕ\n(1)",
        "Rϕ⁻¹\n(1)",
    ),
    (
        qml.ControlledPhaseShift(1.2345, wires=(0, 1)),
        "Rϕ",
        "Rϕ\n(1.23)",
        "Rϕ\n(1)",
        "Rϕ⁻¹\n(1)",
    ),
    (qml.CRX(1.234, wires=(0, 1)), "RX", "RX\n(1.23)", "RX\n(1)", "RX⁻¹\n(1)"),
    (qml.CRY(1.234, wires=(0, 1)), "RY", "RY\n(1.23)", "RY\n(1)", "RY⁻¹\n(1)"),
    (qml.CRZ(1.234, wires=(0, 1)), "RZ", "RZ\n(1.23)", "RZ\n(1)", "RZ⁻¹\n(1)"),
    (
        qml.CRot(1.234, 2.3456, 3.456, wires=(0, 1)),
        "Rot",
        "Rot\n(1.23,\n2.35,\n3.46)",
        "Rot\n(1,\n2,\n3)",
        "Rot⁻¹\n(1,\n2,\n3)",
    ),
    (qml.U1(1.2345, wires=0), "U1", "U1\n(1.23)", "U1\n(1)", "U1⁻¹\n(1)"),
    (qml.U2(1.2345, 2.3456, wires=0), "U2", "U2\n(1.23,\n2.35)", "U2\n(1,\n2)", "U2⁻¹\n(1,\n2)"),
    (
        qml.U3(1.2345, 2.345, 3.4567, wires=0),
        "U3",
        "U3\n(1.23,\n2.35,\n3.46)",
        "U3\n(1,\n2,\n3)",
        "U3⁻¹\n(1,\n2,\n3)",
    ),
    (
        qml.IsingXX(1.2345, wires=(0, 1)),
        "IsingXX",
        "IsingXX\n(1.23)",
        "IsingXX\n(1)",
        "IsingXX⁻¹\n(1)",
    ),
    (
        qml.IsingYY(1.2345, wires=(0, 1)),
        "IsingYY",
        "IsingYY\n(1.23)",
        "IsingYY\n(1)",
        "IsingYY⁻¹\n(1)",
    ),
    (
        qml.IsingZZ(1.2345, wires=(0, 1)),
        "IsingZZ",
        "IsingZZ\n(1.23)",
        "IsingZZ\n(1)",
        "IsingZZ⁻¹\n(1)",
    ),
]


class TestLabel:
    """Test the label method on parametric ops"""

    @pytest.mark.parametrize("op, label1, label2, label3, label4", label_data)
    def test_label_method(self, op, label1, label2, label3, label4):
        """Test label method with plain scalers."""

        assert op.label() == label1
        assert op.label(decimals=2) == label2
        assert op.label(decimals=0) == label3

        op.inv()
        assert op.label(decimals=0) == label4
        op.inv()

    def test_label_tf(self):
        """Test label methods work with tensorflow variables"""
        tf = pytest.importorskip("tensorflow")

        op1 = qml.RX(tf.Variable(0.123456), wires=0)
        assert op1.label(decimals=2) == "RX\n(0.12)"

        op2 = qml.CRX(tf.Variable(0.12345), wires=(0, 1))
        assert op2.label(decimals=2) == "RX\n(0.12)"

        op3 = qml.Rot(tf.Variable(0.1), tf.Variable(0.2), tf.Variable(0.3), wires=0)
        assert op3.label(decimals=2) == "Rot\n(0.10,\n0.20,\n0.30)"

    def test_label_torch(self):
        """Test label methods work with torch tensors"""
        torch = pytest.importorskip("torch")

        op1 = qml.RX(torch.tensor(1.23456), wires=0)
        assert op1.label(decimals=2) == "RX\n(1.23)"

        op2 = qml.CRX(torch.tensor(1.23456), wires=(0, 1))
        assert op2.label(decimals=2) == "RX\n(1.23)"

        op3 = qml.Rot(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3), wires=0)
        assert op3.label(decimals=2) == "Rot\n(0.10,\n0.20,\n0.30)"

    def test_label_jax(self):
        """Test the label method works with jax"""
        jax = pytest.importorskip("jax")

        op1 = qml.RX(jax.numpy.array(1.23456), wires=0)
        assert op1.label(decimals=2) == "RX\n(1.23)"

        op2 = qml.CRX(jax.numpy.array(1.23456), wires=(0, 1))
        assert op2.label(decimals=2) == "RX\n(1.23)"

        op3 = qml.Rot(jax.numpy.array(0.1), jax.numpy.array(0.2), jax.numpy.array(0.3), wires=0)
        assert op3.label(decimals=2) == "Rot\n(0.10,\n0.20,\n0.30)"

    def test_string_parameter(self):
        """Test labelling works if variable is a string instead of a float."""

        op1 = qml.RX("x", wires=0)
        assert op1.label() == "RX"
        assert op1.label(decimals=0) == "RX\n(x)"

        op2 = qml.CRX("y", wires=(0, 1))
        assert op2.label(decimals=0) == "RX\n(y)"

        op3 = qml.Rot("x", "y", "z", wires=0)
        assert op3.label(decimals=0) == "Rot\n(x,\ny,\nz)"


control_data = [
    (qml.Rot(1, 2, 3, wires=0), Wires([])),
    (qml.RX(1.23, wires=0), Wires([])),
    (qml.RY(1.23, wires=0), Wires([])),
    (qml.MultiRZ(1.234, wires=(0, 1, 2)), Wires([])),
    (qml.PauliRot(1.234, "IXY", wires=(0, 1, 2)), Wires([])),
    (qml.PhaseShift(1.234, wires=0), Wires([])),
    (qml.U1(1.234, wires=0), Wires([])),
    (qml.U2(1.234, 2.345, wires=0), Wires([])),
    (qml.U3(1.234, 2.345, 3.456, wires=0), Wires([])),
    (qml.IsingXX(1.234, wires=(0, 1)), Wires([])),
    (qml.IsingYY(1.234, wires=(0, 1)), Wires([])),
    (qml.IsingZZ(1.234, wires=(0, 1)), Wires([])),
    ### Controlled Ops
    (qml.ControlledPhaseShift(1.234, wires=(0, 1)), Wires(0)),
    (qml.CPhase(1.234, wires=(0, 1)), Wires(0)),
    (qml.CRX(1.234, wires=(0, 1)), Wires(0)),
    (qml.CRY(1.234, wires=(0, 1)), Wires(0)),
    (qml.CRZ(1.234, wires=(0, 1)), Wires(0)),
    (qml.CRot(1.234, 2.2345, 3.456, wires=(0, 1)), Wires(0)),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test the ``control_wires`` attribute for parametrized operations."""
    assert op.control_wires == control_wires
