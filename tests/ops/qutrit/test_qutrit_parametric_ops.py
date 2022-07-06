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
Unit tests for the available built-in parametric qutrit operations.
"""
from functools import reduce
import pytest
import copy
import numpy as np
from pennylane import numpy as npp

import pennylane as qml
from pennylane.wires import Wires
from tests.ops.qubit.test_parametric_ops import NON_PARAMETRIZED_OPERATIONS

from gate_data import TSHIFT, TCLOCK

PARAMETRIZED_OPERATIONS = [
    qml.TRX(0.123, wires=0, subspace=[1, 2]),
    qml.QutritUnitary(TSHIFT, wires=0),
    qml.ControlledQutritUnitary(TCLOCK, wires=[0], control_wires=[2]),
]

BROADCASTED_OPERATIONS = [
    qml.TRX([0.142, -0.61, 2.3], wires=0, subspace=[1, 2]),
    qml.QutritUnitary(np.array([TSHIFT, TCLOCK]), wires=0),
    qml.ControlledQutritUnitary(np.array([TSHIFT, TCLOCK]), wires=[0], control_wires=[2]),
]

NON_PARAMETRIZED_OPERATIONS = [
    qml.TShift(wires=0),
    qml.TClock(wires=0),
    qml.TAdd(wires=[0, 1]),
    qml.TSWAP(wires=[0, 1]),
]


ALL_OPERATIONS = NON_PARAMETRIZED_OPERATIONS + PARAMETRIZED_OPERATIONS

dot_broadcasted = lambda a, b: np.einsum("...ij,...jk->...ik", a, b)
multi_dot_broadcasted = lambda matrices: reduce(dot_broadcasted, matrices)


class TestOperations:
    @pytest.mark.parametrize("op", ALL_OPERATIONS + BROADCASTED_OPERATIONS)
    def test_parametrized_op_copy(self, op, tol):
        """Tests that copied parametrized ops function as expected"""
        copied_op = copy.copy(op)
        assert np.allclose(op.matrix(), copied_op.matrix(), atol=tol)

        op.inv()
        copied_op2 = copy.copy(op)
        assert np.allclose(op.matrix(), copied_op2.matrix(), atol=tol)
        op.inv()

    @pytest.mark.parametrize("op", PARAMETRIZED_OPERATIONS)
    def test_adjoint_unitaries(self, op, tol):
        op_d = op.adjoint()
        res1 = np.dot(op.matrix(), op_d.matrix())
        res2 = np.dot(op_d.matrix(), op.matrix())
        assert np.allclose(res1, np.eye(3 ** len(op.wires)), atol=tol)
        assert np.allclose(res2, np.eye(3 ** len(op.wires)), atol=tol)
        assert op.wires == op_d.wires

    @pytest.mark.parametrize("op", BROADCASTED_OPERATIONS)
    def test_adjoint_unitaries_broadcasted(self, op, tol):
        op_d = op.adjoint()
        res1 = dot_broadcasted(op.matrix(), op_d.matrix())
        res2 = dot_broadcasted(op_d.matrix(), op.matrix())
        I = [np.eye(3 ** len(op.wires))] * op.batch_size
        assert np.allclose(res1, I, atol=tol)
        assert np.allclose(res2, I, atol=tol)
        assert op.wires == op_d.wires


# TODO: Add tests for parameter frequencies
# TODO: Add tests for decompositions


class TestMatrix:
    def test_trx(self, tol):
        """Test x rotation is correct"""

        # test identity for theta = 0
        expected = np.eye(3)
        assert np.allclose(qml.TRX.compute_matrix(0, subspace=[0, 1]), expected, atol=tol, rtol=0)
        assert np.allclose(qml.TRX.compute_matrix(0, subspace=[1, 2]), expected, atol=tol, rtol=0)
        assert np.allclose(qml.TRX.compute_matrix(0, subspace=[0, 2]), expected, atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, -1j, 0], [-1j, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi / 2, subspace=[0, 1]), expected, atol=tol, rtol=0
        )

        expected = np.array([[np.sqrt(2), 0, 0], [0, 1, -1j], [0, -1j, 1]]) / np.sqrt(2)
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi / 2, subspace=[1, 2]), expected, atol=tol, rtol=0
        )

        expected = np.array([[1, 0, -1j], [0, np.sqrt(2), 0], [-1j, 0, 1]]) / np.sqrt(2)
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi / 2, subspace=[0, 2]), expected, atol=tol, rtol=0
        )

        # test identity for broadcasted theta=pi/2
        pi_half = np.array([np.pi / 2] * 2)
        expected = np.tensordot(
            [1, 1], np.array([[1, -1j, 0], [-1j, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2), axes=0
        )
        assert np.allclose(
            qml.TRX.compute_matrix(pi_half, subspace=[0, 1]), expected, atol=tol, rtol=0
        )

        expected = np.tensordot(
            [1, 1], np.array([[1, 0, -1j], [0, np.sqrt(2), 0], [-1j, 0, 1]]) / np.sqrt(2), axes=0
        )
        assert np.allclose(
            qml.TRX.compute_matrix(pi_half, subspace=[0, 2]), expected, atol=tol, rtol=0
        )

        expected = np.tensordot(
            [1, 1], np.array([[np.sqrt(2), 0, 0], [0, 1, -1j], [0, -1j, 1]]) / np.sqrt(2), axes=0
        )
        assert np.allclose(
            qml.TRX.compute_matrix(pi_half, subspace=[1, 2]), expected, atol=tol, rtol=0
        )

        # test identity for theta=pi
        expected = -1j * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1j]])
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi, subspace=[0, 1]), expected, atol=tol, rtol=0
        )

        expected = -1j * np.array([[1j, 0, 0], [0, 0, 1], [0, 1, 0]])
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi, subspace=[1, 2]), expected, atol=tol, rtol=0
        )

        expected = -1j * np.array([[0, 0, 1], [0, 1j, 0], [1, 0, 0]])
        assert np.allclose(
            qml.TRX.compute_matrix(np.pi, subspace=[0, 2]), expected, atol=tol, rtol=0
        )
