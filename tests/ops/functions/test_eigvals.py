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
Unit tests for the eigvals transform
"""
from functools import reduce

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.op_transforms import OperationTransformError

from gate_data import I, X, Y, Z, H, S, CNOT, Roty as RY

one_qubit_no_parameter = [
    qml.PauliX,
    qml.PauliY,
    qml.PauliZ,
    qml.Hadamard,
    qml.S,
    qml.T,
    qml.SX,
]

one_qubit_one_parameter = [qml.RX, qml.RY, qml.RZ, qml.PhaseShift]


class TestSingleOperation:
    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_non_parametric_instantiated(self, op_class):
        """Verify that the eigenvalues of non-parametric one qubit gates is correct
        when provided as an instantiated operation"""
        op = op_class(wires=0)
        res = qml.eigvals(op)
        expected = op.get_eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_non_parametric_gates_qfunc(self, op_class):
        """Verify that the eigenvalues of non-parametric one qubit gates is correct
        when provided as a qfunc"""
        res = qml.eigvals(op_class)(wires=0)
        expected = op_class(wires=0).get_eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_non_parametric_gates_qnode(self, op_class):
        """Verify that the eigenvalues of non-parametric one qubit gates is correct
        when provided as a QNode"""
        dev = qml.device("default.qubit", wires=1)
        qnode = qml.QNode(lambda: op_class(wires=0) and qml.probs(wires=0), dev)
        res = qml.eigvals(qnode)()
        expected = op_class(wires=0).get_eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_instantiated(self, op_class):
        """Verify that the eigenvalues of non-parametric one qubit gates is correct
        when provided as an instantiated operation"""
        op = op_class(0.54, wires=0)
        res = qml.eigvals(op)
        expected = op.get_eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_qfunc(self, op_class):
        """Verify that the eigenvalues of non-parametric one qubit gates is correct
        when provided as a qfunc"""
        res = qml.eigvals(op_class)(0.54, wires=0)
        expected = op_class(0.54, wires=0).get_eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_qnode(self, op_class):
        """Verify that the eigenvalues of non-parametric one qubit gates is correct
        when provided as a QNode"""
        dev = qml.device("default.qubit", wires=1)
        qnode = qml.QNode(lambda x: op_class(x, wires=0) and qml.probs(wires=0), dev)
        res = qml.eigvals(qnode)(0.54)
        expected = op_class(0.54, wires=0).get_eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_adjoint(self, op_class):
        """Test that the adjoint is correctly taken into account"""
        res = qml.eigvals(qml.adjoint(op_class))(0.54, wires=0)
        expected = op_class(-0.54, wires=0).get_eigvals()
        assert np.allclose(res, expected)

    def test_ctrl(self):
        """Test that the ctrl is correctly taken into account"""
        res = qml.eigvals(qml.ctrl(qml.PauliX, 0))(wires=1)
        expected = np.linalg.eigvals(qml.matrix(qml.CNOT(wires=[0, 1])))
        assert np.allclose(res, expected)

    # @pytest.mark.parametrize("target_wire", [0, 2, 3, 4])
    # def test_CNOT_permutations(self, target_wire):
    #     """Test CNOT: 2-qubit gate with different target wires, some non-adjacent."""
    #     res = qml.eigvals(qml.CNOT, wire_order=[0, 1, 2, 3, 4])(wires=[1, target_wire])

    #     # compute the expected matrix
    #     perm = np.swapaxes(
    #         np.swapaxes(np.arange(2**5).reshape([2] * 5), 0, 1), 0, target_wire
    #     ).flatten()
    #     expected = reduce(np.kron, [CNOT, I, I, I])[:, perm][perm]
    #     assert np.allclose(res, expected)

    # def test_hamiltonian(self):
    #     """Test that the matrix of a Hamiltonian is correctly returned"""
    #     H = qml.PauliZ(0) @ qml.PauliY(1) - 0.5 * qml.PauliX(1)
    #     mat = qml.eigvals(H, wire_order=[1, 0, 2])
    #     expected = reduce(np.kron, [Y, Z, I]) - 0.5 * reduce(np.kron, [X, I, I])

    # @pytest.mark.xfail(
    #     reason="This test will fail because Hamiltonians are not queued to tapes yet!"
    # )
    # def test_hamiltonian_qfunc(self):
    #     """Test that the matrix of a Hamiltonian is correctly returned"""

    #     def ansatz(x):
    #         return qml.PauliZ(0) @ qml.PauliY(1) - x * qml.PauliX(1)

    #     x = 0.5
    #     mat = qml.eigvals(ansatz, wire_order=[1, 0, 2])(x)
    #     expected = reduce(np.kron, [Y, Z, I]) - x * reduce(np.kron, [X, I, I])