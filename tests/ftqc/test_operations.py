# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the operations in the FTQC module"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.ftqc.operations import RotXZX
from pennylane.ops.functions import assert_valid
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumScript
from pennylane.wires import Wires


class TestRotXZX:

    def test_op_init(self):
        """Test that the operator initialization works as expected"""
        phi, theta, omega = np.pi / 4, 1.23, -0.5
        wires = 0
        op = RotXZX(phi, theta, omega, wires=wires)

        assert op.wires == Wires(wires)
        assert op.data == (phi, theta, omega)

    def test_is_valid_op(self):
        """Assert RotXZX is a valid operator"""
        op = RotXZX(1.2, 2.3, -0.5, wires=0)
        assert_valid(op)

    def test_single_qubit_rot_angles(self):
        """Test that the single_qubit_rot_angles method works as expected for the
        RotXZX gate"""
        phi, theta, omega = np.pi / 4, 1.23, -0.5
        op = RotXZX(phi, theta, omega, wires=0)

        assert op.single_qubit_rot_angles() == (phi, theta, omega)

    @pytest.mark.parametrize("use_graph", [True, False])
    def test_decomposition(self, use_graph):
        """Test that the RotXZX has the expected decomposition"""
        phi, theta, omega = np.pi / 4, 1.23, -0.5
        wire = 0
        expected_decomp = [qml.RX(phi, wire), qml.RZ(theta, wire), qml.RX(omega, wire)]

        if use_graph:
            decomp_qfuncs = qml.list_decomps(RotXZX)
            assert len(decomp_qfuncs) == 1

            with AnnotatedQueue() as q:
                decomp_qfuncs[0](phi, theta, omega, wires=wire)

            ops = QuantumScript.from_queue(q).operations
        else:
            ops = RotXZX.compute_decomposition(phi, theta, omega, wires=wire)

        for op, expected_op in zip(ops, expected_decomp):
            qml.assert_equal(op, expected_op)

    def test_adjoint(self):
        """Test that the adjoint method works as expected for the RotXZX gate"""
        input_state = np.random.random(2) + 1j * np.random.random(2)
        input_state = input_state / np.linalg.norm(input_state)

        phi, theta, omega = np.pi / 4, 1.23, -0.5

        op = RotXZX(phi, theta, omega, wires=0)
        adj_op = qml.adjoint(op, lazy=False)

        assert isinstance(adj_op, RotXZX)
        assert adj_op.data == (-omega, -theta, -phi)

        @qml.qnode(qml.device("default.qubit"))
        def circuit(state):
            qml.StatePrep(state, wires=0)
            RotXZX(phi, theta, omega, wires=0)
            qml.adjoint(RotXZX(phi, theta, omega, wires=0))
            return qml.state()

        assert np.allclose(circuit(input_state), input_state)
