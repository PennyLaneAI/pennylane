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
Tests for the controlled decompositions.
"""
import pytest

import numpy as np
import pennylane as qml
from pennylane.ops import ctrl_decomp_zyz
from pennylane.wires import Wires


class TestControlledDecompositionZYZ:
    """tests for qml.ops.ctrl_decomp_zyz"""

    def test_invalid_op_error(self):
        """Tests that an error is raised when an invalid operation is passed"""
        with pytest.raises(
            ValueError, match="The target operation must be a single-qubit operation"
        ):
            _ = ctrl_decomp_zyz(qml.CNOT([0, 1]), [2])

    su2_ops = [
        qml.RX(0.123, wires=0),
        qml.RY(0.123, wires=0),
        qml.RZ(0.123, wires=0),
        qml.Rot(0.123, 0.456, 0.789, wires=0),
    ]

    unitary_ops = [
        qml.Hadamard(0),
        qml.PauliZ(0),
        qml.S(0),
        qml.PhaseShift(1.5, wires=0),
        qml.QubitUnitary(
            np.array(
                [
                    [-0.28829348 - 0.78829734j, 0.30364367 + 0.45085995j],
                    [0.53396245 - 0.10177564j, 0.76279558 - 0.35024096j],
                ]
            ),
            wires=0,
        ),
        qml.DiagonalQubitUnitary(np.array([1, -1]), wires=0),
    ]

    @pytest.mark.parametrize("op", su2_ops + unitary_ops)
    @pytest.mark.parametrize("control_wires", ([1], [1, 2], [1, 2, 3]))
    def test_decomposition_circuit(self, op, control_wires, tol):
        """Tests that the controlled decomposition of a single-qubit operation
        behaves as expected in a quantum circuit"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def decomp_circuit():
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
            ctrl_decomp_zyz(op, Wires(control_wires))
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
            qml.ctrl(op, control_wires)
            return qml.probs()

        res = decomp_circuit()
        expected = expected_circuit()
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op", su2_ops)
    @pytest.mark.parametrize("control_wires", ([1], [1, 2], [1, 2, 3]))
    def test_decomposition_matrix(self, op, control_wires, tol):
        """Tests that the matrix representation of the controlled ZYZ decomposition
        of a single-qubit operation is correct"""
        expected_op = qml.ctrl(op, control_wires)
        res = qml.matrix(ctrl_decomp_zyz, wire_order=control_wires + [0])(op, control_wires)
        expected = expected_op.matrix()

        assert np.allclose(expected, res, atol=tol, rtol=0)

    def test_correct_decomp(self):
        """Test that the operations in the decomposition are correct."""
        phi, theta, omega = 0.123, 0.456, 0.789
        op = qml.Rot(phi, theta, omega, wires=0)
        control_wires = [1, 2, 3]
        decomps = ctrl_decomp_zyz(op, Wires(control_wires))

        expected_ops = [
            qml.RZ(0.123, wires=0),
            qml.RY(0.456 / 2, wires=0),
            qml.MultiControlledX(wires=control_wires + [0]),
            qml.RY(-0.456 / 2, wires=0),
            qml.RZ(-(0.123 + 0.789) / 2, wires=0),
            qml.MultiControlledX(wires=control_wires + [0]),
            qml.RZ((0.789 - 0.123) / 2, wires=0),
        ]
        assert all(
            qml.equal(decomp_op, expected_op)
            for decomp_op, expected_op in zip(decomps, expected_ops)
        )
        assert len(decomps) == 7

    @pytest.mark.parametrize("op", su2_ops + unitary_ops)
    @pytest.mark.parametrize("control_wires", ([1], [1, 2], [1, 2, 3]))
    def test_decomp_queues_correctly(self, op, control_wires, tol):
        """Test that any incorrect operations aren't queued when using
        ``ctrl_decomp_zyz``."""
        decomp = ctrl_decomp_zyz(op, control_wires=Wires(control_wires))
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def queue_from_list():
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
            for o in decomp:
                qml.apply(o)
            return qml.state()

        @qml.qnode(dev)
        def queue_from_qnode():
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
            ctrl_decomp_zyz(op, control_wires=Wires(control_wires))
            return qml.state()

        res1 = queue_from_list()
        res2 = queue_from_qnode()
        assert np.allclose(res1, res2, atol=tol, rtol=0)

    def test_trivial_ops_in_decomposition(self):
        """Test that an operator decomposition doesn't have trivial rotations."""
        op = qml.RZ(np.pi, wires=0)
        decomp = ctrl_decomp_zyz(op, [1])
        expected = [
            qml.RZ(np.pi, wires=0),
            qml.MultiControlledX(wires=[1, 0]),
            qml.RZ(-np.pi / 2, wires=0),
            qml.MultiControlledX(wires=[1, 0]),
            qml.RZ(-np.pi / 2, wires=0),
        ]

        assert len(decomp) == 5
        assert all(qml.equal(o, e) for o, e in zip(decomp, expected))
