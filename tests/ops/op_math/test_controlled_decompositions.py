# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for controlled operation decompositions.
"""

from functools import reduce
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
        decomps = ctrl_decomp_zyz(op, Wires(control_wires))
        expected_op = qml.ctrl(op, control_wires)

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def decomp_circuit():
            for i in range(4):
                qml.Hadamard(i)
            for decomp_op in decomps:
                qml.apply(decomp_op)
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            for i in range(4):
                qml.Hadamard(i)
            qml.apply(expected_op)
            return qml.probs()

        res = decomp_circuit()
        expected = expected_circuit()
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op", su2_ops)
    @pytest.mark.parametrize("control_wires", ([1], [1, 2], [1, 2, 3]))
    def test_decomposition_matrix(self, op, control_wires, tol):
        """Tests that the matrix representation of the controlled ZYZ decomposition
        of a single-qubit operation is correct"""
        # TODO: Update tests to include non-SU(2) unitaries once supported
        decomps = ctrl_decomp_zyz(op, Wires(control_wires))
        expected_op = qml.ctrl(op, control_wires)
        decomp_mats = [
            np.kron(np.eye(2 ** len(control_wires)), decomp_op.matrix())
            if not isinstance(decomp_op, (qml.CNOT, qml.Toffoli, qml.MultiControlledX))
            else decomp_op.matrix()
            for decomp_op in decomps
        ]

        res = reduce(np.matmul, reversed(decomp_mats))
        expected = expected_op.matrix()
        assert np.allclose(expected, res, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "control_wires, control_gate",
        [([1], qml.CNOT), ([1, 2], qml.Toffoli), ([1, 2, 3], qml.MultiControlledX)],
    )
    def test_correct_decomp(self, control_wires, control_gate):
        """Test that the operations in the decomposition are correct."""
        phi, theta, omega = 0.123, 0.456, 0.789
        op = qml.Rot(phi, theta, omega, wires=0)
        decomps = ctrl_decomp_zyz(op, Wires(control_wires))

        assert len(decomps) == 7
        assert isinstance(decomps[2], control_gate)
        assert isinstance(decomps[5], control_gate)
        assert isinstance(decomps[0], qml.RZ)
        assert isinstance(decomps[4], qml.RZ)
        assert isinstance(decomps[6], qml.RZ)
        assert isinstance(decomps[1], qml.RY)
        assert isinstance(decomps[3], qml.RY)

        decomp_angles = [
            0.123,
            0.456 / 2,
            None,
            -0.456 / 2,
            -(0.123 + 0.789) / 2,
            None,
            (0.789 - 0.123) / 2,
        ]

        for i in range(7):
            if i in {2, 5}:
                assert decomps[i].wires == control_wires + Wires([0])
            else:
                assert decomps[i].wires == Wires([0])
                assert decomps[i].data[0] == decomp_angles[i]
