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

"""Unit tests for the decomposition transforms in the FTQC module"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.qubit.apply_operation import apply_operation
from pennylane.ftqc.decomposition import _rot_to_xzx, convert_to_mbqc_gateset, mbqc_gate_set
from pennylane.ftqc.operations import RotXZX
from pennylane.ops.op_math.decompositions.single_qubit_unitary import _get_xzx_angles


class TestGateSetDecomposition:
    """Test decomposition to the MBQC gate-set"""

    def test_rot_to_xzx_decomp_rule(self):
        """Test the _rot_to_xzx rule"""
        phi, theta, omega = 1.39, -0.123, np.pi / 7

        mat = qml.Rot.compute_matrix(phi, theta, omega)
        a1, a2, a3, _ = _get_xzx_angles(mat)

        with qml.queuing.AnnotatedQueue() as q:
            _rot_to_xzx(phi, theta, omega, wires=0)

        ops = qml.tape.QuantumScript.from_queue(q).operations
        assert len(ops) == 1
        assert isinstance(ops[0], RotXZX)
        assert np.allclose(ops[0].data, (a3, a2, a1))

    def test_decomposition_to_xzx_is_valid(self):
        """Test that the analytic result of applying the XZX rotation is as expected
        when decomposing from Rot to RotXZX to RX/RZ rotations"""

        state = np.random.random(2) + 1j * np.random.random(2)
        state = state / np.linalg.norm(state)

        phi, theta, omega = 1.39, -0.123, np.pi / 7

        # rot op and output state
        rot_op = qml.Rot(phi, theta, omega, wires=0)
        expected_state = apply_operation(rot_op, state)

        # decomposed op and output state
        with qml.queuing.AnnotatedQueue() as q:
            _rot_to_xzx(phi, theta, omega, wires=0)
        xzx_op = qml.tape.QuantumScript.from_queue(q).operations[0]
        base_rot_ops = xzx_op.decomposition()
        for op in base_rot_ops:
            assert op.__class__ in {qml.RX, qml.RZ}
            state = apply_operation(op, state)

        assert np.allclose(expected_state, state)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_convert_to_mbqc_gateset(self):
        """Test the convert_to_mbqc_gateset works as expected"""
        tape = qml.tape.QuantumScript(
            [qml.Rot(1.2, -0.34, 0.056, wires="a"), qml.RY(0.46, 1), qml.CZ([0, 1])]
        )

        (new_tape,), _ = convert_to_mbqc_gateset(tape)

        for op in new_tape.operations:
            assert op.__class__ in mbqc_gate_set

    def test_error_if_old_decomp_method(self):
        """Test that a clear error is raised if trying to use the convert_to_mbqc_gateset
        transform with the old decomposition method"""

        tape = qml.tape.QuantumScript(
            [qml.Rot(1.2, -0.34, 0.056, wires="a"), qml.RY(0.46, 1), qml.CZ([0, 1])]
        )

        with pytest.raises(RuntimeError, match="requires the graph-based decomposition method"):
            _, _ = convert_to_mbqc_gateset(tape)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_qnode_integration(self):
        """Test that a QNode containing Rot and other unsupported gates is decomposed as
        expected by the convert_to_mbqc_gateset"""

        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.RY(1.23, 0)
            qml.RX(0.456, 1)
            qml.Rot(np.pi / 3, -0.5, np.pi / 5, 2)
            qml.CZ([0, 1])
            qml.CY([1, 2])
            return qml.state()

        decomposed_circuit = convert_to_mbqc_gateset(circuit)

        assert np.allclose(circuit(), decomposed_circuit())
