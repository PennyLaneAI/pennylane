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

from functools import partial

import networkx as nx
import numpy as np
import pytest

import pennylane as qml
from pennylane import math
from pennylane.devices.qubit.apply_operation import apply_operation
from pennylane.ftqc import (
    GraphStatePrep,
    RotXZX,
    cond_measure,
    convert_to_mbqc_gateset,
    diagonalize_mcms,
    measure_arbitrary_basis,
    measure_x,
)
from pennylane.ftqc.decomposition import _rot_to_xzx


class TestGateSetDecomposition:
    """Test decomposition to the MBQC gate-set"""

    def test_rot_to_xzx_decomp_rule(self):
        """Test the _rot_to_xzx rule"""
        phi, theta, omega = 1.39, -0.123, np.pi / 7

        mat = qml.Rot.compute_matrix(phi, theta, omega)
        a1, a2, a3 = math.decomposition.xzx_rotation_angles(mat)

        with qml.queuing.AnnotatedQueue() as q:
            _rot_to_xzx(phi, theta, omega, wires=0)

        ops = qml.tape.QuantumScript.from_queue(q).operations
        assert len(ops) == 1
        assert isinstance(ops[0], RotXZX)
        assert np.allclose(ops[0].data, (a1, a2, a3))

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
        xzx_op = q.queue[0]
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

        expected_ops = [RotXZX, RotXZX, qml.H, qml.CNOT, qml.H]
        for op, expected_type in zip(new_tape.operations, expected_ops):
            assert isinstance(op, expected_type)

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
            qml.RY(1.23, 0)  # RotXZX
            qml.RX(0.456, 1)  # RotXZX
            qml.Rot(np.pi / 3, -0.5, np.pi / 5, 2)  # RotXZX
            qml.CZ([0, 1])  # H(1), CNOT, H(1)
            qml.CY([1, 2])  # RotXZX(2), CNOT, RotXZX(2), CNOT, S(1)
            return qml.state()

        decomposed_circuit = convert_to_mbqc_gateset(circuit)

        tape = qml.workflow.construct_tape(decomposed_circuit)()
        expected_ops = (
            [RotXZX] * 3 + [qml.H, qml.CNOT, qml.H] + [RotXZX, qml.CNOT, RotXZX, qml.CNOT, qml.S]
        )

        for op, expected_type in zip(tape.operations, expected_ops):
            assert isinstance(op, expected_type)

        assert np.allclose(circuit(), decomposed_circuit())

    def test_explicit_mbqc_implementation_matches(self):
        """Test that the explicit MBQC implementation of the RotXZX gate that
        a Rot gate decomposes to produces the expected analytic result"""

        dev = qml.device("default.qubit")

        state = np.random.random(2) + 1j * np.random.random(2)
        state = state / np.linalg.norm(state)

        @qml.qnode(dev)
        def rot_ref(angles):
            """Reference circuit for applying some state preparation and then
            performing a rotation, expressed using the PL Rot gate (ZYZ)"""
            qml.StatePrep(state, wires=0)
            qml.Rot(*angles, wires=0)
            return qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.expval(qml.Z(0))

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="tree-traversal")
        def rot_mbqc(rot_xzx_gate):
            """This circuit accepts a RotXZX gate, and creates a circuit equivalent
            to performing state preparation and then applying the rotation gate,
            with the rotation expressed in the MBQC formalism"""
            # prep input node
            qml.StatePrep(state, wires=[0])

            # prep graph state
            GraphStatePrep(nx.grid_graph((4,)), wires=[1, 2, 3, 4])

            # entangle input and graph state
            qml.CZ([0, 1])

            # MBQC Z rotation: X, X, +/- angle, X
            angles = rot_xzx_gate.single_qubit_rot_angles()
            m1 = measure_x(0)
            m2 = cond_measure(
                m1,
                partial(measure_arbitrary_basis, angle=angles[0]),
                partial(measure_arbitrary_basis, angle=-angles[0]),
            )(plane="XY", wires=1)
            m3 = cond_measure(
                m2,
                partial(measure_arbitrary_basis, angle=angles[1]),
                partial(measure_arbitrary_basis, angle=-angles[1]),
            )(plane="XY", wires=2)
            m4 = cond_measure(
                (m1 + m3) % 2,
                partial(measure_arbitrary_basis, angle=angles[2]),
                partial(measure_arbitrary_basis, angle=-angles[2]),
            )(plane="XY", wires=3)

            # corrections based on measurement outcomes
            qml.cond((m1 + m3) % 2, qml.Z)(4)
            qml.cond((m2 + m4) % 2, qml.X)(4)

            return qml.expval(qml.X(4)), qml.expval(qml.Y(4)), qml.expval(qml.Z(4))

        angles = 1.39, -0.123, np.pi / 7

        # convert Rot to XZX using the decomposition rule
        with qml.queuing.AnnotatedQueue() as q:
            _rot_to_xzx(*angles, wires=0)
        xzx_op = q.queue[0]

        assert np.allclose(rot_ref(angles), rot_mbqc(xzx_op))
