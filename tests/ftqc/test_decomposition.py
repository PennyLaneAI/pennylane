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
    convert_to_mbqc_formalism,
    convert_to_mbqc_gateset,
    diagonalize_mcms,
    measure_arbitrary_basis,
    measure_x,
)
from pennylane.ftqc.decomposition import (
    _rot_to_xzx,
    cnot_stencil,
    convert_single_qubit_gate,
    h_stencil,
    rot_stencil,
    rz_stencil,
    s_stencil,
)
from pennylane.ftqc.utils import QubitMgr


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


class TestMBQCFormalismConversion:
    """Test the transform convert_to_mbqc_formalism, converting to the MBQC formalism with
    online corrections immediately after each gate"""

    @pytest.mark.parametrize(
        "op",
        [qml.H(2), qml.S(2), qml.RZ(1.23, 2), RotXZX(0.12, 0.34, 0.56, 2)],
    )
    def test_convert_single_qubit_gate(self, op):
        # or density_matrix instead?
        dev = qml.device("lightning.qubit", wires=5)
        q_mgr = QubitMgr(num_qubits=5, start_idx=0)
        wire_map = {2: q_mgr.acquire_qubit()}

        ref_tape = qml.tape.QuantumScript(
            [op], measurements=[qml.expval(qml.X(2)), qml.expval(qml.Y(2)), qml.expval(qml.Z(2))]
        )

        with qml.queuing.AnnotatedQueue() as q:
            w = op.wires[0]
            wire_map[w] = convert_single_qubit_gate(q_mgr, op=op, in_wire=wire_map[w])
            qml.expval(qml.X(wire_map[w]))
            qml.expval(qml.Y(wire_map[w]))
            qml.expval(qml.X(wire_map[w]))

        tape = qml.tape.QuantumScript.from_queue(q)
        (diagonalized_tape,), _ = diagonalize_mcms(tape)

        print(diagonalized_tape.operations)
        print(diagonalized_tape.measurements)
        res = qml.execute(
            [
                diagonalized_tape,
            ],
            device=dev,
        )
        print(res)
        raise RuntimeError

    def test_convert_cnot(self):

        # check stencil is called
        # check that applying and getting density state works as expected
        raise RuntimeError

    @pytest.mark.parametrize("all_outcomes", (True, False))
    def test_conversion_of_mbqc_gates_all_counts(self, all_outcomes):

        if all_outcomes is True:
            pytest.skip(reason="issue with dynamic one-shot, fix later if it matters")

        dev = qml.device("lightning.qubit", shots=500)

        tape = qml.tape.QuantumScript(
            [qml.H(1)],
            measurements=[qml.counts(wires=[0, 1], all_outcomes=all_outcomes)],
            shots=1000,
        )

        # after transform, only state prep and MCMs are present on the tape
        (transformed_tape,), _ = convert_to_mbqc_formalism(tape)
        (diagonalized_tape,), _ = diagonalize_mcms(transformed_tape)
        res = qml.execute([diagonalized_tape], dev)[0]

        if all_outcomes is True:
            assert list(res.keys()) == ["00", "01", "10", "11"]
            assert res["11"] == 0
            assert res["10"] == 0
        else:
            assert list(res.keys()) == ["00", "01"]

    @pytest.mark.parametrize(
        "gate, wire",
        [(qml.X, 2), (qml.Y, 1), (qml.Z, 0), (qml.Identity, 7), (qml.GlobalPhase(1.23), 3)],
    )
    def test_pauli_gates_are_updated(self, gate, wire):

        dev = qml.device("lightning.qubit", shots=500)

        tape = qml.tape.QuantumScript(
            [qml.H(wire), gate(wire)],
            measurements=[qml.sample(wires=[0, 1])],
            shots=1000,
        )

        # after transform, only state prep and MCMs are present on the tape
        (transformed_tape,), _ = convert_to_mbqc_formalism(tape)

        graph_op = transformed_tape.operations[0]
        final_op = transformed_tape.operations[-1]

        assert isinstance(graph_op, GraphStatePrep)
        assert isinstance(final_op, gate)

        assert final_op.wires[0] != wire
        assert graph_op.wires[-1] == final_op.wires[0]

    @pytest.mark.parametrize(
        "gate, args",
        [
            (qml.Identity, [3]),
            (qml.Identity, []),
            (qml.GlobalPhase, [1.23]),
            (qml.GlobalPhase, [1.23, 3]),
        ],
    )
    def test_identity_gates_are_supported(self, gate, args):

        initial_op = gate(*args)

        tape = qml.tape.QuantumScript(
            [qml.H(3), initial_op],
            measurements=[qml.sample(wires=[0, 1])],
            shots=1000,
        )

        # after transform, only state prep and MCMs are present on the tape
        (transformed_tape,), _ = convert_to_mbqc_formalism(tape)

        graph_op = transformed_tape.operations[0]
        final_op = transformed_tape.operations[-1]

        assert isinstance(graph_op, GraphStatePrep)
        assert isinstance(final_op, gate)

        if gate is qml.Identity and initial_op.wires:
            assert final_op.wires[0] != 3
            assert graph_op.wires[-1] == final_op.wires[0]
        # both GlobalPhases and the no-wire Identity have no wires, phase is retained
        else:
            assert final_op.wires == ()
            assert final_op.data == initial_op.data

    @pytest.mark.parametrize(
        "measurements", ([qml.expval(qml.X(0))], [qml.sample(wires=0), qml.counts(wires=1)])
    )
    def test_error_raised_for_bad_measurements(self, measurements):

        tape = qml.tape.QuantumScript([qml.H(1), qml.X(0)], measurements=measurements, shots=1000)

        with pytest.raises(
            NotImplementedError,
            match="final measurements have not been converted to a single samples or counts measurement",
        ):
            _, _ = convert_to_mbqc_formalism(tape)

    @pytest.mark.parametrize(
        "mp", (qml.sample(wires=[2, 3]), qml.sample(), qml.counts(wires=[2, 3]), qml.counts())
    )
    def test_tape_wires_if_no_mp_wires(self, mp):

        tape = qml.tape.QuantumScript([qml.H(1), qml.X(0), qml.Y(2)], measurements=[mp], shots=1000)

        (transformed_tape,), _ = convert_to_mbqc_formalism(tape)

        if mp.wires:
            assert len(transformed_tape.measurements[0].wires) == len(mp.wires)
        else:
            assert len(transformed_tape.measurements[0].wires) == len(tape.wires)

    @pytest.mark.slow
    @pytest.mark.parametrize("meas_type", [qml.sample, qml.counts])
    def test_conversion_of_multi_wire_circuit(self, meas_type):

        dev = qml.device("lightning.qubit", shots=500)

        theta = 2.5
        with qml.queuing.AnnotatedQueue() as q:
            RotXZX(theta, 0, theta / 2, 0)
            RotXZX(theta / 2, 0, theta / 4, 1)
            qml.RZ(theta / 3, 0)
            qml.X(0)
            qml.H(1)
            qml.S(1)
            qml.Y(1)
            qml.CNOT([0, 1])

        base_tape = qml.tape.QuantumScript.from_queue(q, shots=500)
        reference_tape = base_tape.copy(
            measurements=[
                qml.expval(qml.X(0)),
                qml.expval(qml.X(1)),
                qml.expval(qml.Y(0)),
                qml.expval(qml.Y(1)),
                qml.expval(qml.Z(0)),
                qml.expval(qml.Z(1)),
            ]
        )

        # after transform, only state prep and MCMs are present on the tape
        (transformed_tape,), _ = convert_to_mbqc_formalism(
            base_tape.copy(measurements=[meas_type(wires=[0, 1])])
        )
        expected_gates = (GraphStatePrep, qml.CZ, qml.measurements.MidMeasureMP)
        for op in transformed_tape.operations:
            if isinstance(op, qml.ops.Conditional):
                assert isinstance(op.base, (qml.X, qml.Z, qml.measurements.MidMeasureMP))
            else:
                assert isinstance(op, expected_gates)

        # diagonalize final measurements, convert to the mbqc formalism, and execute
        res = []
        for obs in (qml.X, qml.Y, qml.Z):
            ops = base_tape.operations + obs(0).diagonalizing_gates() + obs(1).diagonalizing_gates()
            tape = base_tape.copy(operations=ops, measurements=[meas_type(wires=[0, 1])])
            (mbqc_tape,), _ = convert_to_mbqc_formalism(tape)
            (diagonalized_tape,), _ = diagonalize_mcms(mbqc_tape)

            outcome = qml.execute([diagonalized_tape], dev)[0]
            for wire in (0, 1):
                mp = qml.expval(obs(wire))
                if meas_type == qml.sample:
                    res.append(mp.process_samples(outcome, wire_order=[0, 1]))
                else:
                    res.append(mp.process_counts(outcome, wire_order=[0, 1]))

        reference_result = qml.execute([reference_tape], dev)[0]
        assert np.allclose(res, reference_result)
