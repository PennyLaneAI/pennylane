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
from flaky import flaky

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
    cnot_corrections,
    cnot_measurements,
    queue_cnot,
    queue_corrections,
    queue_measurements,
    queue_single_qubit_gate,
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

    @pytest.mark.parametrize("op", [qml.S(7), qml.H(1), qml.RZ(1.2, 2), RotXZX(1.2, 2.3, 3.4, 2)])
    def test_queue_measurements(self, op):
        """Test that queue_measurements returns MeasurementValues as expected"""

        wires = [i + 10 for i in range(5)]
        mvs = queue_measurements(op, wires=wires)

        assert len(mvs) == 4
        for mv in mvs:
            assert mv.measurements[0].wires[0] in wires

    def test_cnot_measurements(self):
        """Test that cnot_measurements returns MeasurementValues as expected"""

        wires = (10, 11, [i + 12 for i in range(13)])
        mvs = cnot_measurements(wires)

        # expected number of MVs tied to the expected set of wires
        assert len(mvs) == 13
        for mv in mvs:
            assert mv.measurements[0].wires[0] in [i + 10 for i in range(15)]

    def test_queue_measurements_gate_not_implemented(self):
        """Test that a NotImplemented error is raised if queue_single_qubit_gate is
        passed an unsupported operation"""

        with pytest.raises(NotImplementedError, match="Received unsupported gate of type"):
            queue_measurements(qml.Identity, wires="a")

    @pytest.mark.parametrize("op", [qml.S(7), qml.H(1), qml.RZ(1.2, 2), RotXZX(1.2, 2.3, 3.4, 2)])
    def test_queue_corrections(self, op):
        """Test that queue_corrections returns byproduct operators as expected."""

        # Note: this tests the basic behaviour - the accuracy of the returned ops is tested further
        # down by running over many shots and ensuring the average result is as expected.

        # measurement selected because it queues at least one correction for all ops
        correction_function = queue_corrections(op, [0, 1, 1, 0])

        with qml.queuing.AnnotatedQueue() as q:
            correction_function("a")

        assert q.queue
        for byprodcut_op in q.queue:
            assert qml.equal(byprodcut_op, qml.X("a")) or qml.equal(byprodcut_op, qml.Z("a"))

    def test_cnot_corrections(self):
        """Test that the function produced by cnot_corrections queues the byproduct operations as expected"""

        # Note: this tests the basic behaviour - the accuracy of the returned ops is tested further
        # down by running over many shots and ensuring the average result is as expected.

        # measurement selected because it queues all the correction ops
        correction_function = cnot_corrections([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])

        with qml.queuing.AnnotatedQueue() as q:
            correction_function(ctrl_wire="ctrl", target_wire="target")

        assert len(q.queue) == 4
        for op, expected_op in zip(
            q.queue, [qml.Z("ctrl"), qml.X("ctrl"), qml.Z("target"), qml.X("target")]
        ):
            assert qml.equal(op, expected_op)

    def test_queue_corrections_gate_not_implemented(self):
        """Test that a NotImplemented error is raised if queue_single_qubit_gate is
        passed an unsupported operation"""

        with pytest.raises(NotImplementedError, match="Received unsupported gate of type"):
            queue_corrections(qml.Identity, measurements=[0, 0, 0, 0])

    def test_invalid_op_in_tape_raises_error(self):
        """Test that a NotImplemented error is raised if the tape isn't valid for conversion
        to the MBQC formalism using this transform"""

        tape = qml.tape.QuantumScript(
            [qml.H(0), qml.RZ(0.23, 0), qml.RX(1.23, 0)], measurements=[qml.sample(wires=0)]
        )

        with pytest.raises(NotImplementedError, match="unsupported gate"):
            _, _ = convert_to_mbqc_formalism(tape)

    @pytest.mark.parametrize(
        "op",
        [qml.H(2), qml.S(2), qml.RZ(1.23, 2), RotXZX(0, 1.23, 0, 2), RotXZX(0.12, 0.34, 0.56, 2)],
    )
    def test_queue_single_qubit_gate(self, op):
        """Test that the queue_single_qubit_gate function queues state preparation, MCMs
        and byproduct corrections that are equivalent to the input operator"""

        dev = qml.device("lightning.qubit", wires=5)
        q_mgr = QubitMgr(num_qubits=5, start_idx=0)
        wire_map = {2: q_mgr.acquire_qubit()}
        w = op.wires[0]

        ref_tape = qml.tape.QuantumScript(
            [qml.Rot(1.2, 0.34, 0.7, w), op],
            measurements=[qml.expval(qml.X(w)), qml.expval(qml.Y(w)), qml.expval(qml.Z(w))],
        )

        # queue ops with queue_single_qubit_gate
        with qml.queuing.AnnotatedQueue() as q:
            qml.Rot(1.2, 0.34, 0.7, wire_map[w])
            wire_map[w], measurements = queue_single_qubit_gate(
                q_mgr, op=op, in_wire=wire_map[w], diagonalize_mcms=False
            )
            queue_corrections(op, measurements)(wire_map[w])
            qml.expval(qml.X(wire_map[w]))
            qml.expval(qml.Y(wire_map[w]))
            qml.expval(qml.Z(wire_map[w]))

        tape = qml.tape.QuantumScript.from_queue(q, shots=3000)

        # tape contains expected ops
        assert isinstance(tape.operations[1], GraphStatePrep)
        assert isinstance(tape.operations[2], qml.CZ)
        for tape_op in tape.operations[3:]:
            assert isinstance(tape_op, tuple([qml.measurements.MidMeasureMP, qml.ops.Conditional]))

        # tape yields expected results
        with qml.queuing.AnnotatedQueue() as q:
            qml.Rot(1.2, 0.34, 0.7, wire_map[w])
            wire_map[w], measurements = queue_single_qubit_gate(
                q_mgr, op=op, in_wire=wire_map[w], diagonalize_mcms=True
            )
            queue_corrections(op, measurements)(wire_map[w])
            qml.expval(qml.X(wire_map[w]))
            qml.expval(qml.Y(wire_map[w]))
            qml.expval(qml.Z(wire_map[w]))

        diagonalized_tape = qml.tape.QuantumScript.from_queue(q, shots=3000)
        res, res_ref = qml.execute([diagonalized_tape, ref_tape], device=dev, mcm_method="one-shot")
        assert np.allclose(res, res_ref, atol=0.05)

    def test_queue_cnot(self):
        """Test that the queue_cnot function queues state preparation, MCMs and byproduct
        corrections that are equivalent to the input operator"""

        dev = qml.device("lightning.qubit", wires=15, seed=42)
        q_mgr = QubitMgr(num_qubits=15, start_idx=0)

        op = qml.CNOT([2, 3])
        ctrl = op.wires[0]
        target = op.wires[1]
        wire_map = {w: q_mgr.acquire_qubit() for w in op.wires}

        ref_tape = qml.tape.QuantumScript(
            [qml.Rot(1.2, 0.34, 0.7, ctrl), qml.Rot(0.65, 0.43, 0.21, target), op],
            measurements=[
                qml.expval(qml.X(ctrl)),
                qml.expval(qml.Y(ctrl)),
                qml.expval(qml.Z(ctrl)),
                qml.expval(qml.X(target)),
                qml.expval(qml.Y(target)),
                qml.expval(qml.Z(target)),
            ],
        )

        # queue ops with queue_cnot
        with qml.queuing.AnnotatedQueue() as q:
            qml.Rot(1.2, 0.34, 0.7, wire_map[ctrl])
            qml.Rot(0.65, 0.43, 0.21, wire_map[target])
            wire_map[ctrl], wire_map[target], measurements = queue_cnot(
                q_mgr, wire_map[ctrl], wire_map[target]
            )
            cnot_corrections(measurements)(wire_map[ctrl], wire_map[target])
            qml.expval(qml.X(wire_map[ctrl]))
            qml.expval(qml.Y(wire_map[ctrl]))
            qml.expval(qml.Z(wire_map[ctrl]))
            qml.expval(qml.X(wire_map[target]))
            qml.expval(qml.Y(wire_map[target]))
            qml.expval(qml.Z(wire_map[target]))

        tape = qml.tape.QuantumScript.from_queue(q, shots=2000)

        # tape contains expected ops
        assert isinstance(tape.operations[2], GraphStatePrep)
        assert isinstance(tape.operations[3], qml.CZ)
        assert isinstance(tape.operations[4], qml.CZ)
        for op in tape.operations[5:]:
            assert isinstance(op, tuple([qml.measurements.MidMeasureMP, qml.ops.Conditional]))

        # tape yields expected results
        (diagonalized_tape,), _ = diagonalize_mcms(tape)

        res, res_ref = qml.execute([diagonalized_tape, ref_tape], device=dev, mcm_method="one-shot")
        assert np.allclose(res, res_ref, atol=0.1)

    @pytest.mark.parametrize(
        "gate, wire",
        [(qml.X, 2), (qml.Y, 1), (qml.Z, 0)],
    )
    def test_pauli_gates_are_updated(self, gate, wire):
        """Test that the wires on Pauli operators are updated to match the physical
        location of the state after application of the preceding MCM operations"""

        tape = qml.tape.QuantumScript(
            [qml.H(wire), gate(wire)],
            measurements=[qml.sample(wires=[0, 1])],
            shots=1000,
        )

        (transformed_tape,), _ = convert_to_mbqc_formalism(tape)

        graph_op = transformed_tape.operations[0]
        entanglement_op = transformed_tape.operations[1]
        measurements = transformed_tape.operations[2:6]
        byproducts = transformed_tape.operations[6:8]
        final_op = transformed_tape.operations[8]

        # tape consists of converted Hadamard, and the pauli gate
        assert isinstance(graph_op, GraphStatePrep)
        assert isinstance(entanglement_op, qml.CZ)
        for m in measurements:
            assert isinstance(m, qml.measurements.MidMeasureMP)
        for bp in byproducts:
            assert isinstance(bp, qml.ops.Conditional)
        assert isinstance(final_op, gate)

        # the wire the pauli op is applied on is updated to reflect the
        # location of the relevant state after the MBQC operation is applied
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
        """Test that Identity gates are left on the tape as-is"""

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
        "measurements",
        (
            [qml.expval(qml.X(0))],
            [qml.sample(wires=0), qml.counts(wires=2)],
            [qml.sample(wires=0), qml.sample(wires=2)],
        ),
    )
    def test_error_raised_for_bad_measurements(self, measurements):
        """Test that an error is raised if the final output isn't a single sample measurement"""

        tape = qml.tape.QuantumScript([qml.H(1), qml.X(0)], measurements=measurements, shots=1000)

        with pytest.raises(
            NotImplementedError,
            match="final measurements have not been converted to a single samples measurement",
        ):
            _, _ = convert_to_mbqc_formalism(tape)

    @pytest.mark.parametrize("mp", (qml.sample(wires=[2, 3]), qml.sample()))
    def test_tape_wires_if_no_mp_wires(self, mp):
        """Test that wires are taken the measurement process if possible, and otherwise
        taken from the tape (no wires specified is interpreted as "all tape wires")"""

        tape = qml.tape.QuantumScript([qml.H(1), qml.X(0), qml.Y(2)], measurements=[mp], shots=1000)

        (transformed_tape,), _ = convert_to_mbqc_formalism(tape)

        if mp.wires:
            assert len(transformed_tape.measurements[0].wires) == len(mp.wires)
        else:
            assert len(transformed_tape.measurements[0].wires) == len(tape.wires)

    # this test takes 15-20 seconds for a run locally with 500 shots. It tests inherently
    # probabilistic behaviour. The low shot count means a high variance in the results,
    # but we want to ensure they are *usually* quite close to the analytic output, in order
    # to test our protocol for conversion to the MBQC formalism with multiple gates and
    # wires E2E. Following discussion at the FTQC team meeting, we are marking
    # this test as flaky and keeping it here for the time being.
    @flaky(max_runs=5, min_passes=3)
    @pytest.mark.slow
    def test_conversion_of_multi_wire_circuit(self):
        """Test that the transform converts the tape to the expected set of gates
        correctly, and the returned tape continues to produce the expected output"""

        dev = qml.device("lightning.qubit", seed=1234)

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

        base_tape = qml.tape.QuantumScript.from_queue(q)
        tape = base_tape.copy(measurements=[qml.sample(wires=[0, 1])])
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
        (transformed_tape,), _ = convert_to_mbqc_formalism(tape)
        expected_gates = (GraphStatePrep, qml.CZ, qml.measurements.MidMeasureMP, qml.X, qml.Y)
        for op in transformed_tape.operations:
            if isinstance(op, qml.ops.Conditional):
                assert isinstance(op.base, (qml.X, qml.Z, qml.measurements.MidMeasureMP))
            else:
                assert isinstance(op, expected_gates)

        # diagonalize final measurements, convert to the mbqc formalism, and execute
        res = []
        for obs in (qml.X, qml.Y, qml.Z):
            ops = base_tape.operations + obs(0).diagonalizing_gates() + obs(1).diagonalizing_gates()
            tape = base_tape.copy(
                operations=ops, measurements=[qml.sample(wires=[0, 1])], shots=500
            )
            (diagonalized_tape,), _ = convert_to_mbqc_formalism(tape, diagonalize_mcms=True)

            samples = qml.execute([diagonalized_tape], dev)[0]
            for wire in (0, 1):
                mp = qml.expval(obs(wire))
                res.append(mp.process_samples(samples, wire_order=[0, 1]))

        reference_result = qml.execute([reference_tape], dev)[0]

        # analytic results, to 2 s.f., are (-0.40, 0.95, -0.37, -0.25, 0.82, 6.2e-17)
        # an atol of 0.1 is not ideal for comparing to these results, but it's enough
        # to catch changes that modify the results, and we have to choose here between
        # very slow, or fairly noisy
        assert np.allclose(res, reference_result, atol=0.1)
