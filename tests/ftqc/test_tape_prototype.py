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

"""Basic tests and sanity checks for the tape-based execution pipeline prototype. Not intended to be comprehensive."""

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.ftqc import RotXZX
from pennylane.ftqc.ftqc_device import (
    FTQCQubit,
    LightningQubitBackend,
    NullQubitBackend,
    QuantumScriptSequence,
    split_at_non_clifford_gates,
)
from pennylane.measurements import MidMeasureMP, Shots
from pennylane.ops import CZ, RZ, Conditional, H, S, X, Y, Z


@pytest.mark.parametrize(
    "backend_cls, n_wires, name",
    [(LightningQubitBackend, 25, "lightning"), (NullQubitBackend, 1000, "null")],
)
def test_backend_initializes(backend_cls, n_wires, name):
    """Test that the backends initialize successfully"""

    backend = backend_cls()

    assert isinstance(backend.device, qml.devices.Device)
    assert backend.device.name == f"{name}.qubit"
    assert backend.wires == qml.wires.Wires(range(n_wires))
    assert isinstance(backend.capabilities, qml.devices.DeviceCapabilities)


@pytest.mark.parametrize("backend_cls", [LightningQubitBackend, NullQubitBackend])
def test_ftqc_device_initializes(backend_cls):
    """Test that the ftqc.qubit device initializes as expected"""

    backend = backend_cls()
    dev = FTQCQubit(wires=2, backend=backend)

    assert isinstance(dev, qml.devices.Device)
    assert dev.name == "ftqc.qubit"
    assert dev.wires == qml.wires.Wires([0, 1])
    assert isinstance(dev.capabilities, qml.devices.DeviceCapabilities)


@pytest.mark.parametrize("backend_cls", [LightningQubitBackend, NullQubitBackend])
def test_executing_arbitrary_circuit(backend_cls):
    """Test that an arbitrary circuit is preprocessed to be expressed in the
    MBQC formalism before executing, and executes as expected"""

    qml.decomposition.enable_graph()

    backend = backend_cls()
    dev = FTQCQubit(wires=2, backend=backend)

    @qml.set_shots(shots=3000)
    @qml.qnode(dev)
    def circ():
        qml.RY(1.2, 0)
        qml.RZ(0.2, 0)
        qml.RX(0.45, 1)
        return qml.expval(X(0)), qml.expval(Y(0)), qml.expval(Y(1))

    ftqc_circ = qml.qnode(device=dev)(circ)
    ftqc_circ = qml.set_shots(ftqc_circ, shots=1500)

    ref_circ = qml.qnode(device=qml.device("lightning.qubit", wires=2))(circ)

    # the processed circuit is two tapes (split_non_commuting), returning
    # only samples, and expressed in the MBQC formalism
    tapes, _ = qml.workflow.construct_batch(ftqc_circ, level="device")()
    print(tapes)
    assert len(tapes) == 2
    for sequence in tapes:
        assert isinstance(sequence, QuantumScriptSequence)
        assert all(isinstance(mp, qml.measurements.SampleMP) for mp in sequence.measurements)
        for inner_tape in sequence.tapes:
            assert all(
                isinstance(op, (Conditional, CZ, H, MidMeasureMP)) for op in inner_tape.operations
            )

    # circuit executes
    res = ftqc_circ()

    expected_res = (1.0, 1.0, 1.0) if backend_cls is NullQubitBackend else ref_circ()

    assert np.allclose(res, expected_res, atol=0.05)


class TestQuantumScriptSequence:
    """Test the behaviour of the class introduced to represent a QuantumScript
    that is broken into multipe execution steps (potentially with additional runtime
    behaviour between each segment)"""

    def get_test_sequence(self):
        operations = [[X(0), Y(1)], [Y(0), Z(0)], [H(1), S(0)]]
        measurements = [[], [], [qml.expval(X(0))]]
        tapes = [
            qml.tape.QuantumScript(ops, measurements=meas, shots=Shots(1000))
            for ops, meas in zip(operations, measurements)
        ]
        sequence = QuantumScriptSequence(tapes)
        return sequence

    @pytest.mark.parametrize("shots", [None, Shots([10, 10])])
    def test_quantum_sequence_initializes(self, shots):
        """Test that a QuantumScriptSequence can be initialized as expected"""

        ops1 = [X(0), Y(1), qml.RX(1.23, 2)]
        ops2 = [qml.RY(1.2, 0), RZ(0.45, 0), X(1)]
        ops3 = [S(2), H(1), qml.CNOT([1, 0]), S(0)]
        final_measurements = [qml.expval(X(0)), qml.sample(wires=[2])]

        operations = [ops1, ops2, ops3]
        measurements = [[], [], final_measurements]

        tapes = [
            qml.tape.QuantumScript(ops, measurements=meas, shots=Shots(1000))
            for ops, meas in zip(operations, measurements)
        ]

        sequence = QuantumScriptSequence(tapes, shots=shots)

        # wires are as expected
        assert sequence.num_wires == 3
        assert sequence.wires == qml.wires.Wires([0, 1, 2])

        # shots are as expected
        assert sequence.shots == Shots(1000) if shots is None else shots
        for tape in sequence.tapes:
            assert tape.shots == Shots(1)

        # stored tapes are as expected
        assert len(sequence.tapes) == 3
        assert len(sequence.intermediate_tapes) == 2
        assert isinstance(sequence.final_tape, qml.tape.QuantumScript)

        # overall measurements and operations are as expected
        assert sequence.measurements == final_measurements
        for op_list1, op_list2 in zip(sequence.operations, operations):
            assert op_list1 == op_list2

        assert repr(sequence) == "<QuantumScriptSequence: wires=[0, 1, 2]>"

    def test_sequence_copy(self):
        """Test that copying a sequence works as expected with no updates"""

        sequence = self.get_test_sequence()
        copied_sequence = sequence.copy()

        # tapes are all the same, shots are all the same
        for tape1, tape2 in zip(copied_sequence.tapes, sequence.tapes):
            qml.assert_equal(tape1, tape2)
        assert copied_sequence.shots == sequence.shots

    def test_sequence_copy_updates_measurements(self):
        """Test that copying a sequence works as expected when updating measurements"""

        sequence = self.get_test_sequence()
        copied_sequence = sequence.copy(measurements=[qml.var(Y(12))])

        # intermediate tapes and shots are all the same
        for tape1, tape2 in zip(copied_sequence.intermediate_tapes, sequence.intermediate_tapes):
            qml.assert_equal(tape1, tape2)
        assert copied_sequence.shots == sequence.shots

        # final tapes have the same operations, but the copy has updated measurements
        assert copied_sequence.final_tape.operations == sequence.final_tape.operations
        assert (
            copied_sequence.measurements
            == copied_sequence.final_tape.measurements
            == [qml.var(Y(12))]
        )

        # wires are updated accordingly when measurement adds a new wire
        assert copied_sequence.num_wires == 3
        assert copied_sequence.wires == qml.wires.Wires([0, 1, 12])

    def test_sequence_copy_updates_tapes(self):
        """Test that copying a sequence works as expected when updating tapes"""

        sequence = self.get_test_sequence()

        # get some new, modified tapes and make an updated copy
        qml.decomposition.enable_graph()
        new_tapes = []
        for tape in sequence.tapes:
            new_tape, fn = qml.transforms.decompose(tape, gate_set={qml.Rot, qml.GlobalPhase})
            new_tapes.append(fn(new_tape))

        copied_sequence = sequence.copy(tapes=new_tapes)

        # none of the copied tapes are the same, and they've all been decomposed
        for tape1, tape2 in zip(copied_sequence.tapes, sequence.tapes):
            assert not qml.equal(tape1, tape2)
            assert isinstance(tape1.operations[0], qml.Rot)

        # shots aren't affected
        assert copied_sequence.shots == sequence.shots

    def test_sequence_copy_updates_shots(self):
        """Test that copying a sequence works as expected when updating shots"""

        sequence = self.get_test_sequence()
        copied_sequence = sequence.copy(shots=Shots(15))

        # tapes still match, but shots are udpated
        # note that all tapes inside the sequence have shots=1, and
        # the overall shots for the execution are only stored on the sequence
        for tape1, tape2 in zip(copied_sequence.tapes, sequence.tapes):
            qml.assert_equal(tape1, tape2)
        assert copied_sequence.shots != sequence.shots
        assert copied_sequence.shots == Shots(15)

    def test_changing_ops_in_copy_raises_error(self):
        """Test that trying to update operations or set copy_operations to True
        when copying a QuantumScriptSequence raises an error"""

        operations = [[X(0), Y(1)], [Y(0), Z(0)], [H(1), S(0)]]
        measurements = [[], [], [qml.expval(X(0))]]
        tapes = [
            qml.tape.QuantumScript(ops, measurements=meas, shots=Shots(1000))
            for ops, meas in zip(operations, measurements)
        ]
        sequence = QuantumScriptSequence(tapes)

        with pytest.raises(RuntimeError, match="Can't use copy_operations"):
            _ = sequence.copy(copy_operations=True)

        with pytest.raises(TypeError, match="cannot update 'operations'"):
            _ = sequence.copy(operations=[X(1), Y(12)])

    def test_changing_measurements_and_tapes_in_copy_raises_error(self):
        """Test that trying to update both measurements and or tapes
        when copying a QuantumScriptSequence raises an error"""

        operations = [[X(0), Y(1)], [Y(0), Z(0)], [H(1), S(0)]]
        measurements = [[], [], [qml.expval(X(0))]]
        tapes = [
            qml.tape.QuantumScript(ops, measurements=meas, shots=Shots(1000))
            for ops, meas in zip(operations, measurements)
        ]
        sequence = QuantumScriptSequence(tapes)

        with pytest.raises(RuntimeError, match="Can't update tapes and measurements"):
            _ = sequence.copy(measurements=[qml.expval(X(0))], tapes=tapes)

    def test_get_standard_wire_map(self):
        """Test that getting the standard wire map works as expected"""

        tape1 = qml.tape.QuantumScript([X(12), Y("a"), Z(0)])
        tape2 = qml.tape.QuantumScript([H(1)], measurements=[qml.expval(Y(3))])

        sequence = QuantumScriptSequence([tape1, tape2], shots=Shots(100))

        # pylint: disable=protected-access
        assert sequence._get_standard_wire_map() == {12: 0, "a": 1, 0: 2, 1: 3, 3: 4}

    def test_map_to_standard_wires(self):
        """Test that map_to_standard_wires correctly affects both the overall
        sequence wires and the underlying tape wires"""

        tape1 = qml.tape.QuantumScript([X(12), Y("a"), Z(0), S("b")])
        tape2 = qml.tape.QuantumScript([H(1)], measurements=[qml.expval(Y("b"))])
        sequence = QuantumScriptSequence([tape1, tape2], shots=Shots(100))

        new_sequence = sequence.map_to_standard_wires()

        assert new_sequence.tapes[0].operations == [X(0), Y(1), Z(2), S(3)]
        assert new_sequence.tapes[1].operations == [H(4)]
        assert new_sequence.measurements == new_sequence.tapes[1].measurements == [qml.expval(Y(3))]

    def test_split_at_non_clifford_gates_creates_sequence(self):
        """Test that split_at_non_clifford_gates splits the tape before
        each non-Clifford gate and returns a QuantumScriptSequence"""

        dev = qml.device("lightning.qubit", wires=3)

        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circ():
            RotXZX(0.64, 0.33, 1.2, 0)
            RotXZX(0.16, 0.93, 0.29, 1)
            H(0)
            S(1)
            RZ(0.54, 1)
            H(0)
            S(1)
            H(2)
            RotXZX(0.82, 0.66, 0.26, 2)
            H(2)
            H(0)
            S(1)
            return qml.expval(X(0)), qml.expval(Y(2))

        tape = qml.workflow.construct_tape(circ)()
        sequence, fn = split_at_non_clifford_gates(tape)
        sequence = fn(sequence)
        sequence, fn = split_at_non_clifford_gates(tape)
        sequence = fn(sequence)

        assert isinstance(sequence, QuantumScriptSequence)
        assert tape.measurements == sequence.measurements

        assert len(sequence.tapes) == 4
        assert sequence.tapes[0].operations == [RotXZX(0.64, 0.33, 1.2, 0)]
        assert sequence.tapes[1].operations == [RotXZX(0.16, 0.93, 0.29, 1), H(0), S(1)]
        assert sequence.tapes[2].operations == [RZ(0.54, 1), H(0), S(1), H(2)]
        assert sequence.tapes[3].operations == [RotXZX(0.82, 0.66, 0.26, 2), H(2), H(0), S(1)]

    def test_executing_sequence(self):
        """Test that a QuantumScriptSequence can be executed with a backend, and that
        LightningQubitBackend.execute matches the results for executing the standard
        circuit on lightning.qubit"""

        dev = qml.device("lightning.qubit", wires=3)
        backend = LightningQubitBackend()

        @qml.qnode(dev)
        def circ():
            RotXZX(0.64, 0.33, 1.2, 0)
            RotXZX(0.16, 0.93, 0.29, 1)
            H(0)
            S(1)
            RZ(0.54, 1)
            H(0)
            S(1)
            H(2)
            RotXZX(0.82, 0.66, 0.26, 2)
            H(2)
            H(0)
            S(1)
            return qml.expval(X(0)), qml.expval(Y(2))

        # execute basic circuit on lightning.qubit for expected results
        expected_result = circ()

        shots_circ = qml.set_shots(circ, 5000)
        tape = qml.workflow.construct_tape(shots_circ)()
        sequence, _ = split_at_non_clifford_gates(tape)
        sequence, _ = split_at_non_clifford_gates(tape)

        raw_samples = backend.execute(
            [
                sequence[0],
                sequence[0],
            ],
            ExecutionConfig(),
        )
        sequence_results = np.average(raw_samples[0], axis=0)

        # expected results are approximately -0.24 and -0.59, a tolerance of 0.05 is fine
        assert np.allclose(expected_result, sequence_results, atol=0.05)


class TestBackendExecution:

    def test_lightning_backend_execution_with_sequences(self, mocker):

        tape1 = qml.tape.QuantumScript([qml.RX(0.21, 0)])
        tape2 = qml.tape.QuantumScript([qml.RX(0.21, 0)])
        tape3 = qml.tape.QuantumScript(
            [qml.RX(0.21, 0)], measurements=[qml.expval(Y(0)), qml.expval(Z(0))]
        )

        sequence = QuantumScriptSequence([tape1, tape2, tape3], shots=3000)

        backend = LightningQubitBackend()
        spy_simulate = mocker.spy(backend, "simulate")

        results = backend.execute([sequence, sequence, sequence], ExecutionConfig())

        assert len(results) == 3

        # all results are as expected - backend state is correctly maintained during a
        # sequence execution and reset between executions
        for res in results:
            assert len(res) == 3000
            assert np.allclose(np.average(res, axis=0), [-np.sin(0.63), np.cos(0.63)], atol=0.05)

        # executed 3 sequences of 3 segments, with 3000 shots each
        assert spy_simulate.call_count == 3 * 3 * 3000

    def test_null_qubit_backend_execution(self):
        """Test execution of QuantumScriptSequences succeeds with the null.qubit
        backend."""

        tape1 = qml.tape.QuantumScript([H(0)])
        tape2 = qml.tape.QuantumScript([H(0)], measurements=[qml.expval(Y(0)), qml.expval(Z(0))])

        sequence = QuantumScriptSequence([tape1, tape1, tape2], shots=3000)

        backend = NullQubitBackend()

        results = backend.execute([sequence, sequence, sequence], ExecutionConfig())

        assert len(results) == 3
        for res in results:
            assert len(res) == 2
            assert np.allclose(res, 0)
