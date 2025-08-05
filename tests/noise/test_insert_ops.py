# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Tests for the insert transform.
"""
from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.noise.insert_ops import insert
from pennylane.tape import QuantumScript


class TestInsert:
    """Tests for the insert function using input tapes"""

    with qml.queuing.AnnotatedQueue() as q_tape:
        qml.RX(0.9, wires=0)
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.5, wires=0)
        qml.RX(0.6, wires=1)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    tape = QuantumScript.from_queue(q_tape)
    with qml.queuing.AnnotatedQueue() as q_tape_with_prep:
        qml.StatePrep([1, 0], wires=0)
        qml.RX(0.9, wires=0)
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.5, wires=0)
        qml.RX(0.6, wires=1)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    tape_with_prep = QuantumScript.from_queue(q_tape_with_prep)
    with qml.queuing.AnnotatedQueue() as q_custom_tape:
        qml.RX(0.9, wires=0)
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.5, wires=0)
        qml.PauliZ(wires=1)
        qml.Identity(wires=2)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    custom_tape = QuantumScript.from_queue(q_custom_tape)

    def test_multiwire_op(self):
        """Tests if a ValueError is raised when multiqubit operations are requested"""
        with pytest.raises(ValueError, match="Only single-qubit operations can be inserted into"):
            insert(self.tape, qml.CNOT, [])

    @pytest.mark.parametrize("pos", [1, ["all", qml.RY, int], "ABC", str])
    def test_invalid_position(self, pos):
        """Test if a ValueError is raised when an invalid position is requested"""
        with pytest.raises(ValueError, match="Position must be either 'start', 'end', or 'all'"):
            insert(self.tape, qml.AmplitudeDamping, 0.4, position=pos)

    def test_start(self):
        """Test if the expected tape is returned when the start position is requested"""
        tapes, _ = insert(self.tape, qml.AmplitudeDamping, 0.4, position="start")
        tape = tapes[0]

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.AmplitudeDamping(0.4, wires=0)
            qml.AmplitudeDamping(0.4, wires=1)
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape_exp = QuantumScript.from_queue(q_tape_exp)
        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)

    def test_all(self):
        """Test if the expected tape is returned when the all position is requested"""
        tapes, _ = insert(self.tape, qml.PhaseDamping, 0.4, position="all")
        tape = tapes[0]

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.RX(0.9, wires=0)
            qml.PhaseDamping(0.4, wires=0)
            qml.RY(0.4, wires=1)
            qml.PhaseDamping(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseDamping(0.4, wires=0)
            qml.PhaseDamping(0.4, wires=1)
            qml.RY(0.5, wires=0)
            qml.PhaseDamping(0.4, wires=0)
            qml.RX(0.6, wires=1)
            qml.PhaseDamping(0.4, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape_exp = QuantumScript.from_queue(q_tape_exp)
        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)

    def test_before(self):
        """Test if the expected tape is returned when the before argument is True"""
        tapes, _ = insert(self.tape, qml.PhaseDamping, 0.4, position="all", before=True)
        tape = tapes[0]
        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.PhaseDamping(0.4, wires=0)
            qml.RX(0.9, wires=0)
            qml.PhaseDamping(0.4, wires=1)
            qml.RY(0.4, wires=1)
            qml.PhaseDamping(0.4, wires=0)
            qml.PhaseDamping(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseDamping(0.4, wires=0)
            qml.RY(0.5, wires=0)
            qml.PhaseDamping(0.4, wires=1)
            qml.RX(0.6, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape_exp = QuantumScript.from_queue(q_tape_exp)
        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)

    op_lst = [qml.RX, qml.PauliZ, qml.Identity]

    @pytest.mark.parametrize("op", op_lst)
    def test_operation_as_position(self, op):
        """Test if expected tape is returned when an operation is passed in position"""
        tapes, _ = insert(self.custom_tape, qml.PhaseDamping, 0.4, position=op, before=True)
        tape = tapes[0]

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            if op == qml.RX:
                qml.PhaseDamping(0.4, wires=0)
            qml.RX(0.9, wires=0)

            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)

            if op == qml.PauliZ:
                qml.PhaseDamping(0.4, wires=1)
            qml.PauliZ(wires=1)

            if op == qml.Identity:
                qml.PhaseDamping(0.4, wires=2)
            qml.Identity(wires=2)

            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape_exp = QuantumScript.from_queue(q_tape_exp)
        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)

    def test_operation_list_as_position(self):
        """Test if expected tape is returned when an operation list is passed in position"""
        tapes, _ = insert(self.tape, qml.PhaseDamping, 0.4, position=[qml.RX, qml.RY])
        tape = tapes[0]

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.RX(0.9, wires=0)
            qml.PhaseDamping(0.4, wires=0)
            qml.RY(0.4, wires=1)
            qml.PhaseDamping(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.PhaseDamping(0.4, wires=0)
            qml.RX(0.6, wires=1)
            qml.PhaseDamping(0.4, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape_exp = QuantumScript.from_queue(q_tape_exp)
        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)

    def test_end(self):
        """Test if the expected tape is returned when the end position is requested"""
        tapes, _ = insert(self.tape, qml.GeneralizedAmplitudeDamping, [0.4, 0.5], position="end")
        tape = tapes[0]

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.GeneralizedAmplitudeDamping(0.4, 0.5, wires=0)
            qml.GeneralizedAmplitudeDamping(0.4, 0.5, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape_exp = QuantumScript.from_queue(q_tape_exp)
        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)

    def test_start_with_state_prep(self):
        """Test if the expected tape is returned when the start position is requested in a tape
        that has state preparation"""
        tapes, _ = insert(self.tape_with_prep, qml.AmplitudeDamping, 0.4, position="start")
        tape = tapes[0]

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.StatePrep([1, 0], wires=0)
            qml.AmplitudeDamping(0.4, wires=0)
            qml.AmplitudeDamping(0.4, wires=1)
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape_exp = QuantumScript.from_queue(q_tape_exp)
        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)

    def test_all_with_state_prep(self):
        """Test if the expected tape is returned when the all position is requested in a tape
        that has state preparation"""
        tapes, _ = insert(self.tape_with_prep, qml.PhaseDamping, 0.4, position="all")
        tape = tapes[0]

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.StatePrep([1, 0], wires=0)
            qml.RX(0.9, wires=0)
            qml.PhaseDamping(0.4, wires=0)
            qml.RY(0.4, wires=1)
            qml.PhaseDamping(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseDamping(0.4, wires=0)
            qml.PhaseDamping(0.4, wires=1)
            qml.RY(0.5, wires=0)
            qml.PhaseDamping(0.4, wires=0)
            qml.RX(0.6, wires=1)
            qml.PhaseDamping(0.4, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape_exp = QuantumScript.from_queue(q_tape_exp)
        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)

    def test_end_with_state_prep(self):
        """Test if the expected tape is returned when the end position is requested in a tape
        that has state preparation"""
        tapes, _ = insert(
            self.tape_with_prep, qml.GeneralizedAmplitudeDamping, [0.4, 0.5], position="end"
        )
        tape = tapes[0]

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.StatePrep([1, 0], wires=0)
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.GeneralizedAmplitudeDamping(0.4, 0.5, wires=0)
            qml.GeneralizedAmplitudeDamping(0.4, 0.5, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape_exp = QuantumScript.from_queue(q_tape_exp)
        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)

    def test_with_qfunc_op(self):
        """Test if the transform works as expected if the operation is a qfunc rather than single
        operation"""

        def op(x, y, wires):
            qml.RX(x, wires=wires)
            qml.PhaseShift(y, wires=wires)

        tapes, _ = insert(self.tape, op=op, op_args=[0.4, 0.5], position="end")
        tape = tapes[0]

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.RX(0.4, wires=0)
            qml.PhaseShift(0.5, wires=0)
            qml.RX(0.4, wires=1)
            qml.PhaseShift(0.5, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape_exp = QuantumScript.from_queue(q_tape_exp)
        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)


def test_insert_qnode():
    """Test that a QNode with the insert decorator gives a different result than one
    without."""
    dev = qml.device("default.mixed", wires=2)

    @partial(insert, op=qml.AmplitudeDamping, op_args=0.2, position="end")
    @qml.qnode(dev)
    def f_noisy(w, x, y, z):
        qml.RX(w, wires=0)
        qml.RY(x, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=0)
        qml.RX(z, wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    @qml.qnode(dev)
    def f(w, x, y, z):
        qml.RX(w, wires=0)
        qml.RY(x, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=0)
        qml.RX(z, wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    args = [0.1, 0.2, 0.3, 0.4]

    assert not np.isclose(f_noisy(*args), f(*args))


@pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
def test_insert_dev(dev_name):
    """Test if an device transformed by the insert function does successfully add noise to
    subsequent circuit executions"""
    with qml.queuing.AnnotatedQueue() as q_in_tape:
        qml.RX(0.9, wires=0)
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.5, wires=0)
        qml.RX(0.6, wires=1)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        qml.expval(qml.PauliZ(0))

    in_tape = QuantumScript.from_queue(q_in_tape)
    dev = qml.device(dev_name, wires=2)

    program = dev.preprocess_transforms()
    res_without_noise = qml.execute(
        [in_tape], dev, qml.gradients.param_shift, transform_program=program
    )

    new_dev = insert(dev, qml.PhaseShift, 0.4)
    new_program = new_dev.preprocess_transforms()
    tapes, _ = new_program([in_tape])
    tape = tapes[0]
    res_with_noise = qml.execute(
        [in_tape], new_dev, qml.gradients.param_shift, transform_program=new_program
    )

    with qml.queuing.AnnotatedQueue() as q_tape_exp:
        qml.RX(0.9, wires=0)
        qml.PhaseShift(0.4, wires=0)
        qml.RY(0.4, wires=1)
        qml.PhaseShift(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(0.4, wires=0)
        qml.PhaseShift(0.4, wires=1)
        qml.RY(0.5, wires=0)
        qml.PhaseShift(0.4, wires=0)
        qml.RX(0.6, wires=1)
        qml.PhaseShift(0.4, wires=1)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        qml.expval(qml.PauliZ(0))

    tape_exp = QuantumScript.from_queue(q_tape_exp)
    assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
    assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
    assert all(
        np.allclose(o1.parameters, o2.parameters)
        for o1, o2 in zip(tape.operations, tape_exp.operations)
    )
    assert len(tape.measurements) == 2
    assert tape.observables[0].name == "Prod"
    assert tape.observables[0].wires.tolist() == [0, 1]
    assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)
    assert tape.observables[1].name == "PauliZ"
    assert tape.observables[1].wires.tolist() == [0]
    assert isinstance(tape.measurements[1], qml.measurements.ExpectationMP)

    assert not np.allclose(res_without_noise, res_with_noise)


def test_insert_template():
    """Test that ops are inserted correctly into a decomposed template"""
    dev = qml.device("default.mixed", wires=2)

    @partial(insert, op=qml.PhaseDamping, op_args=0.3, position="all")
    @qml.qnode(dev)
    def f1(w1, w2):
        qml.SimplifiedTwoDesign(w1, w2, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev)
    def f2(w1, w2):
        qml.RY(w1[0], wires=0)
        qml.PhaseDamping(0.3, wires=0)
        qml.RY(w1[1], wires=1)
        qml.PhaseDamping(0.3, wires=1)
        qml.CZ(wires=[0, 1])
        qml.PhaseDamping(0.3, wires=0)
        qml.PhaseDamping(0.3, wires=1)
        qml.RY(w2[0][0][0], wires=0)
        qml.PhaseDamping(0.3, wires=0)
        qml.RY(w2[0][0][1], wires=1)
        qml.PhaseDamping(0.3, wires=1)
        return qml.expval(qml.PauliZ(0))

    w1 = np.random.random(2)
    w2 = np.random.random((1, 1, 2))

    assert np.allclose(f1(w1, w2), f2(w1, w2))


def test_insert_transform_works_with_non_qwc_obs():
    """Test that the insert transform catches and reports errors from the enclosed function."""

    def op(noise_param, wires):
        # pylint: disable=unused-argument
        qml.CRX(noise_param, wires=[0, 1])
        qml.CNOT(wires=[1, 0])

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    @partial(insert, op=op, op_args=0.3, position="all")
    def noisy_circuit(circuit_param):
        qml.RY(circuit_param, wires=0)
        qml.Hadamard(wires=0)
        qml.T(wires=0)
        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))

    @qml.qnode(dev)
    def explicit_circuit(circuit_param):
        qml.RY(circuit_param, wires=0)
        op(0.3, None)
        qml.Hadamard(wires=0)
        op(0.3, None)
        qml.T(wires=0)
        op(0.3, None)
        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))

    assert np.allclose(noisy_circuit(0.4), explicit_circuit(0.4))
