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
from copy import deepcopy

import numpy as np
import pytest

import pennylane as qml
import pennylane.transforms.insert_ops
from pennylane.measurements import Expectation
from pennylane.tape import QuantumTape
from pennylane.transforms.insert_ops import insert


class TestInsert:
    """Tests for the insert function using input tapes"""

    with QuantumTape() as tape:
        qml.RX(0.9, wires=0)
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.5, wires=0)
        qml.RX(0.6, wires=1)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    with QuantumTape() as tape_with_prep:
        qml.QubitStateVector([1, 0], wires=0)
        qml.RX(0.9, wires=0)
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.5, wires=0)
        qml.RX(0.6, wires=1)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    with QuantumTape() as custom_tape:
        qml.RX(0.9, wires=0)
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.5, wires=0)
        qml.PauliZ(wires=1)
        qml.Identity(wires=2)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def test_multiwire_op(self):
        """Tests if a ValueError is raised when multiqubit operations are requested"""
        with pytest.raises(ValueError, match="Only single-qubit operations can be inserted into"):
            insert(qml.CNOT, [])(self.tape)

    @pytest.mark.parametrize("pos", [1, ["all", qml.RY, int], "ABC", str])
    def test_invalid_position(self, pos):
        """Test if a ValueError is raised when an invalid position is requested"""
        with pytest.raises(ValueError, match="Position must be either 'start', 'end', or 'all'"):
            insert(qml.AmplitudeDamping, 0.4, position=pos)(self.tape)

    def test_start(self):
        """Test if the expected tape is returned when the start position is requested"""
        tape = insert(qml.AmplitudeDamping, 0.4, position="start")(self.tape)

        with QuantumTape() as tape_exp:
            qml.AmplitudeDamping(0.4, wires=0)
            qml.AmplitudeDamping(0.4, wires=1)
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    def test_all(self):
        """Test if the expected tape is returned when the all position is requested"""
        tape = insert(qml.PhaseDamping, 0.4, position="all")(self.tape)

        with QuantumTape() as tape_exp:
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

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    def test_before(self):
        """Test if the expected tape is returned when the before argument is True"""
        tape = insert(qml.PhaseDamping, 0.4, position="all", before=True)(self.tape)

        with QuantumTape() as tape_exp:
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

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    op_lst = [qml.RX, qml.PauliZ, qml.Identity]

    @pytest.mark.parametrize("op", op_lst)
    def test_operation_as_position(self, op):
        """Test if expected tape is returned when an operation is passed in position"""
        tape = insert(qml.PhaseDamping, 0.4, position=op, before=True)(self.custom_tape)

        with QuantumTape() as tape_exp:
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

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    def test_operation_list_as_position(self):
        """Test if expected tape is returned when an operation list is passed in position"""
        tape = insert(qml.PhaseDamping, 0.4, position=[qml.RX, qml.RY])(self.tape)

        with QuantumTape() as tape_exp:
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

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    def test_end(self):
        """Test if the expected tape is returned when the end position is requested"""
        tape = insert(qml.GeneralizedAmplitudeDamping, [0.4, 0.5], position="end")(self.tape)

        with QuantumTape() as tape_exp:
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.GeneralizedAmplitudeDamping(0.4, 0.5, wires=0)
            qml.GeneralizedAmplitudeDamping(0.4, 0.5, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    def test_start_with_state_prep(self):
        """Test if the expected tape is returned when the start position is requested in a tape
        that has state preparation"""
        tape = insert(qml.AmplitudeDamping, 0.4, position="start")(self.tape_with_prep)

        with QuantumTape() as tape_exp:
            qml.QubitStateVector([1, 0], wires=0)
            qml.AmplitudeDamping(0.4, wires=0)
            qml.AmplitudeDamping(0.4, wires=1)
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    def test_all_with_state_prep(self):
        """Test if the expected tape is returned when the all position is requested in a tape
        that has state preparation"""
        tape = insert(qml.PhaseDamping, 0.4, position="all")(self.tape_with_prep)

        with QuantumTape() as tape_exp:
            qml.QubitStateVector([1, 0], wires=0)
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

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    def test_end_with_state_prep(self):
        """Test if the expected tape is returned when the end position is requested in a tape
        that has state preparation"""
        tape = insert(qml.GeneralizedAmplitudeDamping, [0.4, 0.5], position="end")(
            self.tape_with_prep
        )

        with QuantumTape() as tape_exp:
            qml.QubitStateVector([1, 0], wires=0)
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.GeneralizedAmplitudeDamping(0.4, 0.5, wires=0)
            qml.GeneralizedAmplitudeDamping(0.4, 0.5, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    def test_with_qfunc_op(self):
        """Test if the transform works as expected if the operation is a qfunc rather than single
        operation"""

        def op(x, y, wires):
            qml.RX(x, wires=wires)
            qml.PhaseShift(y, wires=wires)

        tape = insert(op, [0.4, 0.5], position="end")(self.tape)

        with QuantumTape() as tape_exp:
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

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation


def test_insert_qnode():
    """Test that a QNode with the insert decorator gives a different result than one
    without."""
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    @insert(qml.AmplitudeDamping, 0.2, position="end")
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


def test_insert_dev(mocker, monkeypatch):
    """Test if a device transformed by the insert function does successfully add noise to
    subsequent circuit executions"""
    with QuantumTape() as in_tape:
        qml.RX(0.9, wires=0)
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.5, wires=0)
        qml.RX(0.6, wires=1)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        qml.expval(qml.PauliZ(0))

    dev = qml.device("default.mixed", wires=2)
    res_without_noise = qml.execute([in_tape], dev, qml.gradients.param_shift)

    new_dev = insert(qml.PhaseDamping, 0.4)(dev)
    spy = mocker.spy(new_dev, "default_expand_fn")

    res_with_noise = qml.execute([in_tape], new_dev, qml.gradients.param_shift)
    tape = spy.call_args[0][0]

    with QuantumTape() as tape_exp:
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
        qml.expval(qml.PauliZ(0))

    assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
    assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
    assert all(
        np.allclose(o1.parameters, o2.parameters)
        for o1, o2 in zip(tape.operations, tape_exp.operations)
    )
    assert len(tape.measurements) == 2
    assert tape.observables[0].name == ["PauliZ", "PauliZ"]
    assert tape.observables[0].wires.tolist() == [0, 1]
    assert tape.measurements[0].return_type is Expectation
    assert tape.observables[1].name == "PauliZ"
    assert tape.observables[1].wires.tolist() == [0]
    assert tape.measurements[1].return_type is Expectation

    assert not np.allclose(res_without_noise, res_with_noise)


def test_insert_template():
    """Test that ops are inserted correctly into a decomposed template"""
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    @insert(qml.PhaseDamping, 0.3, position="all")
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


def test_insert_decorator_doesnt_cause_deque_error():
    """Test that the insert transform catches and reports errors from the enclosed function."""

    def noise(noise_param, wires):
        qml.CRX(noise_param, wires=[0, 1])
        qml.CNOT(wires=[1, 0])

    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    @qml.transforms.insert(noise, 0.3, position="all")
    def noisy_circuit(circuit_param):
        qml.RY(circuit_param, wires=0)
        qml.Hadamard(wires=0)
        qml.T(wires=0)
        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))

    try:
        noisy_circuit(0.4)
        assert False
    except Exception as e:
        # This tape's expansion fails, but shouldn't cause a downstream IndexError. See issue #3103
        assert not isinstance(e, IndexError)
        assert isinstance(e, qml.QuantumFunctionError)
