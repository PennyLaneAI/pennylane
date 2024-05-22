# Copyright 2024 Xanadu Quantum Technologies Inc.

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
from pennylane.measurements import Expectation
from pennylane.tape import QuantumScript
from pennylane.transforms.add_noise import add_noise


class TestAddNoise:
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

    # conditionals
    c0, c1 = qml.noise.op_eq(qml.RX), qml.noise.op_in([qml.RY, qml.RZ])
    c2 = qml.noise.op_eq("StatePrep")

    # callables
    @staticmethod
    def n0(op, **kwargs):  # pylint: disable=unused-argument
        """Mapped callable for c0"""
        qml.RZ(op.parameters[0] * 0.05, op.wires)
        qml.apply(op)
        qml.RZ(-op.parameters[0] * 0.05, op.wires)

    n1 = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

    @staticmethod
    def n2(op, **kwargs):
        """Mapped callable for c2"""
        qml.ThermalRelaxationError(0.4, kwargs["t1"], 0.2, 0.6, op.wires)

    noise_model = qml.NoiseModel({c0: n0, c1: n1})
    noise_model_with_prep = noise_model + qml.NoiseModel({c2: n2}, t1=0.4)

    def test_noise_model_error(self):
        """Tests if a ValueError is raised when noise model is not given"""
        with pytest.raises(
            ValueError, match="Argument noise_model must be an instance of NoiseModel"
        ):
            add_noise(self.tape, {})

    def test_level_error(self):
        """Tests if a NotImplementedError is raised when level is given"""
        with pytest.raises(
            NotImplementedError, match="Support for level argument is not currently present"
        ):
            add_noise(self.tape, qml.NoiseModel({}), level="device")

    def test_noise_tape(self):
        """Test if the expected tape is returned with the transform"""
        [tape], _ = add_noise(self.tape, self.noise_model)

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.RZ(0.045, wires=0)
            qml.RX(0.9, wires=0)
            qml.RZ(-0.045, wires=0)
            qml.RY(0.4, wires=1)
            qml.AmplitudeDamping(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.AmplitudeDamping(0.4, wires=0)
            qml.RZ(0.03, wires=1)
            qml.RX(0.6, wires=1)
            qml.RZ(-0.03, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        tape_exp = QuantumScript.from_queue(q_tape_exp)

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert (
            tape.observables[0].name == "Prod"
            if qml.operation.active_new_opmath()
            else ["PauliZ", "PauliZ"]
        )
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    def test_noise_tape_with_state_prep(self):
        """Test if the expected tape is returned with the transform"""
        [tape], _ = add_noise(self.tape_with_prep, self.noise_model_with_prep)

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.StatePrep([1, 0], wires=0)
            qml.ThermalRelaxationError(0.4, 0.4, 0.2, 0.6, wires=0)
            qml.RZ(0.045, wires=0)
            qml.RX(0.9, wires=0)
            qml.RZ(-0.045, wires=0)
            qml.RY(0.4, wires=1)
            qml.AmplitudeDamping(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.AmplitudeDamping(0.4, wires=0)
            qml.RZ(0.03, wires=1)
            qml.RX(0.6, wires=1)
            qml.RZ(-0.03, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        tape_exp = QuantumScript.from_queue(q_tape_exp)

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert (
            tape.observables[0].name == "Prod"
            if qml.operation.active_new_opmath()
            else ["PauliZ", "PauliZ"]
        )
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation


def test_add_noise_qnode():
    """Test that a QNode with add_noise decorator gives a different result."""
    dev = qml.device("default.mixed", wires=2)

    c, n = qml.noise.op_in([qml.RY, qml.RZ]), qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

    @partial(add_noise, noise_model=qml.NoiseModel({c: n}))
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


def test_add_noise_dev():
    """Test if an device transformed by the add_noise function does successfully add noise to
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
    dev = qml.device("default.qubit", wires=2)
    program, _ = dev.preprocess()
    res_without_noise = qml.execute(
        [in_tape], dev, qml.gradients.param_shift, transform_program=program
    )

    c, n = qml.noise.op_in([qml.RX, qml.RY]), qml.noise.partial_wires(qml.PhaseShift, 0.4)
    new_dev = add_noise(dev, noise_model=qml.NoiseModel({c: n}))
    new_program, _ = new_dev.preprocess()
    [tape], _ = new_program([in_tape])
    res_with_noise = qml.execute([in_tape], new_dev, qml.gradients, transform_program=new_program)

    with qml.queuing.AnnotatedQueue() as q_tape_exp:
        qml.RX(0.9, wires=0)
        qml.PhaseShift(0.4, wires=0)
        qml.RY(0.4, wires=1)
        qml.PhaseShift(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
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
    assert (
        tape.observables[0].name == "Prod"
        if qml.operation.active_new_opmath()
        else ["PauliZ", "PauliZ"]
    )
    assert tape.observables[0].wires.tolist() == [0, 1]
    assert tape.measurements[0].return_type is Expectation
    assert tape.observables[1].name == "PauliZ"
    assert tape.observables[1].wires.tolist() == [0]
    assert tape.measurements[1].return_type is Expectation

    assert not np.allclose(res_without_noise, res_with_noise)


def test_add_noise_old_dev(mocker):
    """Test if a old device transformed by the add_noise function does successfully add noise to
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
    dev = qml.device("default.mixed", wires=2)
    res_without_noise = qml.execute([in_tape], dev, qml.gradients.param_shift)

    c, n = qml.noise.op_in([qml.RX, qml.RY]), qml.noise.partial_wires(qml.PhaseDamping, 0.4)
    new_dev = add_noise(dev, noise_model=qml.NoiseModel({c: n}))
    spy = mocker.spy(new_dev, "default_expand_fn")

    res_with_noise = qml.execute([in_tape], new_dev, qml.gradients.param_shift)
    tape = spy.call_args[0][0]

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
        qml.expval(qml.PauliZ(0))
    tape_exp = QuantumScript.from_queue(q_tape_exp)

    assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
    assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
    assert all(
        np.allclose(o1.parameters, o2.parameters)
        for o1, o2 in zip(tape.operations, tape_exp.operations)
    )
    assert len(tape.measurements) == 2
    assert (
        tape.observables[0].name == "Prod"
        if qml.operation.active_new_opmath()
        else ["PauliZ", "PauliZ"]
    )
    assert tape.observables[0].wires.tolist() == [0, 1]
    assert tape.measurements[0].return_type is Expectation
    assert tape.observables[1].name == "PauliZ"
    assert tape.observables[1].wires.tolist() == [0]
    assert tape.measurements[1].return_type is Expectation

    assert not np.allclose(res_without_noise, res_with_noise)


def test_add_noise_template():
    """Test that ops are inserted correctly into a decomposed template"""
    dev = qml.device("default.mixed", wires=2)

    c, n = qml.noise.op_in([qml.RX, qml.RY]), qml.noise.partial_wires(qml.PhaseDamping, 0.3)

    @partial(add_noise, noise_model=qml.NoiseModel({c: n}))
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
        qml.RY(w2[0][0][0], wires=0)
        qml.PhaseDamping(0.3, wires=0)
        qml.RY(w2[0][0][1], wires=1)
        qml.PhaseDamping(0.3, wires=1)
        return qml.expval(qml.PauliZ(0))

    w1 = np.random.random(2)
    w2 = np.random.random((1, 1, 2))

    assert np.allclose(f1(w1, w2), f2(w1, w2))
