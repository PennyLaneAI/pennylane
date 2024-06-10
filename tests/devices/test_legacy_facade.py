# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Contains unit tests for the LegacyDeviceFacade class.
"""
# pylint: disable=protected-access
import numpy as np

import pennylane as qml
from pennylane.devices.legacy_facade import (
    LegacyDeviceFacade,
    legacy_device_batch_transform,
    legacy_device_expand_fn,
    set_shots,
)


class DummyDevice(qml.devices.LegacyDevice):

    author = "some string"
    name = "my legacy device"
    short_name = "something"
    version = 0.0

    observables = {}
    operations = {"RX", "RY", "RZ", "PauliX", "PauliY", "PauliZ", "CNOT"}
    pennylane_requires = 0.38

    def reset(self):
        pass

    def apply(self, operation, wires, par):
        return 0.0

    def expval(self, observable, wires, par):
        return 0.0


def test_shots():
    """Test the shots behavior of a dummy legacy device."""
    legacy_dev = DummyDevice(shots=(100, 100))
    dev = LegacyDeviceFacade(legacy_dev)

    assert dev.shots == qml.measurements.Shots((100, 100))

    with set_shots(legacy_dev, 50):
        assert legacy_dev.shots == 50

    assert legacy_dev._shot_vector == list(qml.measurements.Shots((100, 100)).shot_vector)
    assert legacy_dev.shots == 200
    assert dev.shots == qml.measurements.Shots((100, 100))


def test_legacy_device_expand_fn():
    """Test that legacy_device_expand_fn expands operations to the target gateset."""

    tape = qml.tape.QuantumScript([qml.X(0), qml.IsingXX(0.5, wires=(0, 1))], [qml.state()])

    expected = qml.tape.QuantumScript(
        [qml.X(0), qml.CNOT((0, 1)), qml.RX(0.5, 0), qml.CNOT((0, 1))], [qml.state()]
    )
    (new_tape,), fn = legacy_device_expand_fn(tape, device=DummyDevice())
    assert qml.equal(new_tape, expected)
    assert fn(("A",)) == "A"


def test_legacy_device_batch_transform():
    """Test that legacy_device_batch_transform still performs the operations that the legacy batch transform did."""

    tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0) + qml.Y(0))])
    (tape1, tape2), fn = legacy_device_batch_transform(tape, device=DummyDevice())

    assert qml.equal(tape1, qml.tape.QuantumScript([], [qml.expval(qml.X(0))]))
    assert qml.equal(tape2, qml.tape.QuantumScript([], [qml.expval(qml.Y(0))]))
    assert fn((1.0, 2.0)) == np.array(3.0)


def test_batch_transform_supports_hamiltonian():
    """Assert that the legacy device batch transform absorbs the shot sensitive behavior of the
    hamiltonian expansion."""

    class HamiltonianDev(DummyDevice):

        observables = {"Hamiltonian"}

    H = qml.Hamiltonian([1, 1], [qml.X(0), qml.Y(0)])

    tape = qml.tape.QuantumScript([], [qml.expval(H)], shots=None)
    batch, _ = legacy_device_batch_transform(tape, device=HamiltonianDev())
    assert len(batch) == 1

    tape = qml.tape.QuantumScript([], [qml.expval(H)], shots=50)
    batch, _ = legacy_device_batch_transform(tape, device=HamiltonianDev())
    assert len(batch) == 2


def test_basic_properties():
    """Test the basic properties of the device."""

    ld = DummyDevice(wires=(0, 1))
    d = LegacyDeviceFacade(ld)
    assert d.target_device is ld
    assert d.name == "something"
    assert d.author == "some string"
    assert d.version == 0.0
    assert repr(d) == f"<LegacyDeviceFacade: {repr(ld)}>"
    assert d.wires == qml.wires.Wires((0, 1))


def test_preprocessing_program():
    """Test the population of the preprocessing program."""

    dev = DummyDevice(wires=(0, 1))
    program, _ = LegacyDeviceFacade(dev).preprocess()

    assert program[0].transform == legacy_device_batch_transform.transform
    assert program[1].transform == legacy_device_expand_fn.transform
    assert program[2].transform == qml.defer_measurements.transform

    m0 = qml.measure(0)
    tape = qml.tape.QuantumScript(
        [qml.X(0), qml.IsingXX(0.5, (0, 1)), *m0.measurements],
        [qml.expval(qml.Hamiltonian([1, 1], [qml.X(0), qml.Y(0)]))],
        shots=50,
    )

    (tape1, tape2), fn = program((tape,))
    expected_ops = [qml.X(0), qml.CNOT((0, 1)), qml.RX(0.5, 0), qml.CNOT((0, 1)), qml.CNOT((0, 2))]
    assert tape1.operations == expected_ops
    assert tape2.operations == expected_ops

    assert tape1.measurements == [qml.expval(qml.X(0))]
    assert tape2.measurements == [qml.expval(qml.Y(0))]
    assert tape1.shots == qml.measurements.Shots(50)
    assert tape2.shots == qml.measurements.Shots(50)

    assert qml.math.allclose(fn((1.0, 2.0)), 3.0)


def test_preprocessing_program_supports_mid_measure():
    """Test that if the device natively supports mid measure, defer_measurements wont be applied."""

    class MidMeasureDev(DummyDevice):

        _capabilities = {"supports_mid_measure": True}

    dev = MidMeasureDev()
    program, _ = LegacyDeviceFacade(dev).preprocess()
    assert qml.defer_measurements not in program
