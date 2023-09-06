# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the measurements module"""
import pytest

import pennylane as qml
from pennylane.measurements import (
    ClassicalShadowMP,
    MeasurementProcess,
    MeasurementTransform,
    SampleMeasurement,
    SampleMP,
    Shots,
    StateMeasurement,
    StateMP,
    expval,
    var,
)

# pylint: disable=too-few-public-methods, unused-argument


class NotValidMeasurement(MeasurementProcess):
    @property
    def return_type(self):
        return "NotValidReturnType"


def test_no_measure():
    """Test that failing to specify a measurement
    raises an exception"""
    dev = qml.device("default.qubit.legacy", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.PauliY(0)

    with pytest.raises(qml.QuantumFunctionError, match="must return either a single measurement"):
        _ = circuit(0.65)


def test_shape_unrecognized_error():
    """Test that querying the shape of a measurement process with an
    unrecognized return type raises an error."""
    dev = qml.device("default.qubit.legacy", wires=2)
    mp = NotValidMeasurement()
    with pytest.raises(
        qml.QuantumFunctionError,
        match="The shape of the measurement NotValidMeasurement is not defined",
    ):
        mp.shape(dev, Shots(None))


@pytest.mark.parametrize("stat_func", [expval, var])
def test_not_an_observable(stat_func):
    """Test that a UserWarning is raised if the provided
    argument might not be hermitian."""

    dev = qml.device("default.qubit.legacy", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.RX(0.52, wires=0)
        return stat_func(qml.prod(qml.PauliX(0), qml.PauliZ(0)))

    with pytest.warns(UserWarning, match="Prod might not be hermitian."):
        _ = circuit()


class TestSampleMeasurement:
    """Tests for the SampleMeasurement class."""

    def test_custom_sample_measurement(self):
        """Test the execution of a custom sampled measurement."""

        class MyMeasurement(SampleMeasurement):
            # pylint: disable=signature-differs
            def process_samples(self, samples, wire_order, shot_range, bin_size):
                return qml.math.sum(samples[..., self.wires])

        dev = qml.device("default.qubit.legacy", wires=2, shots=1000)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return MyMeasurement(wires=[0]), MyMeasurement(wires=[1])

        assert qml.math.allequal(circuit(), [1000, 0])

    def test_sample_measurement_without_shots(self):
        """Test that executing a sampled measurement with ``shots=None`` raises an error."""

        class MyMeasurement(SampleMeasurement):
            # pylint: disable=signature-differs
            def process_samples(self, samples, wire_order, shot_range, bin_size):
                return qml.math.sum(samples[..., self.wires])

        dev = qml.device("default.qubit.legacy", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return MyMeasurement(wires=[0]), MyMeasurement(wires=[1])

        with pytest.raises(
            ValueError, match="Shots must be specified in the device to compute the measurement "
        ):
            circuit()

    def test_method_overriden_by_device(self):
        """Test that the device can override a measurement process."""

        dev = qml.device("default.qubit.legacy", wires=2, shots=1000)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.sample(wires=[0]), qml.sample(wires=[1])

        circuit.device.measurement_map[SampleMP] = "test_method"
        circuit.device.test_method = lambda obs, shot_range=None, bin_size=None: 2

        assert qml.math.allequal(circuit(), [2, 2])


class TestStateMeasurement:
    """Tests for the SampleMeasurement class."""

    def test_custom_state_measurement(self):
        """Test the execution of a custom state measurement."""

        class MyMeasurement(StateMeasurement):
            def process_state(self, state, wire_order):
                return qml.math.sum(state)

        dev = qml.device("default.qubit.legacy", wires=2)

        @qml.qnode(dev)
        def circuit():
            return MyMeasurement()

        assert circuit() == 1

    def test_sample_measurement_with_shots(self):
        """Test that executing a state measurement with shots raises a warning."""

        class MyMeasurement(StateMeasurement):
            def process_state(self, state, wire_order):
                return qml.math.sum(state)

        dev = qml.device("default.qubit.legacy", wires=2, shots=1000)

        @qml.qnode(dev)
        def circuit():
            return MyMeasurement()

        with pytest.warns(
            UserWarning,
            match="Requested measurement MyMeasurement with finite shots",
        ):
            circuit()

    def test_method_overriden_by_device(self):
        """Test that the device can override a measurement process."""

        dev = qml.device("default.qubit.legacy", wires=2)

        @qml.qnode(dev, interface="autograd")
        def circuit():
            return qml.state()

        circuit.device.measurement_map[StateMP] = "test_method"
        circuit.device.test_method = lambda obs, shot_range=None, bin_size=None: 2

        assert circuit() == 2


class TestMeasurementTransform:
    """Tests for the MeasurementTransform class."""

    def test_custom_measurement(self):
        """Test the execution of a custom measurement."""

        class MyMeasurement(MeasurementTransform):
            def process(self, tape, device):
                return {device.shots: len(tape)}

        dev = qml.device("default.qubit.legacy", wires=2, shots=1000)

        @qml.qnode(dev)
        def circuit():
            return MyMeasurement()

        assert circuit() == {dev.shots: len(circuit.tape)}

    def test_method_overriden_by_device(self):
        """Test that the device can override a measurement process."""

        dev = qml.device("default.qubit.legacy", wires=2, shots=1000)

        @qml.qnode(dev)
        def circuit():
            return qml.classical_shadow(wires=0)

        circuit.device.measurement_map[ClassicalShadowMP] = "test_method"
        circuit.device.test_method = lambda tape: 2

        assert circuit() == 2
