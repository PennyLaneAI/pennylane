# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Tests for the basic default behavior of the Device API.
"""
# pylint:disable=unused-argument
import pytest

import pennylane as qml
from pennylane.devices.experimental import Device, ExecutionConfig, DefaultExecutionConfig
from pennylane.wires import Wires


def test_execute_method_abstract():
    """Test that a device can't be instantiated without an execute method."""

    # pylint: disable=too-few-public-methods
    class BadDevice(Device):
        """A bad device"""

    with pytest.raises(TypeError, match=r"instantiate abstract class BadDevice"):
        BadDevice()  # pylint: disable=abstract-class-instantiated


class TestMinimalDevice:
    """Tests for a device with only a minimal execute provided."""

    # pylint: disable=too-few-public-methods
    class MinimalDevice(Device):
        """A device with only a dummy execute method provided."""

        # pylint:disable=unused-argnument
        def execute(self, circuits, execution_config=DefaultExecutionConfig):
            return (0,)

    dev = MinimalDevice()

    def test_device_name(self):
        """Test the default name is the name of the class"""
        assert self.dev.name == "MinimalDevice"

    @pytest.mark.parametrize(
        "wires,shots,expected",
        [
            (None, None, "<MinimalDevice device at 0x"),
            ([1, 3], None, "<MinimalDevice device (wires=2) at 0x"),
            (None, [10, 20], "<MinimalDevice device (shots=30) at 0x"),
            ([1, 3], [10, 20], "<MinimalDevice device (wires=2, shots=30) at 0x"),
        ],
    )
    def test_repr(self, wires, shots, expected):
        """Tests the repr of the device API"""
        assert repr(self.MinimalDevice(wires=wires, shots=shots)).startswith(expected)

    def test_shots(self):
        """Test default behavior for shots."""

        assert self.dev.shots == qml.measurements.Shots(None)

        shots_dev = self.MinimalDevice(shots=100)
        assert shots_dev.shots == qml.measurements.Shots(100)

        with pytest.raises(AttributeError):
            self.dev.shots = 100  # pylint: disable=attribute-defined-outside-init

    def test_tracker_set_on_initialization(self):
        """Test that a new tracker instance is initialized with the class."""
        assert isinstance(self.dev.tracker, qml.Tracker)
        assert self.dev.tracker is not self.MinimalDevice.tracker

    def test_preprocess_single_circuit(self):
        """Test that preprocessing wraps a circuit into a batch."""

        circuit1 = qml.tape.QuantumScript()
        program, config = self.dev.preprocess()
        batch, fn = program((circuit1,))
        assert isinstance(batch, tuple)
        assert len(batch) == 1
        assert batch[0] is circuit1
        assert callable(fn)

        a = (1,)
        assert fn(a) == (1,)
        assert config is qml.devices.experimental.DefaultExecutionConfig

    def test_preprocess_batch_circuits(self):
        """Test that preprocessing a batch doesn't do anything."""

        circuit = qml.tape.QuantumScript()
        in_config = ExecutionConfig()
        in_batch = (circuit, circuit)
        program, config = self.dev.preprocess(in_config)
        batch, fn = program(in_batch)
        assert batch is in_batch
        assert config is in_config
        a = (1, 2)
        assert fn(a) is a

    def test_supports_derivatives_default(self):
        """Test that the default behavior of supports derivatives is false."""

        assert not self.dev.supports_derivatives()
        assert not self.dev.supports_derivatives(ExecutionConfig())

    def test_compute_derivatives_notimplemented(self):
        """Test that compute derivatives raises a notimplementederror."""

        with pytest.raises(NotImplementedError):
            self.dev.compute_derivatives(qml.tape.QuantumScript())

        with pytest.raises(NotImplementedError):
            self.dev.execute_and_compute_derivatives(qml.tape.QuantumScript())

    def test_supports_jvp_default(self):
        """Test that the default behaviour of supports_jvp is false."""
        assert not self.dev.supports_jvp()

    def test_compute_jvp_not_implemented(self):
        """Test that compute_jvp is not implemented by default."""
        with pytest.raises(NotImplementedError):
            self.dev.compute_jvp(qml.tape.QuantumScript(), (0.1,))

        with pytest.raises(NotImplementedError):
            self.dev.execute_and_compute_jvp(qml.tape.QuantumScript(), (0.1,))

    def test_supports_vjp_default(self):
        """Test that the default behavior of supports_jvp is false."""
        assert not self.dev.supports_vjp()

    def test_compute_vjp_not_implemented(self):
        """Test that compute_vjp is not implemented by default."""
        with pytest.raises(NotImplementedError):
            self.dev.compute_vjp(qml.tape.QuantumScript(), (0.1,))

        with pytest.raises(NotImplementedError):
            self.dev.execute_and_compute_vjp(qml.tape.QuantumScript(), (0.1,))

    @pytest.mark.parametrize(
        "wires, expected",
        [
            (None, None),
            (0, Wires([])),
            (Wires([0]), Wires([0])),
            (1, Wires([0])),
            ([1], Wires([1])),
            (2, Wires([0, 1])),
            ([1, 3], Wires([1, 3])),
        ],
    )
    def test_wires_can_be_provided(self, wires, expected):
        """Test that a device can be created with wires."""
        assert self.MinimalDevice(wires=wires).wires == expected

    def test_wires_are_read_only(self):
        """Test that device wires cannot be set after device initialization."""
        with pytest.raises(AttributeError):
            self.dev.wires = [0, 1]  # pylint:disable=attribute-defined-outside-init


class TestProvidingDerivatives:
    """Tests logic when derivatives, vjp, or jvp are overridden."""

    def test_provided_derivative(self):
        """Tests default logic for a device with a derivative provided."""

        class WithDerivative(Device):
            """A device with a derivative."""

            # pylint: disable=unused-argument
            def execute(self, circuits, execution_config: ExecutionConfig = DefaultExecutionConfig):
                return "a"

            def compute_derivatives(
                self, circuits, execution_config: ExecutionConfig = DefaultExecutionConfig
            ):
                return ("b",)

        dev = WithDerivative()
        assert dev.supports_derivatives()
        assert not dev.supports_derivatives(ExecutionConfig(derivative_order=2))
        assert not dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"))
        assert dev.supports_derivatives(ExecutionConfig(gradient_method="device"))

        out = dev.execute_and_compute_derivatives(qml.tape.QuantumScript())
        assert out[0] == "a"
        assert out[1] == ("b",)

    def test_provided_jvp(self):
        """Tests default logic for a device with a jvp provided."""

        # pylint: disable=unused-argnument
        class WithJvp(Device):
            """A device with a jvp."""

            def execute(self, circuits, execution_config: ExecutionConfig = DefaultExecutionConfig):
                return "a"

            def compute_jvp(
                self, circuits, tangents, execution_config: ExecutionConfig = DefaultExecutionConfig
            ):
                return ("c",)

        dev = WithJvp()
        assert dev.supports_jvp()

        out = dev.execute_and_compute_jvp(qml.tape.QuantumScript(), (1.0,))
        assert out[0] == "a"
        assert out[1] == ("c",)

    def test_provided_vjp(self):
        """Tests default logic for a device with a vjp provided."""

        # pylint: disable=unused-argnument
        class WithVjp(Device):
            """A device with a vjp."""

            def execute(self, circuits, execution_config: ExecutionConfig = DefaultExecutionConfig):
                return "a"

            def compute_vjp(
                self,
                circuits,
                cotangents,
                execution_config: ExecutionConfig = DefaultExecutionConfig,
            ):
                return ("c",)

        dev = WithVjp()
        assert dev.supports_vjp()

        out = dev.execute_and_compute_vjp(qml.tape.QuantumScript(), (1.0,))
        assert out[0] == "a"
        assert out[1] == ("c",)
