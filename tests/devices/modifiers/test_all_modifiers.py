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
Tests that apply to all device modifiers or act on a combination of them together.
"""
# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring, no-member
import pytest

import pennylane as qml
from pennylane.devices import Device
from pennylane.devices.modifiers import simulator_tracking, single_tape_support


# pylint: disable=protected-access
def test_chained_modifiers():
    """Test that modifiers can be stacked together."""

    @simulator_tracking
    @single_tape_support
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: qml.devices.ExecutionConfig | None = None):
            return tuple(0.0 for _ in circuits)

    assert DummyDev._applied_modifiers == [single_tape_support, simulator_tracking]

    tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0))], shots=50)
    dev = DummyDev()

    with dev.tracker:
        out = dev.execute(tape)

    # result unwrapped
    assert out == 0.0

    assert len(dev.tracker.history) == 7
    assert dev.tracker.history["batches"] == [1]
    assert dev.tracker.history["simulations"] == [1]
    assert dev.tracker.history["executions"] == [1]
    assert dev.tracker.history["results"] == [0.0]
    assert dev.tracker.history["resources"] == [tape.specs["resources"]]
    assert dev.tracker.history["shots"] == [50]


@pytest.mark.parametrize("modifier", (simulator_tracking, single_tape_support))
class TestModifierDefaultBeahviour:
    """Test generic behavior for device modifiers."""

    def test_error_on_old_interface(self, modifier):
        """Test that a ValueError is raised is called on something that is not a subclass of Device."""

        with pytest.raises(ValueError, match=f"{modifier.__name__} only accepts"):
            modifier(qml.devices.DefaultQutrit)

    def test_adds_to_applied_modifiers_private_property(self, modifier):
        """Test that the modifier is added to the `_applied_modifiers` property."""

        @modifier
        class DummyDev(qml.devices.Device):

            def execute(
                self, circuits, execution_config: qml.devices.ExecutionConfig | None = None
            ):
                return 0.0

        assert DummyDev._applied_modifiers == [modifier]

        @modifier
        class DummyDev2(qml.devices.Device):

            _applied_modifiers = [None]  # some existing value

            def execute(
                self, circuits, execution_config: qml.devices.ExecutionConfig | None = None
            ):
                return 0.0

        assert DummyDev2._applied_modifiers == [None, modifier]

    def test_leaves_undefined_methods_untouched(self, modifier):
        """Test that undefined methods are left the same as the Device class methods."""

        @modifier
        class DummyDev(qml.devices.Device):

            def execute(
                self, circuits, execution_config: qml.devices.ExecutionConfig | None = None
            ):
                return 0.0

        assert DummyDev.compute_derivatives == Device.compute_derivatives
        assert DummyDev.execute_and_compute_derivatives == Device.execute_and_compute_derivatives
        assert DummyDev.compute_jvp == Device.compute_jvp
        assert DummyDev.execute_and_compute_jvp == Device.execute_and_compute_jvp
        assert DummyDev.compute_vjp == Device.compute_vjp
        assert DummyDev.execute_and_compute_vjp == Device.execute_and_compute_vjp

        dev = DummyDev()
        assert not dev.supports_derivatives()
        assert not dev.supports_jvp()
        assert not dev.supports_vjp()
