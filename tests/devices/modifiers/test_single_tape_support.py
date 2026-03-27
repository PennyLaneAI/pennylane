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
Tests the `single_tape_support` device modifier.

"""
from typing import Optional

# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring
import pennylane as qml
from pennylane.devices.modifiers import single_tape_support


def test_wraps_execute():
    """Test that execute now accepts a single circuit."""

    @single_tape_support
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return tuple(0.0 for _ in circuits)

    t = qml.tape.QuantumScript()
    dev = DummyDev()
    assert dev.execute(t) == 0.0


def test_wraps_compute_derivatives():
    """Test that compute_derivatives now accepts a single circuit."""

    @single_tape_support
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return tuple(0.0 for _ in circuits)

        def compute_derivatives(
            self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None
        ):
            return tuple("a" for _ in circuits)

    t = qml.tape.QuantumScript()
    dev = DummyDev()
    assert dev.compute_derivatives(t) == "a"


def test_wraps_execute_and_compute_derivatives():
    """Test that execute_and_compute_derivatives now accepts a single circuit."""

    @single_tape_support
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return tuple(0.0 for _ in circuits)

        def execute_and_compute_derivatives(
            self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None
        ):
            return tuple("a" for _ in circuits), tuple("b" for _ in circuits)

    t = qml.tape.QuantumScript()
    dev = DummyDev()
    assert dev.execute_and_compute_derivatives(t) == ("a", "b")
    assert dev.execute_and_compute_derivatives((t,)) == (("a",), ("b",))


def test_wraps_compute_jvp():
    """Test that compute_jvp now accepts a single circuit."""

    @single_tape_support
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return tuple(0.0 for _ in circuits)

        def compute_jvp(
            self, circuits, tangents, execution_config: Optional[qml.devices.ExecutionConfig] = None
        ):
            assert len(tangents) == len(circuits)
            return tuple("a" for _ in circuits)

    t = qml.tape.QuantumScript()
    dev = DummyDev()
    assert dev.compute_jvp(t, tangents=(1, 1, 1, 1)) == "a"
    assert dev.compute_jvp((t,), tangents=((1, 2, 3),)) == ("a",)


def test_wraps_execute_and_compute_jvp():
    """Test that execute_and_compute_jvp now accepts a single circuit."""

    @single_tape_support
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return tuple(0.0 for _ in circuits)

        def execute_and_compute_jvp(
            self, circuits, tangents, execution_config: Optional[qml.devices.ExecutionConfig] = None
        ):
            assert len(tangents) == len(circuits)
            return tuple("a" for _ in circuits), tuple("b" for _ in circuits)

    t = qml.tape.QuantumScript()
    dev = DummyDev()
    assert dev.execute_and_compute_jvp(t, tangents=(1, 1, 1, 1)) == ("a", "b")
    assert dev.execute_and_compute_jvp((t,), tangents=((1, 2, 3),)) == (("a",), ("b",))


def test_wraps_compute_vjp():
    """Test that compute_vjp now accepts a single circuit."""

    @single_tape_support
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return tuple(0.0 for _ in circuits)

        def compute_vjp(
            self,
            circuits,
            cotangents,
            execution_config: Optional[qml.devices.ExecutionConfig] = None,
        ):
            assert len(cotangents) == len(circuits)
            return tuple("a" for _ in circuits)

    t = qml.tape.QuantumScript()
    dev = DummyDev()
    assert dev.compute_vjp(t, cotangents=(1, 1, 1, 1)) == "a"
    assert dev.compute_vjp((t,), cotangents=((1, 2, 3),)) == ("a",)


def test_wraps_execute_and_compute_vjp():
    """Test that execute_and_compute_vjp now accepts a single circuit."""

    @single_tape_support
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return tuple(0.0 for _ in circuits)

        def execute_and_compute_vjp(
            self,
            circuits,
            cotangents,
            execution_config: Optional[qml.devices.ExecutionConfig] = None,
        ):
            assert len(cotangents) == len(circuits)
            return tuple("a" for _ in circuits), tuple("b" for _ in circuits)

    t = qml.tape.QuantumScript()
    dev = DummyDev()
    assert dev.execute_and_compute_vjp(t, cotangents=(1, 1, 1, 1)) == ("a", "b")
    assert dev.execute_and_compute_vjp((t,), cotangents=((1, 2, 3),)) == (("a",), ("b",))
