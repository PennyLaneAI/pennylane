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
This module tests the default qubit interpreter.
"""
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")
pytestmark = pytest.mark.jax

# must be below the importorskip
# pylint: disable=wrong-import-position
from pennylane.devices.qubit.dq_interpreter import DefaultQubitInterpreter


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


def test_initialization():
    dq = DefaultQubitInterpreter(num_wires=3, shots=None)
    assert dq.num_wires == 3
    assert dq.shots == qml.measurements.Shots(None)
    assert isinstance(dq.initial_key, jax.numpy.ndarray)
    assert dq.stateref is None


def test_setup():
    key = jax.random.PRNGKey(1234)
    dq = DefaultQubitInterpreter(num_wires=2, shots=2, key=key)
    assert dq.stateref is None

    dq.setup()
    assert isinstance(dq.stateref, dict)
    assert list(dq.stateref.keys()) == ["state", "key", "mcms"]

    assert dq.stateref["key"] is key
    assert dq.key is key

    assert dq.stateref["mcms"] == {}
    assert dq.mcms is dq.stateref["mcms"]

    assert dq.state is dq.stateref["state"]
    expected = jax.numpy.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    assert qml.math.allclose(dq.state, expected)

    dq.cleanup()


def test_simple_execution():

    @DefaultQubitInterpreter(num_wires=1, shots=None)
    def f(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    res = f(0.5)
    assert qml.math.allclose(res, jax.numpy.cos(0.5))

    g = jax.grad(f)(jax.numpy.array(0.5))
    assert qml.math.allclose(g, -jax.numpy.sin(0.5))


def test_sampling():

    @DefaultQubitInterpreter(num_wires=2, shots=10)
    def sampler():
        qml.X(0)
        return qml.sample(wires=(0, 1))

    results = sampler()

    expected0 = jax.numpy.ones((10,))  # zero wire
    expected1 = jax.numpy.zeros((10,))  # one wire
    expected = jax.numpy.hstack([expected0, expected1]).T

    assert qml.math.allclose(results, expected)
