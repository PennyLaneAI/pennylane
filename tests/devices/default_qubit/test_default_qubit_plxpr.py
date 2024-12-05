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
"""Tests for default qubit executing jaxpr."""

import pytest

import pennylane as qml

jax = pytest.importorskip("jax")
pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


def test_requires_wires():
    """Test that a device error is raised if device wires are not specified."""

    jaxpr = jax.make_jaxpr(lambda x: x + 1)(0.1)
    dev = qml.device("default.qubit")

    with pytest.raises(qml.DeviceError, match="Device wires are required."):
        dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.2)


def test_no_partitioned_shots():
    """Test that an error is raised if the device has partitioned shots."""

    jaxpr = jax.make_jaxpr(lambda x: x + 1)(0.1)
    dev = qml.device("default.qubit", wires=1, shots=(100, 100))

    with pytest.raises(qml.DeviceError, match="Shot vectors are unsupported with jaxpr execution."):
        dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.2)


def test_use_device_prng():
    """Test that sampling depends on the device prng."""

    key1 = jax.random.PRNGKey(1234)
    key2 = jax.random.PRNGKey(1234)

    dev1 = qml.device("default.qubit", wires=1, shots=100, seed=key1)
    dev2 = qml.device("default.qubit", wires=1, shots=100, seed=key2)

    def f():
        qml.H(0)
        return qml.sample(wires=0)

    jaxpr = jax.make_jaxpr(f)()

    samples1 = dev1.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
    samples2 = dev2.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    assert qml.math.allclose(samples1, samples2)


def test_no_prng_key():
    """Test that that sampling works without a provided prng key."""

    dev = qml.device("default.qubit", wires=1, shots=100)

    def f():
        return qml.sample(wires=0)

    jaxpr = jax.make_jaxpr(f)()
    res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
    assert qml.math.allclose(res, jax.numpy.zeros(100))


def test_simple_execution():
    """Test the execution, jitting, and gradient of a simple quantum circuit."""

    def f(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    jaxpr = jax.make_jaxpr(f)(0.123)

    dev = qml.device("default.qubit", wires=1)

    res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)
    assert qml.math.allclose(res, jax.numpy.cos(0.5))
