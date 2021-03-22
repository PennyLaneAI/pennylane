# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the JAX interface"""
import pytest
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
import numpy as np
from functools import partial
import pennylane as qml
from pennylane.tape import JacobianTape
from pennylane.interfaces.jax import JAXInterface


class TestJAXQuantumTape:
    """Test the JAX interface applied to a tape"""

    def test_interface_str(self):
        """Test that the interface string is correctly identified as JAX"""
        with JAXInterface.apply(JacobianTape()) as tape:
            qml.RX(0.5, wires=0)
            qml.expval(qml.PauliX(0))

        assert tape.interface == "jax"
        assert isinstance(tape, JAXInterface)

    def test_get_parameters(self):
        """Test that the get_parameters function correctly gets the trainable parameters and all
        parameters, depending on the trainable_only argument"""
        a = jnp.array(0.1)
        b = jnp.array(0.2)
        c = jnp.array(0.3)
        d = jnp.array(0.4)

        with JAXInterface.apply(JacobianTape()) as tape:
            qml.Rot(a, b, c, wires=0)
            qml.RX(d, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliX(0))

        np.testing.assert_array_equal(tape.get_parameters(), [a, b, c, d])

    def test_execution(self):
        """Test execution"""
        a = jnp.array(0.1)
        b = jnp.array(0.2)

        def cost(a, b, device):
            with JAXInterface.apply(JacobianTape()) as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))
            return tape.execute(device)

        dev = qml.device("default.qubit", wires=1)
        res = cost(a, b, device=dev)
        assert res.shape == (1,)
        # Easiest way to test object is a device array instead of np.array
        assert "DeviceArray" in res.__repr__()


    def test_state_raises(self):
        """Test returning state raises exception"""
        a = jnp.array(0.1)
        b = jnp.array(0.2)

        def cost(a, b, device):
            with JAXInterface.apply(JacobianTape()) as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.state()
            return tape.execute(device)

        dev = qml.device("default.qubit", wires=1)
        # TODO(chase): Make this actually work and not raise an error.
        with pytest.raises(ValueError):
            res = cost(a, b, device=dev)

    def test_execution_with_jit(self):
        """Test execution"""
        a = jnp.array(0.1)
        b = jnp.array(0.2)

        def cost(a, b, device):
            with JAXInterface.apply(JacobianTape()) as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))
            return tape.execute(device)

        # Not a JAX device!
        dev = qml.device("default.qubit", wires=1)
        dev_cost = partial(cost, device=dev)
        res = jax.jit(dev_cost)(a, b)
        assert res.shape == (1,)
        # Easiest way to test object is a device array instead of np.array
        assert "DeviceArray" in res.__repr__()

    def test_qnode_interface(self):

        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="jax")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jnp.array(0.1)
        b = jnp.array(0.2)

        res = circuit(a, b)
        assert "DeviceArray" in res.__repr__()

