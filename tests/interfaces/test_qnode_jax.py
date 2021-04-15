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
import pennylane as qml
from pennylane import qnode, QNode
from pennylane.tape import JacobianTape, QubitParamShiftTape


def test_qnode_intergration():
    """Test a simple use of qnode with a JAX interface and non-JAX device"""
    dev = qml.device("default.mixed", wires=2)  # A non-JAX device

    @qml.qnode(dev, interface="jax")
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RZ(weights[1], wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    weights = jnp.array([0.1, 0.2])
    val = circuit(weights)
    assert "DeviceArray" in val.__repr__()


def test_to_jax():
    """Test the to_jax method"""
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev, interface="autograd")
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RZ(weights[1], wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    circuit.to_jax()
    weights = jnp.array([0.1, 0.2])
    val = circuit(weights)
    assert "DeviceArray" in val.__repr__()


def test_simple_jacobian():
    """Test the use of jax.jaxrev"""
    dev = qml.device("default.mixed", wires=2)  # A non-JAX device.

    @qml.qnode(dev, interface="jax", diff_method="parameter-shift")
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    weights = jnp.array([0.1, 0.2])
    grads = jax.jacrev(circuit)(weights)
    # This is the easiest way to ensure our object is a DeviceArray instead
    # of a numpy array.
    assert "DeviceArray" in grads.__repr__()
    assert grads.shape == (2,)
    np.testing.assert_allclose(grads, np.array([-0.09784342, -0.19767685]))


def test_simple_grad():
    """Test the use of jax.grad"""
    dev = qml.device("default.mixed", wires=2)  # A non-JAX device.

    @qml.qnode(dev, interface="jax", diff_method="parameter-shift")
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RZ(weights[1], wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    weights = jnp.array([0.1, 0.2])
    val = jax.grad(circuit)(weights)
    assert "DeviceArray" in val.__repr__()


@pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff"])
def test_differentiable_expand(diff_method):
    """Test that operation and nested tapes expansion
    is differentiable"""

    class U3(qml.U3):
        def expand(self):
            theta, phi, lam = self.data
            wires = self.wires

            with JacobianTape() as tape:
                qml.Rot(lam, theta, -lam, wires=wires)
                qml.PhaseShift(phi + lam, wires=wires)

            return tape

    dev = qml.device("default.mixed", wires=1)
    a = jnp.array(0.1)
    p = jnp.array([0.1, 0.2, 0.3])

    @qnode(dev, diff_method=diff_method, interface="jax")
    def circuit(a, p):
        qml.RX(a, wires=0)
        U3(p[0], p[1], p[2], wires=0)
        return qml.expval(qml.PauliX(0))

    res = circuit(a, p)

    expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
        np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
    )
    tol = 1e-5
    assert np.allclose(res, expected, atol=tol, rtol=0)

    res = jax.grad(circuit, argnums=1)(a, p)
    expected = np.array(
        [
            np.cos(p[1]) * (np.cos(a) * np.cos(p[0]) - np.sin(a) * np.sin(p[0]) * np.sin(p[2])),
            np.cos(p[1]) * np.cos(p[2]) * np.sin(a)
            - np.sin(p[1]) * (np.cos(a) * np.sin(p[0]) + np.cos(p[0]) * np.sin(a) * np.sin(p[2])),
            np.sin(a) * (np.cos(p[0]) * np.cos(p[1]) * np.cos(p[2]) - np.sin(p[1]) * np.sin(p[2])),
        ]
    )
    assert np.allclose(res, expected, atol=tol, rtol=0)


def qtransform(qnode, a, framework=jnp):
    """Transforms every RY(y) gate in a circuit to RX(-a*cos(y))"""

    def construct(self, args, kwargs):
        """New quantum tape construct method, that performs
        the transform on the tape in a define-by-run manner"""

        t_op = []

        QNode.construct(self, args, kwargs)

        new_ops = []
        for o in self.qtape.operations:
            # here, we loop through all tape operations, and make
            # the transformation if a RY gate is encountered.
            if isinstance(o, qml.RY):
                t_op.append(qml.RX(-a * framework.cos(o.data[0]), wires=o.wires))
                new_ops.append(t_op[-1])
            else:
                new_ops.append(o)

        self.qtape._ops = new_ops
        self.qtape._update()

    import copy

    new_qnode = copy.deepcopy(qnode)
    new_qnode.construct = construct.__get__(new_qnode, QNode)
    return new_qnode


@pytest.mark.parametrize(
    "dev_name,diff_method",
    [("default.mixed", "finite-diff"), ("default.qubit.autograd", "parameter-shift")],
)
def test_transform(dev_name, diff_method, tol):
    """Test an example transform"""
    dev = qml.device(dev_name, wires=1)

    @qnode(dev, interface="jax", diff_method=diff_method)
    def circuit(weights):
        op1 = qml.RY(weights[0], wires=0)
        op2 = qml.RX(weights[1], wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    weights = np.array([0.32, 0.543])
    a = np.array(0.5)

    def loss(weights, a):
        # transform the circuit QNode with trainable weight 'a'
        new_circuit = qtransform(circuit, a)

        # evaluate the transformed QNode
        res = new_circuit(weights)

        # evaluate the original QNode with pre-processed parameters
        res2 = circuit(jnp.sin(weights))

        # return the sum of the two QNode evaluations
        return res + res2

    res = loss(weights, a)

    grad = jax.grad(loss, argnums=[0, 1])(weights, a)
    assert len(grad) == 2
    assert grad[0].shape == weights.shape
    assert grad[1].shape == a.shape

    # compare against the expected values
    tol = 1e-5
    assert np.allclose(res, 1.8244501889992706, atol=tol, rtol=0)
    assert np.allclose(grad[0], [-0.26610258, -0.47053553], atol=tol, rtol=0)
    assert np.allclose(grad[1], 0.06486032, atol=tol, rtol=0)
