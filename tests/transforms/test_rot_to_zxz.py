"""
Tests for the rot_to_zxz transforms.
"""

import pytest

import pennylane as qml
import tensorflow as tf
import torch

from pennylane.numpy import pi

from pennylane.transforms import rot_to_zxz

# A simple circuit to test correctness
def circuit(phi0, theta0, omega0):
    qml.Hadamard(wires=0)
    qml.Rot(phi0, theta0, omega0, wires=0)
    return qml.state()


phi0, theta0, omega0 = pi, pi / 3, 5 * pi / 7


class TestDecomposeRotIntoZXZ:
    """Tests to ensure that the transform is correct."""

    def test_rot_to_zxz(self):
        """Test that the transform is correct."""
        dev = qml.device("default.qubit", wires=2)
        state_in = qml.QNode(circuit, dev)(phi0, theta0, omega0)
        state_trans = qml.QNode(rot_to_zxz(circuit), dev)(phi0, theta0, omega0)

        assert qml.math.allclose(state_in, state_trans)

    def test_rot_to_zxz_tf(self):
        """Test that the transform is correct in the tf interface."""
        phi0_tf = tf.Variable(phi0)
        theta0_tf = tf.Variable(theta0)
        omega0_tf = tf.Variable(omega0)

        dev = qml.device("default.qubit", wires=2)
        qnode_in = qml.QNode(circuit, dev, interface="tf")
        qnode_trans = qml.QNode(rot_to_zxz(circuit), dev, interface="tf")

        state_in = qnode_in(phi0_tf, theta0_tf, omega0_tf)
        state_trans = qnode_trans(phi0_tf, theta0_tf, omega0_tf)

        assert qml.math.allclose(state_in, state_trans)


#     def test_rot_to_zxz_torch(self):
#         """Test that the transform is correct in the torch interface."""
#         phi0_torch = torch.tensor(phi0, dtype=torch.float64,
#                                   requires_grad=True)
#         theta0_torch = torch.tensor(theta0, dtype=torch.float64,
#                                     requires_grad=True)
#         omega0_torch = torch.tensor(omega0, dtype=torch.float64,
#                                     requires_grad=True)

#         dev = qml.device("default.qubit", wires=2)
#         qnode_in = qml.QNode(circuit, dev, interface="torch")
#         qnode_trans = qml.QNode(rot_to_zxz(circuit), dev, interface="torch")

#         state_in = qnode_in(phi0_torch, theta0_torch, omega0_torch)
#         state_trans = qnode_trans(phi0_torch, theta0_torch, omega0_torch)

#         assert qml.math.allclose(state_in, state_trans)


# A simple circuit to test differentiability
def circuit2(params):
    phi0, theta0, omega0 = [a for a in params[0]]
    phi1, theta1, omega1 = [a for a in params[1]]
    qml.Rot(phi0, theta0, omega0, wires=0)
    qml.Rot(phi1, theta1, omega1, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


params = [[pi, pi / 3, 5 * pi / 7], [3 * pi / 8, pi / 3, 2 * pi / 7]]


class TestDecomposeRotIntoZXZDifferentiability:
    """Tests to ensure that the transform is differentiable."""

    def test_rot_to_zxz(self):
        """Test that the transform is differentiable in the autograd
        interface."""

        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit2, dev)
        circuit_trans = rot_to_zxz(circuit2)
        qnode_trans = qml.QNode(circuit_trans, dev)
        grad_in = qml.grad(qnode, argnum=[0])(params)
        grad_trans = qml.grad(qnode_trans, argnum=[0])(params)

        assert qml.math.allclose(grad_in, grad_trans)

    def test_rot_to_zxz_tf(self):
        """Test that the transform is differentiable in the tensorflow
        interface."""

        params_tf = tf.Variable(params)

        dev = qml.device("default.qubit", wires=2)
        qnode_in = qml.QNode(circuit2, dev, interface="tf")
        qnode_trans = qml.QNode(rot_to_zxz(circuit2), dev, interface="tf")

        with tf.GradientTape() as tape:
            loss = qnode_in(params_tf)
            grad_in = tape.gradient(loss, params_tf)

        with tf.GradientTape() as tape:
            loss = qnode_trans(params_tf)
            grad_trans = tape.gradient(loss, params_tf)

        qml.math.allclose(grad_in, grad_trans)


#     def test_rot_to_zxz_torch(self):
#         """Test that the transform is differentiable in the torch
#         interface."""

#         params_torch = torch.tensor(params, dtype=torch.float64, requires_grad=True)
#         params_torch_in = params_torch
#         params_torch_trans = params_torch

#         dev = qml.device("default.qubit", wires=2)
#         qnode_in = qml.QNode(circuit2, dev, interface="torch")
#         qnode_trans = qml.QNode(rot_to_zxz(circuit2), dev, interface="torch")

#         result_in = qnode_in(params_torch_in)
#         result_trans = qnode_trans(params_torch_trans)

#         result_in.backward()
#         result_trans.backward()

#         assert qml.math.allclose(params_torch_in.grad, params_torch_trans.grad)
