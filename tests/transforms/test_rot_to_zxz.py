"""
Tests for the rot_to_zxz transforms.
"""

import pytest

import pennylane as qml
from pennylane.numpy import pi

from pennylane.transforms import rot_to_zxz

# A sample circuit for testing
def circuit(params):
    phi0, theta0, omega0 = [a for a in params[0]]
    phi1, theta1, omega1 = [a for a in params[1]]
    qml.Rot(phi0, theta0, omega0,wires=0)
    qml.Rot(phi1, theta1, omega1,wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))
params = [[pi, pi/3, 5*pi/7], [3*pi/8, pi/3, 2*pi/7]]


class TestDecomposeRotIntoZXZ:
    """Tests to ensure that the transform is differentiable."""

    @pytest.mark.parametrize("circuit,params")
    def test_rot_to_zxz(self,circuit,params):
        """Test that the transform is differentiable in the autograd
        interface."""

        dev = qml.device('default.qubit', wires=2)
        qnode = qml.QNode(circuit, dev)
        circuit_trans = rot_to_zxz(circuit)
        qnode_trans = qml.QNode(circuit_trans, dev)
        grad_in = qml.grad(qnode, argnum=[0])(params)
        grad_trans = qml.grad(qnode_trans, argnum=[0])(params)

        assert qml.math.allclose(grad_in, grad_trans)
        
