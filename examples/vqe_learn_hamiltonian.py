"""Variational quantum eigensolver example.

In this demo we use a fixed quantum circuit
and optimize the (classical) Hamiltonian parameters
to lower the energy expectation. """

import openqml as qm
from openqml import numpy as np


def ansatz():
    """ Ansatz of the variational circuit."""
    initial_state = np.array([1, 1, 0, 1])/np.sqrt(3)
    qm.QubitStateVector(initial_state, wires=[0, 1])

    qm.Rot(0.4, 0.3, 1.3, [0])
    qm.CNOT([0, 1])


dev1 = qm.device('default.qubit', wires=2)


@qm.qfunc(dev1)
def circuit_X():
    """Circuit measuring the X operator"""
    ansatz()
    qm.expectation.PauliX(1)


@qm.qfunc(dev1)
def circuit_Y():
    """Circuit measuring the Y operator"""
    ansatz()
    qm.expectation.PauliY(1)


@qm.qfunc(dev1)
def circuit_Z():
    """Circuit measuring the Z operator"""
    ansatz()
    qm.expectation.PauliZ(1)


def cost(weights, batch):
    """Cost (error) function to be minimized."""

    expX = circuit_X()
    expY = circuit_Y()
    expZ = circuit_Z()

    return weights[0]*expX + weights[1]*expY - weights[2]*expZ


# initialize x with random value
weights0 = np.random.randn(3)

# train the device
o = qm.Optimizer(cost, weights0)
o.train(max_steps=100)

print('Initial hamiltonian coefficients:', weights0)
print('Trained hamiltonian coefficients:', o.weights)



