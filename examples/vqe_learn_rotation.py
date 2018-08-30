"""Variational quantum eigensolver example.

In this demo we use a variational circuit as an ansatz for
a VQE, and optimize the circuit to lower the energy expectation
of a user-defined Hamiltonian.
"""

import openqml as qm
from openqml import numpy as np


def ansatz(weights):
    """ Ansatz of the variational circuit."""

    initial_state = np.array([1, 1, 0, 1])/np.sqrt(3)
    qm.QubitStateVector(initial_state, wires=[0, 1])

    qm.Rot(weights[0], weights[1], weights[2], [0])
    qm.CNOT([0, 1])


dev1 = qm.device('default.qubit', wires=2)


@qm.qfunc(dev1)
def circuit_X(weights):
    """Circuit measuring the X operator"""
    ansatz(weights)
    qm.expectation.PauliX(1)


@qm.qfunc(dev1)
def circuit_Y(weights):
    """Circuit measuring the Y operator"""
    ansatz(weights)
    qm.expectation.PauliY(1)


@qm.qfunc(dev1)
def circuit_Z(weights):
    """Circuit measuring the Z operator"""
    ansatz(weights)
    qm.expectation.PauliZ(1)


def cost(weights, batch):  # Todo: remove batch
    """Cost (error) function to be minimized."""

    expZ = circuit_Z(weights)
    expX = circuit_X(weights)
    expY = circuit_Y(weights)

    return 0.1*expX + 0.5*expY - 0.3*expZ


# initialize weights with random values
weights0 = np.random.randn(3)

# train the device
o = qm.Optimizer(cost, weights0)
o.train(max_steps=100)

print('Initial rotation angles:', weights0)
print('Optimized rotation angles:', o.weights)
