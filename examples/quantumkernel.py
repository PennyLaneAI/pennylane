"""Quantum kernel method example.

In this demo we use a variational circuit as a linear
classifier in a feature Hilbert space according to
this paper XXX.
"""

import openqml as qm
from openqml import numpy as np

dev1 = qm.device('default.qubit', wires=2)




@qm.qfunc(dev1)
def circuit_Z(x, y, z):
    """QNode"""
    ansatz(x, y, z)
    qm.expectation.PauliZ(1)


@qm.qfunc(dev1)
def circuit_Y(x, y, z):
    """QNode"""
    ansatz(x, y, z)
    qm.expectation.PauliY(1)


@qm.qfunc(dev1)
def circuit_X(x, y, z):  # Todo: argument of circuit
    """QNode"""
    ansatz(x, y, z)
    qm.expectation.PauliX(1)


def cost(weights, batch):  # Todo: remove batch
    """Cost (error) function to be minimized."""

    expZ = circuit_Z(weights)
    expX = circuit_X(weights)
    expY = circuit_Y(weights)

    return 0.1*expX + 0.5*expY - 0.3*expZ


# initialize x with random value
x0 = np.random.randn(3)
o = qm.Optimizer(cost, x0)

# train the circuit
o.train(max_steps=100)

# print the results
print('Initial rotation angles:', x0)
print('Optimized rotation angles:', o.weights)

