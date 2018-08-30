"""Benchmarking quantum automatic differentiation.

In this demo we train a variational circuit using
automatic differentiation and numeric differentiation
and plot the results.
"""

import openqml as qm
from openqml import numpy as np

dev1 = qm.device('default.qubit', wires=4)


@qm.qfunc(dev1)
def circuit_Z(x, y, z):
    """QNode"""

    qm.expectation.PauliZ(0)


def cost(weights, batch):  # Todo: remove batch
    """Cost (error) function to be minimized."""

    expZ = circuit_Z(weights)

    return (1. - expZ)**2


# initialize x with random value
x0 = np.random.randn(3)
o = qm.Optimizer(cost, x0)

# train the circuit
o.train(max_steps=100)

# print the results
print('Initial rotation angles:', x0)
print('Optimized rotation angles:', o.weights)

# Does not learn!!!!!!!!!!????????