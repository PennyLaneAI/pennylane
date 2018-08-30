"""Qmode optimization example.

In this demo, we optimize a beam splitter
to redirect photons from the first to the second mode.
"""

import openqml as qm
from openqml import numpy as np

dev = qm.device('strawberryfields.gaussian', wires=2)


@qm.qfunc(dev)
def circuit(weight):
    """QNode"""

    qm.Displacement(0.5, [0])
    qm.Displacement(0.5, [1])

    qm.Beamsplitter(weight, 0, [0, 1])

    qm.expectation.MFock(0)


def cost(weight, batched):
    """Cost function maximises the value of x quadrature."""
    return -circuit(weight)


# initialize x with random value
weight0 = np.array([np.random.randn()])

o = qm.Optimizer(cost, weight0, optimizer='SGD')

# train the circuit
c = o.train(max_steps=100)

# print the results
print('Initial rotation angles:', weight0)
print('Optimized rotation angles:', o.weights)


