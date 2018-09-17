"""Simple qubit optimization example.

In this demo, we optimize the rotation of one qubit
so that the Z expectation is minimized.
"""

import openqml as qm
from openqml import numpy as np

dev1 = qm.device('default.qubit', wires=1)

@qm.qfunc(dev1)
def circuit(weights):
    """QNode"""
    qm.RX(weights[0], [0])
    qm.RY(weights[1], [0])
    return qm.expectation.PauliZ(0)


def cost(weights):
    """Cost (error) function to be minimized."""
    return circuit(weights)


# initialize weights
weights0 = np.array([0., 0.])
o = qm.Optimizer(cost, weights0, optimizer='Nelder-Mead')


print('Initial rotation angles:', weights0)
print('Initial qfunc output:', circuit(o.weights))

params = []
for step in range(10):
    # train the circuit
    c = o.train(max_steps=1)
    print('Cost:', cost(o.weights))
    params.append(o.weights)




print('Optimized rotation angles:', o.weights)
print('Final qfunc gradient:', qm.grad(circuit, o.weights))
