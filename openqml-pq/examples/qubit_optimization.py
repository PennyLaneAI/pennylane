"""Qubit optimization example for the ProjectQ plugin.

In this demo, we perform rotation on one qubit, entangle it via a CNOT
gate to a second qubit, and then measure the second qubit projected onto PauliZ.
We then optimize the circuit such the resulting expectation value is 1.
"""
#todo
import openqml as qm
from openqml import numpy as np

dev1 = qm.device('projectq.simulator')

@qm.qfunc(dev1)
def circuit(x, y, z):
    """QNode"""
    qm.Rot(x, y, z, [0])
    qm.CNOT([0, 1])
    qm.expectation.PauliZ(1)


def cost(x, batched):
    """Cost (error) function to be minimized."""
    return np.abs(circuit(x)-1)


# initialize x with random value
x0 = np.random.randn(3)
o = qm.Optimizer(cost, x0, optimizer='SGD')

# train the circuit
c = o.train(max_steps=100)

# print the results
print('Initial rotation angles:', x0)
print('Optimized rotation angles:', o.weights)
print('Circuit output at rotation angles:', circuit(*o.weights))
print('Circuit gradient at rotation angles:', qm.grad(circuit, *o.weights)[0])
