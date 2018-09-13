"""Qubit optimization example.

In this demo, we perform rotation on one qubit, entangle it via a CNOT
gate to a second qubit, and then measure the second qubit projected onto PauliZ.
We then optimize the circuit such the resulting expectation value is 1.
"""
import openqml as qm
from openqml import numpy as np

dev1 = qm.device('default.qubit', wires=2)

@qm.qfunc(dev1)
def circuit(x, y, z):
    """QNode"""
    qm.Rot(x, y, z, [0])
    qm.CNOT([0, 1])
    return qm.expectation.PauliZ(1)


def cost(weights, batched):
    """Cost (error) function to be minimized."""
    return np.abs(circuit(*weights)-1/np.sqrt(2))


# initialize x with random value
weights0 = np.random.randn(3)
o = qm.Optimizer(cost, weights0, optimizer='SGD')

# train the circuit
c = o.train(max_steps=100)

# print the results
print('Initial rotation angles:', weights0)
print('Optimized rotation angles:', o.weights)
print('Circuit output at rotation angles:', circuit(*o.weights))
print('Circuit gradient at rotation angles:', qm.grad(circuit, o.weights))
