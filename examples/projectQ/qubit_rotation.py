"""Simple qubit optimization example.

In this demo, we perform rotation on one qubit, entangle it
to a second qubit, and then measure the state of the second qubit.
We then optimize the rotation so that the second qubit is measured
in state |1> with certainty.
"""

import openqml as qm
from openqml import numpy as np

dev1 = qm.device('default.qubit', wires=2)

@qm.qfunc(dev1)
def circuit(theta_x):
    """QNode"""
    qm.RX(theta_x, [0])
    qm.CNOT([0, 1])
    return qm.expectation.PauliZ(1)


def cost(weights):
    """Cost (error) function to be minimized."""
    return np.abs(circuit(*weights)-1)#1/np.sqrt(2))


# initialize x with random value
weights0 = np.array([0.])
o = qm.Optimizer(cost, weights0, optimizer='Nelder-Mead')


print('Initial rotation angles:', weights0)

for step in range(100):
    # train the circuit
    c = o.step()
    print('Cost:', cost(o.weights))
    print('Qfunc output:', circuit(*o.weights))

print('Optimized rotation angles:', o.weights)
print('Final qfunc gradient:', qm.grad(circuit, o.weights))
