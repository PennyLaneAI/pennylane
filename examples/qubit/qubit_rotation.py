"""Simple qubit optimization example.

In this demo, we perform rotation on one qubit, entangle it
to a second qubit, and then measure the state of the second qubit.
We then optimize the rotation so that the second qubit is measured
in state |1> with certainty.
"""

import openqml as qm
from openqml import numpy as np
from openqml._optimize import GradientDescentOptimizer, AdagradOptimizer

dev1 = qm.device('default.qubit', wires=1)

@qm.qfunc(dev1)
def circuit(weights):
    """QNode"""
    qm.RX(weights[0], [0])
    qm.RY(weights[1], [0])
    return qm.expectation.PauliZ(0)


def objective(weights):
    """Cost (error) function to be minimized."""
    return circuit(weights)


weights0 = np.array([0.001, 0.001])
print('Initial rotation angles:', weights0)

o = GradientDescentOptimizer(0.5)

weights = weights0
for step in np.arange(1, 101):
    weights = o.step(objective, weights)
    if step%5==0:
        print('Cost after step {}: {}'.format(step, objective(weights)))

print()
print('Optimized rotation angles:', weights)




