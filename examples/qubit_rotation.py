"""Qubit optimization example.

This "hello world" example for PennyLane optimizes two rotation angles
to flip a qubit from state |0> to state |1>.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

dev = qml.device('default.qubit', wires=1)


@qm.qnode(dev)
def circuit(variables):
    """QNode"""
    qml.RX(variables[0], [0])
    qml.RY(variables[1], [0])
    return qml.expval.PauliZ(0)


def objective(variables):
    """Cost (error) function to be minimized."""
    return circuit(variables)


o = GradientDescentOptimizer(0.5)

var = np.array([0.011, 0.012])
print('Initial rotation angles:'.format(var))
print('Initial cost: {: 0.7f}'.format(objective(var)))

for it in range(100):
    var = o.step(objective, var)
    if it % 5 == 0:
        print('Cost after step {:5d}: {: 0.7f}'.format(it+1, objective(var)))

print('Optimized rotation angles:'.format(var))
