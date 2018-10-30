"""Quadrature displacement example.

This "hello world" example for PennyLane optimizes a displacement
gate to shift the x-quadrature of a Gaussian state to a value of 0.5.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

dev = qml.device('default.gaussian', wires=1)

@qml.qnode(dev)
def circuit(variables):

    qml.Displacement(variables[0], variables[1], [0])

    return qml.expval.X(0)


def objective(variables):
    return np.abs(circuit(variables) - 0.5)**2


o = GradientDescentOptimizer(0.1)

variables = np.array([5.0, 0.0])
print('Initial displacement parameters: {}\n'.format(variables))

for iteration in range(10):
    variables = o.step(objective, variables)

    print('Cost after step {:3d}: {:0.7f}'.format(iteration, objective(variables)))

print('\nOptimized displacement parameters: {}'.format(variables))


