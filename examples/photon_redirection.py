"""Photon redirection example.

This "hello world" example for PennyLane optimizes a beam splitter
to redirect a photon from the first to the second mode.
"""

import openqml as qm
from openqml import numpy as np
from openqml.optimize import GradientDescentOptimizer

dev = qm.device('default.gaussian', wires=1)


@qm.qfunc(dev)
def circuit(variables):

    qm.Displacement(variables[0], variables[1], [0])

    return qm.expectation.X(0)


def objective(variables):
    return np.abs(circuit(variables) - 0.5)**2


o = GradientDescentOptimizer(0.1)

variables = np.array([5.0, 0.0])
print('Initial displacement parameters: {}\n'.format(variables))

for iteration in range(10):
    variables = o.step(objective, variables)

    print('Cost after step {:3d}: {:0.7f}'.format(iteration, objective(variables)))

print('\nOptimized displacement parameters: {}'.format(variables))


