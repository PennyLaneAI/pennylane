"""Simple qubit optimization example.

This "hello world" example for PennyLane optimizes two rotation angles
to flip a qubit from state |0> to state |1>.
"""

import openqml as qm
from openqml import numpy as np
from openqml.optimize import GradientDescentOptimizer

dev = qm.device('default.qubit', wires=1)

@qm.qfunc(dev)
def circuit(variables):
    """QNode"""
    qm.RX(variables[0], [0])
    qm.RY(variables[1], [0])
    return qm.expectation.PauliZ(0)


def objective(variables):
    """Cost (error) function to be minimized."""
    return circuit(variables)


vars_init = np.array([0.001, 0.001])
print('Initial rotation angles:', vars_init)
print('Initial cost: {: 0.7f}'.format(objective(vars_init)))

# Gradient descent
print('\nGradient descent Optimizer')
o = GradientDescentOptimizer(0.5)
variables = vars_init
for it in np.arange(1, 101):
    variables = o.step(objective, variables)
    if it % 5 == 0:
        print('Cost after step {:5d}: {: 0.7f}'.format(it, objective(variables)))
print('Optimized rotation angles:', variables)
