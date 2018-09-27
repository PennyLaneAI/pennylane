"""Photon redirection example.

In this demo we optimize a beam splitter
to redirect a photon from the first to the second mode.
"""

import openqml as qm
from openqml import numpy as np
from openqml._optimize import GradientDescentOptimizer

dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=10)


@qm.qfunc(dev)
def circuit(weights):

    qm.FockState(1, [0])
    qm.Beamsplitter(weights[0], weights[1], [0, 1])

    return qm.expectation.Fock(1)


def objective(weights):
    return circuit(weights)


weights0 = np.array([0.1, 0.0])

o = GradientDescentOptimizer(0.1)
weights = o.step(objective, weights0)


weights = weights0
for iteration in range(101):
    weights = o.step(objective, weights)
    if iteration % 5 == 0:
        print('Cost after step {:3d}: {:0.7f}'
              ''.format(iteration, objective(weights)))




