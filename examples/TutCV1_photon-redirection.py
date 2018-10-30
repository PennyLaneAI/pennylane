"""Photon redirection example.

In this demo we optimize a beam splitter
to redirect a photon from the first to the second mode.
"""

import openqml as qm
import numpy as np
from openqml.optimize import GradientDescentOptimizer

dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=10)


@qm.qnode(dev)
def circuit(var):
    """Variational circuit.

    Args:
        var (array[float]): array containing the variables

    Returns:
        mean photon number of Mode 0
    """
    qm.FockState(1, [0])
    qm.Beamsplitter(var[0], var[1], [0, 1])

    return qm.expval.MeanPhoton(0)


def objective(var):
    """Objective to minimize.

    Args:
        var (array[float]): array containing the variables

    Returns:
        output of the variational circuit
    """

    return circuit(var)


gd = GradientDescentOptimizer(stepsize=0.1)

var = np.array([0.01, 0.01])

for iteration in range(100):
    var = gd.step(objective, var)

    if iteration % 10 == 0:
        print('Cost after step {:3d}: {:0.7f} | Variables [{:0.7f}, {:0.7f}]'
              ''.format(iteration, objective(var), var[0], var[1]))




