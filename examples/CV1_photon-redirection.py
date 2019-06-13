"""Photon redirection example.

In this demo we optimize a beamsplitter
to redirect a photon from the first to the second mode.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

try:
    dev = qml.device('strawberryfields.fock', wires=2, cutoff_dim=10)
except:
    print("To run this example you need to install the strawberryfields plugin...")


@qml.qnode(dev)
def circuit(var):
    """Variational circuit.

    Args:
        var (array[float]): array containing the variables

    Returns:
        mean photon number of mode 0
    """
    qml.FockState(1, wires=0)
    qml.Beamsplitter(var[0], var[1], wires=[0, 1])

    return qml.expval.MeanPhoton(0)


def objective(var):
    """Objective to minimize.

    Args:
        var (array[float]): array containing the variables

    Returns:
        output of the variational circuit
    """
    return circuit(var)


opt = GradientDescentOptimizer(stepsize=0.1)

var_init = np.array([0.01, 0.01])

var = var_init
var_gd = []

for iteration in range(100):
    var = opt.step(objective, var)
    var_gd.append(var)
    
    if iteration % 10 == 0:
        print('Cost after step {:3d}: {:0.7f} | Variables [{:0.7f}, {:0.7f}]'
              ''.format(iteration, objective(var), var[0], var[1]))
