"""Simple qubit optimization example using the ProjectQ simulator backend.

This "Hello world!" example for PennyLane optimizes a two rotations to
flip a qubit from state |0> to |1>.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer, AdagradOptimizer

dev = qml.device('default.qubit', wires=1)

@qml.qnode(dev)
def circuit(var):
    """Variational circuit.

    Args:
        var (array[float]): array of variables

    Returns:
        expectation of Pauli-Z operator
    """
    qml.RX(var[0], wires=0)
    qml.RY(var[1], wires=0)
    return qml.expval(qml.PauliZ(0))


def objective(var):
    """Cost function to be minimized.

    Args:
        var (array[float]): array of variables

    Returns:
        float: output of variational circuit
    """
    return circuit(var)


var_init = np.array([-0.011, -0.012])

gd = GradientDescentOptimizer(0.4)

var = var_init
var_gd = [var]

for it in range(100):
    var = gd.step(objective, var)

    if (it + 1) % 5 == 0:
        var_gd.append(var)
        print('Objective after step {:5d}: {: .7f} | Angles: {}'.format(it + 1, objective(var), var) )
        
ada = AdagradOptimizer(0.4)

var = var_init
var_ada = [var]

for it in range(100):
    var = ada.step(objective, var)
    
    if (it + 1) % 5 == 0:
        var_ada.append(var)
        print('Objective after step {:5d}: {: .7f} | Angles: {}'.format(it + 1, objective(var), var) )
