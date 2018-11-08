"""Simple qubit optimization example using the ProjectQ simulator backend.

This "hello world" example for PennyLane optimizes a two rotations to
flip a qubit from state |0> to |1>.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

dev = qml.device('default.qubit', wires=1)


@qml.qnode(dev)
def circuit(var):
    """Variational circuit.

    Args:
        var (array[float]): array of variables

    Returns:
        expectation of Pauli-Z operator
    """
    qml.RX(var[0], wires=[0])
    qml.RY(var[1], wires=[0])
    return qml.expval.PauliZ(0)


def objective(var):
    """Cost function to be minimized.

    Args:
        var (array[float]): array of variables

    Returns:
        float: result of variational circuit
    """
    return circuit(var)


var_init = np.array([0.011, 0.012])

o = GradientDescentOptimizer(0.5)

var = var_init
for it in range(100):
    var = o.step(objective, var)

    if (it+1) % 5 == 0:
        print('Cost after step {:5d}: {: 0.7f}'.format(it+1, objective(var)))

print('\nOptimized rotation angles: {}'.format(var))
