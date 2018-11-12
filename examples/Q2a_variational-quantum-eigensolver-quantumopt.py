"""Variational quantum eigensolver example.

In this example we optimize a variational circuit to lower
the squared energy expectation of a user-defined Hamiltonian.

We express the Hamiltonian as a sum of two Pauli operators.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

dev = qml.device('default.qubit', wires=2)

def ansatz(var):
    """ Ansatz of the variational circuit.

    Args:
        var (list[float]): list of variables
    """

    qml.Rot(0.3, 1.8, 5.4, wires=1)
    qml.RX(var[0], wires=0)
    qml.RY(var[1], wires=1)
    qml.CNOT(wires=[0, 1])


@qml.qnode(dev)
def circuit_X(var):
    """Variational circuit with final X measurement.

    Args:
        var (list[float]): list of variables

    Returns:
        expectation of Pauli-X observable on Qubit 1
    """
    ansatz(var)
    return qml.expval.PauliX(1)


@qml.qnode(dev)
def circuit_Y(var):
    """Variational circuit with final Y measurement.

    Args:
        var (list[float]): list of variables

    Returns:
        expectation of Pauli-Y observable on Qubit 1
    """
    ansatz(var)
    return qml.expval.PauliY(1)


def cost(var):
    """Cost  function to be minimized.

    Args:
        var (list[float]): list of variables

    Returns:
        float: square of linear combination of the expectations
    """

    expX = circuit_X(var)
    expY = circuit_Y(var)

    return (0.1 * expX + 0.5 * expY) ** 2


# optimizer
opt = GradientDescentOptimizer(0.5)

# minimize cost
var = [0.3, 2.5]
var_gd = [var]
for it in range(20):
    var = opt.step(cost, var)
    var_gd.append(var)

    print('Cost after step {:5d}: {: .7f} | Variables: [{: .5f},{: .5f}]'
          .format(it+1, cost(var), var[0], var[1]) )
