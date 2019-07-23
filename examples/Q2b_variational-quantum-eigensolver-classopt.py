"""Variational quantum eigensolver example.

In this demo we optimize a variational circuit to lower
the squared energy expectation of a user-defined Hamiltonian.

We express the Hamiltonian as a sum of two Pauli operators.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

dev = qml.device('default.qubit', wires=2)


def ansatz():
    """
    Ansatz of the variational circuit.
    """

    qml.Rot(0.3, 1.8, 5.4, wires=1)
    qml.RX(-0.5, wires=0)
    qml.RY( 0.5, wires=1)
    qml.CNOT(wires=[0, 1])


@qml.qnode(dev)
def circuit_X():
    """Variational circuit with final X measurement.

    Returns:
        expectation of Pauli-X observable on Qubit 1
    """
    ansatz()
    return qml.expval(qml.PauliX(1))


@qml.qnode(dev)
def circuit_Y():
    """Variational circuit with final Y measurement.

    Returns:
        expectation of Pauli-Y observable on Qubit 1
    """
    ansatz()
    return qml.expval(qml.PauliY(1))


def cost(var):
    """Cost  function to be minimized.

    Args:
        var (list[float]): list of variables

    Returns:
        float: square of linear combination of the expectations
    """

    expX = circuit_X()
    expY = circuit_Y()

    return (var[0] * expX + var[1] * expY) ** 2


# optimizer
opt = GradientDescentOptimizer(0.5)

# minimize the cost
var = [0.3, 2.5]
var_gd = [var]
for it in range(20):
    var = opt.step(cost, var)
    var_gd.append(var)

    print('Cost after step {:5d}: {: .7f} | Variables: [{: .5f},{: .5f}]'
          .format(it+1, cost(var), var[0], var[1]) )
