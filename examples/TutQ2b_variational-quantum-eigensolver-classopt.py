"""Variational quantum eigensolver example.

In this demo we optimize a variational circuit to lower
the energy expectation of a user-defined Hamiltonian.

We express the Hamiltonian as a sum of two Pauli operators.
"""

import pennylane as qml
from pennylane.optimize import GradientDescentOptimizer

dev = qml.device('default.qubit', wires=2)


def ansatz():
    """
    Ansatz of the variational circuit.
    """

    qml.Rot(0.3, 1.8, 5.4, wires=[1])
    qml.RX(0.5, wires=[0])
    qml.RY(0.9, wires=[1])
    qml.CNOT(wires=[0, 1])


@qml.qnode(dev)
def circuit_X():
    """Variational circuit.

    Returns:
        expectation of Pauli-X observable on Qubit 1
    """
    ansatz()
    return qml.expval.PauliX(1)


@qml.qnode(dev)
def circuit_Y():
    """Variational circuit.

    Returns:
        expectation of Pauli-Y observable on Qubit 1
    """
    ansatz()
    return qml.expval.PauliY(1)


def cost(vars):
    """Cost  function to be minimized.

    Args:
        var (list[float]): list of variables

    Returns:
        float: square of linear combination of the expectations
    """

    expX = circuit_X()
    expY = circuit_Y()

    return (vars[0]*expX + vars[1]*expY)**2


# optimizer
o = GradientDescentOptimizer(0.5)

# minimize the cost
vars = [0.3, 2.5]
for it in range(20):
    vars = o.step(cost, vars)

    print('Cost after step {:5d}: {: .7f} | Variables: [{: .5f},{: .5f}]'
          .format(it+1, cost(vars), vars[0], vars[1]))


