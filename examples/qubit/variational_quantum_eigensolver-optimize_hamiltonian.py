"""Variational quantum eigensolver example.

In this demo we optimize a variational circuit to lower
the energy expectation of a user-defined Hamiltonian.

We express the Hamiltonian as a sum of two Pauli operators.
"""

import openqml as qm
from openqml import numpy as np
from openqml._optimize import GradientDescentOptimizer

dev = qm.device('default.qubit', wires=2)


def ansatz(weights):
    """ Ansatz of the variational circuit."""

    # Hard-code the initial state
    initial_state = np.array([1, 1, 0, 1])/np.sqrt(3)
    qm.QubitStateVector(initial_state, wires=[0, 1])

    # Execute some parametrized quantum gates
    qm.RX(weights[0], [0])
    qm.RY(weights[1], [1])
    qm.CNOT([0, 1])


@qm.qfunc(dev)
def circuit_X(weights):
    """Circuit measuring the X operator for the second qubit"""
    ansatz(weights)
    return qm.expectation.PauliX(1)


@qm.qfunc(dev)
def circuit_Y(weights):
    """Circuit measuring the Y operator for the second qubit"""
    ansatz(weights)
    return qm.expectation.PauliY(1)


def cost(weights):
    """Cost (error) function to be minimized."""

    expX = circuit_X(weights)
    expY = circuit_Y(weights)

    return 0.1*expX + 0.5*expY


# initialize weights
weights0 = np.array([0., 0.])
print('Initial weights:', weights0)

# optimize the cost
o = GradientDescentOptimizer(0.5)
weights = weights0
for iteration in np.arange(1, 21):
    weights = o.step(cost, weights)
    print('Cost after step {:5d}: {: .7}'.format(iteration, cost(weights)))
print('Optimized weights:', weights)
