"""Variational quantum eigensolver example.

In this demo we use a variational circuit as an ansatz for
a VQE, and optimize the circuit to lower the energy expectation
of a user-defined Hamiltonian.
"""

import openqml as qm
from openqml import numpy as np

#dev1 = qm.device('strawberryfields.fock', wires=2, cutoff=15)
dev1 = qm.device('default.qubit', wires=2)


def ansatz(x,y,z):
    qm.Rot(x, y, z, [0])
    qm.CNOT([0, 1])


@qm.qfunc(dev1)
def circuit_Z(x):
    """QNode"""
    ansatz(x[0], x[1], x[2])
    qm.expectation.PauliZ(1)




def cost(weights, batch):  # Todo: remove batch
    """Cost (error) function to be minimized."""

    expZ = circuit_Z(weights)

    return - 0.3*expZ


# initialize x with random value
x0 = np.random.randn(3)
o = qm.Optimizer(cost, x0)

# train the circuit
o.train(max_steps=100)

# print the results
print('Initial rotation angles:', x0)
print('Optimized rotation angles:', o.weights)

# Does not learn!!!!!!!!!!????????