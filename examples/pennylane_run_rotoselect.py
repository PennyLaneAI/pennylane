r"""

.. _rotoselect:

Circuit structure learning with Rotoselect
==========================================

This example demonstrates how to learn the optimal selection of rotation
gates in addition to their parameters so as to minimize a cost
function. We apply the algorithm to VQE (as outlined in a prior
tutorial) and attempt to reproduce the circuit structure for that algorithm
along with the optimal parameters.
"""
##############################################################################
# Background
# ----------
#
# The effects of noise tend to increase with the depth of a quantum circuit, so
# there is incentive to keep circuits as shallow as possible. Generally, the
# chosen set of gates on a circuit are suboptimal for the task at hand. The Rotoselect
# algorithm provides a method for learning the structure of a quantum circuit, in addition
# to those parameters which optimize the cost function.
#
# VQE
# ~~~
#
# We choose to focus on the example of VQE with 2 qubits for simplicity. Here, the Hamiltonian
# is
#
# .. math::
#   H = 0.1*\sigma_x^2+0.5*\sigma_y^2
#
# which act on the second qubit in the circuit. We adopt the ansatz from the previous tutorial and
# and proceed to calculate the ground state using the Rotosolve algorithm. We then attempt to recover
# the ansatz structure (or discover something better) by switching to the Rotoselect algorithm.

import pennylane as qml
from pennylane import numpy as np


n_wires = 2

dev = qml.device('default.qubit',analytic=True,wires=2,shots=1000)

def ansatz(var):
    qml.RX(var[0], wires=0)
    qml.RY(var[1], wires=1)

@qml.qnode(dev)
def circuit_X(params):
    ansatz(params)
    return qml.expval(qml.PauliX(1))

@qml.qnode(dev)
def circuit_Y(params):
    ansatz(params)
    return qml.expval(qml.PauliY(1))


def cost(params):
    X = circuit_X(params)
    Y = circuit_Y(params)
    return 0.1*X + 0.5*Y




params = [0.3,0.25]

for i in range(30):
    print(cost(params))
    for d in range(len(params)):
        phi = 1.3
        params[d] = phi
        M_phi = cost(params)
        params[d] = phi + np.pi/2.
        M_phi_plus = cost(params)
        params[d] = phi - np.pi/2.
        M_phi_minus = cost(params)
        a = np.arctan2(2.*M_phi-M_phi_plus-M_phi_minus, M_phi_plus-M_phi_minus)
        params[d] = -np.pi/2. - a

print(params)
#opt = qml.GradientDescentOptimizer(0.5)

#params = init_params

#for i in range(20):
#    print(cost(params))
#    params = opt.step(cost,params)













