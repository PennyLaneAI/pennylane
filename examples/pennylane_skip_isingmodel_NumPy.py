r"""
.. _isingmodel_NumPy:

Optimization of 3-qubit Ising model in NumPy
=============================================

Qubit Optimization for 3-qubit Ising model using the Pennylane ``default.qubit`` Plugin.
The Ising model is an example of a system whose optimization landscape is
non-convex and hence using gradient descent may not be the best strategy
since the optimization can get stuck in local minima. 
Ising models are used in Quantum Annealing (for example on a D-wave system) to solve Quadratic 
Unconstrained Binary Optimization (QUBO) problems with non-convex cost functions. This example demonstrates how gradient descent optimizers can get stuck in local minima when using non-convex cost functions.

The energy for this system is given by:

.. math::  H=-\sum_{<i,j>} J_{ij} \sigma_i \sigma_{j}

where each spin can be in +1 or -1 spin state and :math:`J_{ij}` are
nearest neighbour coupling strengths.

For simplicity, we will assume that the first spin is always in up state
(+1 eigenstate of Pauli-Z operator). Moreover, we will be solving this problem with 
fixed couplings between the three qubits using the coupling matrix J = [1,-1].

We will then optimize the rotation angles for the other two spins
so that the energy of the system is minimized for the given couplings.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

###############################################################################

# making a Pennylane device using the `default.qubit` Plugin
dev1 = qml.device("default.qubit", wires = 3)

###############################################################################

@qml.qnode(dev1)
def circuit1(p1, p2):
    # We use the general Rot(phi,theta,omega,wires) single-qubit operation
    qml.Rot(p1[0], p1[1], p1[2], wires=1)
    qml.Rot(p2[0], p2[1], p2[2], wires=2)
    return qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliZ(2)

###############################################################################
# Our cost function is the energy of the spin configuration which we will
# optimize using gradient descent.

def cost(var):
    # coupling matrix 
    J = np.array([1, -1])
    # circuit1 function returns a NumPy array of Pauli-Z expectation values
    spins = circuit1(var[0], var[1])
    # the expectation value of Pauli-Z is +1 for spin up and -1 for spin down
    energy = -sum(J_ij * spins[i] * spins[i + 1] for i, J_ij in enumerate(J))
    return energy

###############################################################################
#
# Let’s test these functions for the [1,-1,-1] spin configuration.
# Total energy for this Ising model should be:
#
# .. math:: H = -1(J_1 s_1 \otimes s_2 + J_2 s_2 \otimes s3) = -1 [1 \times 1 \times (-1) + (-1) \times (-1) \times (-1)] = 2 
#

test1 = np.array([0, np.pi, 0])
test2 = np.array([0, np.pi, 0])

cost_check = cost([test1, test2])
print("Energy for [1,-1,-1] spin configuration:",cost_check)

###############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Energy for [1,-1,-1] spin configuration: 2.0

# Random initialization
t1 = np.pi * (np.random.ranf(3))
t2 = np.pi * (np.random.ranf(3))

var_init = np.array([t1, t2])
cost_init = cost(var_init)

print("Randomly initialized angles:",var_init)
print("Corresponding cost before initialization:",cost_init)

###############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Randomly initialized angles: [[2.22725129 3.03645474 0.17098677]
#     [1.10956825 2.83080802 1.31423232]]
#     Corresponding cost before initialization: 1.941314647156032

# Now we use the PennyLane gradient descent optimizer to minimize the cost.

gd = GradientDescentOptimizer(0.4)

var = var_init
var_gd = [var]
cost_gd = [cost_init]

for it in range(100):
    var = gd.step(cost, var)
    if (it + 1) % 5 == 0:
        var_gd.append(var)
        cost_gd.append(cost(var))
        print("Energy after step {:5d}: {: .7f} | Angles: {}".format(it + 1, cost(var), var),"\n")

###############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Energy after step     5:  0.2150387 | Angles: [[2.22725129 1.73664722 0.17098677]
#     [1.10956825 1.87815474 1.31423232]] 
#
#     Energy after step    10: -1.9215770 | Angles: [[2.22725129 0.03964932 0.17098677]
#     [1.10956825 2.74683154 1.31423232]] 
#
#     Energy after step    15: -1.9995023 | Angles: [[2.22725129e+00 1.59934697e-05 1.70986767e-01]
#     [1.10956825e+00 3.11004051e+00 1.31423232e+00]] 
#
#     Energy after step    20: -1.9999970 | Angles: [[2.22725129e+00 5.12582645e-09 1.70986767e-01]
#     [1.10956825e+00 3.13913874e+00 1.31423232e+00]] 
#
#     Energy after step    25: -2.0000000 | Angles: [[2.22725129e+00 1.63984657e-12 1.70986767e-01]
#     [1.10956825e+00 3.14140184e+00 1.31423232e+00]] 
#
#     Energy after step    30: -2.0000000 | Angles: [[2.22725129e+00 4.69058777e-16 1.70986767e-01]
#     [1.10956825e+00 3.14157782e+00 1.31423232e+00]] 
#
#     Energy after step    35: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159150e+00  1.31423232e+00]] 
#
#     Energy after step    40: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159256e+00  1.31423232e+00]] 
#
#     Energy after step    45: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]] 
#
#     Energy after step    50: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]] 
#
#     Energy after step    55: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]] 
#
#     Energy after step    60: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]] 
#
#     Energy after step    65: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]] 
#
#     Energy after step    70: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]] 
#
#     Energy after step    75: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]] 
#
#     Energy after step    80: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]] 
#
#     Energy after step    85: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]] 
#
#     Energy after step    90: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]] 
#
#     Energy after step    95: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]] 
#
#     Energy after step   100: -2.0000000 | Angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]]
#
# The minimum energy is -2  for the spin configuration [1,1,-1] which corresponds to
# :math:`(\phi, \theta, \omega) = (0, 0, 0)` for the second spin and :math:`(\phi, \theta, \omega) = (0, \pi, 0)` for 
# the third spin, respectively. We might not always see this value due to the non-convex cost function.

cost_final = cost(var)
print("Optimized angles:",var)
print("Final cost after optimization:",cost_final)

###############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Optimized angles: [[ 2.22725129e+00 -5.82971598e-17  1.70986767e-01]
#     [ 1.10956825e+00  3.14159265e+00  1.31423232e+00]]
#     Final cost after optimization: -1.9999999999999991
#
# When we initialize close to zero, the optimizer is more likely to get
# stuck in a local minimum. Try it yourself! Download and run this file with different
# initialization parameters and see how the results change.
# The figure below shows outputs of two different runs from different optimizers.
#  
# .. figure:: ../../examples/figures/ising1.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0);
#
# .. figure:: ../../examples/figures/ising2.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0);
