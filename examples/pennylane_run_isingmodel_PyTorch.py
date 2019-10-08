r"""
.. _isingmodel_PyTorch:

Optimization of 3-qubit Ising model in PyTorch
===============================================

The Ising model is an example of a system whose optimization landscape is non-convex. Hence, using gradient descent may not be the best strategy as the optimization can get
stuck in local minima. Consequently, Ising models are popularly used to represent and solve
Quadratic Unconstrained Binary Optimization (QUBO) problems with non-convex cost functions in
Quantum Annealing (for example on a D-wave system). 

The energy for this system is given by:

.. math::  H=-\sum_{<i,j>} J_{ij} \sigma_i \sigma_{j}

where each spin can be in +1 or -1 spin state and :math:`J_{ij}` are the nearest neighbor coupling strengths.

PennyLane implementation
------------------------

This basic tutorial optimizes a 3-qubit Ising model using the PennyLane ``default.qubit``
device with the PyTorch machine learning interface. For simplicity, the first spin can be assumed
to be in the "up" state (+1 eigenstate of Pauli-Z operator) and the coupling matrix can be set to :math:`J = [1,-1]`. The rotation angles for the other two spins are then optimized
so that the energy of the system is minimized for the given couplings.
"""

import torch
from torch.autograd import Variable
import pennylane as qml
from pennylane import numpy as np

###############################################################################
# A three-qubit quantum circuit is initialized to represent the three spins:
 
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev, interface = "torch") 
def circuit(p1, p2):
    # We use the general Rot(phi,theta,omega,wires) single-qubit operation
    qml.Rot(p1[0], p1[1], p1[2], wires=1)
    qml.Rot(p2[0], p2[1], p2[2], wires=2)
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

###############################################################################
# The cost function to be minimized is defined as the energy of the spin configuration

def cost(var1, var2):
    # the circuit function returns a numpy array of Pauli-Z expectation values
    spins = circuit(var1, var2)

    # the expectation value of Pauli-Z is +1 for spin up and -1 for spin down
    energy = -(1 * spins[0] * spins[1]) - (-1 * spins[1] * spins[2])
    return energy

###############################################################################
# Sanity check
# ^^^^^^^^^^^^^
# Let's test the functions above using :math:`[s_1, s_2, s_3] = [1, -1, -1]` spin
# configuration and the given coupling matrix. The total energy for this Ising model
# should be:
#
# .. math:: H = -1(J_1 s_1 \otimes s_2 + J_2 s_2 \otimes s3) = 2 
#

test1 = torch.tensor([0, np.pi, 0])
test2 = torch.tensor([0, np.pi, 0])

cost_check = cost(test1, test2)
print("Energy for [1, -1, -1] spin configuration:", cost_check)

###############################################################################
# Random initialization
# ^^^^^^^^^^^^^^^^^^^^^

torch.manual_seed(56)
p1 = Variable((np.pi * torch.rand(3, dtype=torch.float64)), requires_grad=True)
p2 = Variable((np.pi * torch.rand(3, dtype=torch.float64)), requires_grad=True)

var_init = [p1, p2]
cost_init = cost(p1, p2)

print("Randomly initialized angles:",var_init)
print("Corresponding cost before optimization:",cost_init)

###############################################################################
# Optimization
# ^^^^^^^^^^^^
# Now we use the PyTorch gradient descent optimizer to minimize the cost:

opt = torch.optim.SGD(var_init, lr = 0.1)

steps = 100

def closure():
    opt.zero_grad()
    loss = cost(p1, p2)
    loss.backward()
    return loss

var_pt = [var_init]
cost_pt = [cost_init]

for i in range(steps):
    opt.step(closure)
    if (i + 1) % 5 == 0:
        p1n, p2n = opt.param_groups[0]["params"]
        costn = cost(p1n, p2n)
        var_pt.append([p1n, p2n])
        cost_pt.append(costn)
        print("Energy after step {:5d}: {: .7f} | Angles: {}".format(i + 1, costn, [p1n, p2n]),"\n")

###############################################################################
#
# .. note::
#     When using the *PyTorch* optimizer, keep in mind that:
#
#     1. ``loss.backward()`` computes the gradient of the cost function with respect to all parameters (whcih have ``requires_grad=True``). 
#     2. ``opt.step()`` performs the parameter update based on this *current* gradient and the learning rate. 
#     3. ``opt.zer_grad()`` sets all the gradients back to zero. It’s important to call this before ``loss.backward()`` to avoid the accumulation of  gradients from multiple passes.
#
#     Hence, its standard practice to define the ``closure()`` function that clears up the old gradient, 
#     evaluates the new gradient and passes it onto the optimizer in each step. 
#
# The minimum energy is -2  for the spin configuration :math:`[s_1, s_2, s_3] = [1, 1, -1]`
# which corresponds to
# :math:`(\phi, \theta, \omega) = (0, 0, 0)` for the second spin and :math:`(\phi, \theta, \omega) = (0, \pi, 0)` for 
# the third spin. We might not always see this cost value due to the non-convex cost function.

p1_final, p2_final = opt.param_groups[0]["params"]
print("Optimized angles:",p1_final, p2_final)
print("Final cost after optimization:",cost(p1_final, p2_final))

###############################################################################
#
# Local minimum
# ^^^^^^^^^^^^^
# If the spins are initialized close to the local minimum of zero energy, the optimizer is
# likely to get stuck here and never find the global minimum at -2. The figure below shows
# the results from two different initializations on various optimizers.
# 
# |
# .. image:: ../../examples/figures/ising1.png
#    :width: 48%
# .. image:: ../../examples/figures/ising2.png
#    :width: 48%
# |
#
# Try it yourself! Download and run this file with different
# initialization parameters and see how the results change.

