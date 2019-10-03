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
Plugin with PyTorch machine learning interface. For simplicity, the first spin can be assumed
to be in the "up" state (+1 eigenstate of Pauli-Z operator) and the coupling matrix can be set to :math:`J = [1,-1]`. The rotation angles for the other two spins are then optimized
so that the energy of the system is minimized for the given couplings.
"""

import torch
from torch.autograd import Variable
import pennylane as qml
from pennylane import numpy as np

###############################################################################
# A three-qubit quantum circuit is initialized to represent the three spins:
 
dev = qml.device("default.qubit", wires = 3)

@qml.qnode(dev, interface = "torch") 
def circuit(p1, p2):
    # We use the general Rot(phi,theta,omega,wires) single-qubit operation
    qml.Rot(p1[0], p1[1], p1[2], wires=1)
    qml.Rot(p2[0], p2[1], p2[2], wires=2)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

###############################################################################
# The cost function can be defined as the energy of the spin configuration which is to be
# optimized using gradient descent:

def cost(var1, var2):
    # the circuit function returns a numpy array of Pauli-Z expectation values
    spins = circuit(var1, var2)

    # the expectation value of Pauli-Z is +1 for spin up and -1 for spin down
    energy = -(1 * spins[0] * spins[1]) - (-1 * spins[1] * spins[2])
    return energy

###############################################################################
# Sanity check
# ^^^^^^^^^^^^^
# Let’s test the functions above using :math:`[s_1, s_2, s_3] = [1, -1, -1]` spin
# configuration and the given coupling matrix. The total energy for this Ising model
# should be:
#
# .. math:: H = -1(J_1 s_1 \otimes s_2 + J_2 s_2 \otimes s3) = 2 
#

test1 = torch.tensor([0, np.pi, 0])
test2 = torch.tensor([0, np.pi, 0])

cost_check = cost(test1, test2)
print("Energy for [1, -1, -1] spin configuration:",cost_check)

###############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Energy for [1, -1, -1] spin configuration is: tensor(2.0000, dtype=torch.float64)
#
#
# Random initialization
# ^^^^^^^^^^^^^^^^^^^^^

p1 = Variable((np.pi * torch.rand(3, dtype=torch.float64)), requires_grad=True)
p2 = Variable((np.pi * torch.rand(3, dtype=torch.float64)), requires_grad=True)

var_init = [p1, p2]
cost_init = cost(p1, p2)

print("Randomly initialized angles:",var_init)
print("Corresponding cost before optimization:",cost_init)

###############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Randomly initialized angles: [tensor([0.3461, 2.9970, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 2.8112, 1.1473], dtype=torch.float64, requires_grad=True)]
#     Corresponding cost before initialization: tensor(1.9256, dtype=torch.float64, grad_fn=<SubBackward0>)
#
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
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Energy after step     5:  1.3439279 | Angles: [tensor([0.3461, 2.5032, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 2.3098, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    10:  0.5053288 | Angles: [tensor([0.3461, 1.9946, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 1.8016, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    15: -0.0728912 | Angles: [tensor([0.3461, 1.4770, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 1.3476, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    20: -0.3170200 | Angles: [tensor([0.3461, 0.9930, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 1.1379, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    25: -0.6022581 | Angles: [tensor([0.3461, 0.5703, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 1.2823, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    30: -1.0901277 | Angles: [tensor([0.3461, 0.2054, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 1.6846, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    35: -1.5947685 | Angles: [tensor([ 0.3461, -0.0799,  2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 2.2141, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    40: -1.8839163 | Angles: [tensor([ 0.3461, -0.2145,  2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 2.7601, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    45: -1.9681684 | Angles: [tensor([ 0.3461, -0.1725,  2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.2078, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    50: -1.9475494 | Angles: [tensor([ 0.3461, -0.0402,  2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.4619, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    55: -1.9300671 | Angles: [tensor([0.3461, 0.0668, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.5059, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    60: -1.9582732 | Angles: [tensor([0.3461, 0.0884, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.4033, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    65: -1.9930545 | Angles: [tensor([0.3461, 0.0406, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.2446, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    70: -1.9991189 | Angles: [tensor([ 0.3461, -0.0178,  2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.1080, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    75: -1.9930509 | Angles: [tensor([ 0.3461, -0.0400,  2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.0381, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    80: -1.9941513 | Angles: [tensor([ 0.3461, -0.0226,  2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.0382, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    85: -1.9982029 | Angles: [tensor([0.3461, 0.0059, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.0822, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    90: -1.9996421 | Angles: [tensor([0.3461, 0.0183, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.1349, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step    95: -1.9994925 | Angles: [tensor([0.3461, 0.0104, 2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.1698, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#     Energy after step   100: -1.9993288 | Angles: [tensor([ 0.3461, -0.0033,  2.2130], dtype=torch.float64, requires_grad=True), tensor([2.7079, 3.1779, 1.1473], dtype=torch.float64, requires_grad=True)] 
#
#
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
# the third spin. We might not always see this value due to the non-convex cost function.

p1_final, p2_final = opt.param_groups[0]["params"]
print("Optimized angles:",p1_final, p2_final)
print("Final cost after optimization:",cost(p1_final, p2_final))

###############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Optimized angles: tensor([ 0.3461, -0.0033,  2.2130], dtype=torch.float64, requires_grad=True) tensor([2.7079, 3.1779, 1.1473], dtype=torch.float64, requires_grad=True)
#     Final cost after optimization: tensor(-1.9993, dtype=torch.float64, grad_fn=<SubBackward0>)
#
# Local minimum
# ^^^^^^^^^^^^^
# If the spins are initialized close to the local minimum of zero energy, the optimizer is
# likely to get stuck here and never find the global minimum at -2. 
#
# Try it yourself! Download and run this file with different
# initialization parameters and see how the results change.
# The figure below shows the results from two different initializations on various optimizers.
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

