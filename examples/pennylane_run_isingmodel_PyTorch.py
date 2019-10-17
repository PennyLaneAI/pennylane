r"""
.. _isingmodel_PyTorch:

3-qubit Ising model in PyTorch
==============================

The interacting spins with variable coupling strengths of an `Ising model <https://en.wikipedia.org/wiki/Ising_model>`__ can be used to simulate various machine learning concepts like `Hopfield networks <https://en.wikipedia.org/wiki/Hopfield_network>`__ and `Boltzmann machines <https://en.wikipedia.org/wiki/Boltzmann_machine>`__ :cite:`schuld2018supervised`. They also closely imitate the underlying mathematics of a subclass of computational problems called
`Quadratic Unconstrained Binary Optimization (QUBO) <https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization>`__ problems. 

Ising models are commonly encountered in the subject area of adiabatic quantum computing. Quantum annealing algorithms (for example, as performed on a D-wave system) are often used to find low-energy configurations of Ising problems.
The optimization landscape of the Ising model is non-convex, which can make finding global minima challenging. In this tutorial, we get a closer look at this phenomenon by applying gradient descent techniques to a toy Ising model.  

PennyLane implementation
------------------------

This basic tutorial optimizes a 3-qubit Ising model using the PennyLane ``default.qubit``
device with PyTorch. In the absence of external fields, the Hamiltonian for this system is given by:

.. math::  H=-\sum_{<i,j>} J_{ij} \sigma_i \sigma_{j},

where each spin can be in the +1 or -1 spin state and :math:`J_{ij}` are the nearest-neighbour coupling strengths.

For simplicity, the first spin can be assumed
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

@qml.qnode(dev, interface="torch") 
def circuit(p1, p2):
    # We use the general Rot(phi,theta,omega,wires) single-qubit operation
    qml.Rot(p1[0], p1[1], p1[2], wires=1)
    qml.Rot(p2[0], p2[1], p2[2], wires=2)
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

###############################################################################
# The cost function to be minimized is defined as the energy of the spin configuration:

def cost(var1, var2):
    # the circuit function returns a numpy array of Pauli-Z expectation values
    spins = circuit(var1, var2)

    # the expectation value of Pauli-Z is +1 for spin up and -1 for spin down
    energy = -(1 * spins[0] * spins[1]) - (-1 * spins[1] * spins[2])
    return energy

###############################################################################
# Sanity check
# ^^^^^^^^^^^^^
# Let's test the functions above using the :math:`[s_1, s_2, s_3] = [1, -1, -1]` spin
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

print("Randomly initialized angles:", var_init)
print("Corresponding cost before optimization:", cost_init)

###############################################################################
# Optimization
# ^^^^^^^^^^^^
# Now we use the PyTorch gradient descent optimizer to minimize the cost:

opt = torch.optim.SGD(var_init, lr=0.1)

def closure():
    opt.zero_grad()
    loss = cost(p1, p2)
    loss.backward()
    return loss

var_pt = [var_init]
cost_pt = [cost_init]
x = [0]

for i in range(100):
    opt.step(closure)
    if (i + 1) % 5 == 0:
        x.append(i)
        p1n, p2n = opt.param_groups[0]["params"]
        costn = cost(p1n, p2n)
        var_pt.append([p1n, p2n])
        cost_pt.append(costn)

        # for clarity, the angles are printed as numpy arrays
        print("Energy after step {:5d}: {: .7f} | Angles: {}".format(i+1, costn, [p1n.detach().numpy(), p2n.detach().numpy()]),"\n")
        

###############################################################################
#
# .. note::
#     When using the *PyTorch* optimizer, keep in mind that:
#
#     1. ``loss.backward()`` computes the gradient of the cost function with respect to all parameters with ``requires_grad=True``. 
#     2. ``opt.step()`` performs the parameter update based on this *current* gradient and the learning rate. 
#     3. ``opt.zero_grad()`` sets all the gradients back to zero. It's important to call this before ``loss.backward()`` to avoid the accumulation of gradients from multiple passes.
#
#     Hence, its standard practice to define the ``closure()`` function that clears up the old gradient, 
#     evaluates the new gradient and passes it onto the optimizer in each step. 
#
# The minimum energy is -2 for the spin configuration :math:`[s_1, s_2, s_3] = [1, 1, -1]`
# which corresponds to
# :math:`(\phi, \theta, \omega) = (0, 0, 0)` for the second spin and :math:`(\phi, \theta, \omega) = (0, \pi, 0)` for 
# the third spin. Note that gradient descent optimization might not find this global minimum due to the non-convex cost function, as is shown in the next section.

p1_final, p2_final = opt.param_groups[0]["params"]
print("Optimized angles:", p1_final, p2_final)
print("Final cost after optimization:", cost(p1_final, p2_final))

###############################################################################

import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 4))
plt.plot(x, cost_pt, label = 'global minimum')
plt.xlabel("Optimization steps")
plt.ylabel("Cost / Energy")
plt.legend()
plt.show()

###############################################################################
# Local minimum
# ^^^^^^^^^^^^^
# If the spins are initialized close to the local minimum of zero energy, the optimizer is
# likely to get stuck here and never find the global minimum at -2. 

torch.manual_seed(9)
p3 = Variable((np.pi*torch.rand(3, dtype = torch.float64)), requires_grad = True)
p4 = Variable((np.pi*torch.rand(3, dtype = torch.float64)), requires_grad = True)

var_init_loc = [p3, p4]
cost_init_loc = cost(p3, p4)

print("Corresponding cost before optimization:", cost_init_loc)


###############################################################################

opt = torch.optim.SGD(var_init_loc, lr = 0.1)

def closure():
    opt.zero_grad()
    loss = cost(p3, p4)
    loss.backward()
    return loss

var_pt_loc = [var_init_loc]
cost_pt_loc = [cost_init_loc]

for j in range(100):
    opt.step(closure)
    if (j + 1) % 5 == 0:
        p3n, p4n = opt.param_groups[0]['params']
        costn = cost(p3n, p4n)
        var_pt_loc.append([p3n, p4n])
        cost_pt_loc.append(costn)

        # for clarity, the angles are printed as numpy arrays
        print('Energy after step {:5d}: {: .7f} | Angles: {}'.format(j+1, costn, [p3n.detach().numpy(), p4n.detach().numpy()]),"\n")

###############################################################################

fig = plt.figure(figsize=(6, 4))
plt.plot(x, cost_pt_loc, 'r', label = 'local minimum')
plt.xlabel("Optimization steps")
plt.ylabel("Cost / Energy")
plt.legend()
plt.show()

###############################################################################
# |
# Try it yourself! Download and run this file with different
# initialization parameters and see how the results change.
#
# Further reading
# ^^^^^^^^^^^^^^^
#
# 1. Maria Schuld and Francesco Petruccione. "Supervised Learning with Quantum Computers."
# Springer, 2018.
#
# 2. Andrew Lucas. "Ising formulations of many NP problems."
# `arXiv:1302.5843 <https://arxiv.org/pdf/1302.5843>`__, 2014.
#
# 3. Gary Kochenberger et al. "The Unconstrained Binary Quadratic Programming Problem: A Survey."
# `Journal of Combinatorial Optimization <https://link.springer.com/article/10.1007/s10878-014-9734-0>`__, 2014.

