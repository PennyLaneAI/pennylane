# Qubit Optimization for 3 qubit Ising model using the Pennylane `default.qubit` Plugin.
# Ising models are used in Quantum Annealing to solve Quadratic Unconstrained Binary
# Optimization (QUBO) problems with non-convex cost functions. This example demonstrates how
# gradient descent optimizers can get stuck in local minima when using non-convex cost functions.

# This example uses and compares the PennyLane gradient descent optimizer with the optimizers
# of PyTorch and TensorFlow for this quantum system: see `Q5_isingmodel.ipynb`

import torch
from torch.autograd import Variable
import pennylane as qml
from pennylane import numpy as np

dev3 = qml.device("default.qubit", wires=3)


@qml.qnode(dev3, interface="torch")
def circuit3(p1, p2):
    # Assuming first spin is always up (+1 eigenstate of the Pauli-Z operator),
    # we can optimize the rotation angles for the other two spins
    # so that the energy is minimized for the given couplings

    # We use the general Rot(phi,theta,omega,wires) single-qubit operation
    qml.Rot(p1[0], p1[1], p1[2], wires=1)
    qml.Rot(p2[0], p2[1], p2[2], wires=2)
    return qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliZ(2)


def cost(var1, var2):
    # let coupling matrix be J=[1, -1]
    # circuit3 function returns a numpy array of Pauli-Z exp values
    spins = circuit3(var1, var2)
    # the expectation value of Pauli-Z is plus 1 for spin up and -1 for spin down
    energy = -(1 * spins[0] * spins[1]) - (-1 * spins[1] * spins[2])
    return energy


# Test it for [1,-1,-1] spin configuration. Total energy for this Ising model should be

# H = -1(J1*s1*s2 + J2*s2*s3) = -1 ( 1*1*(-1)  +  (-1)*(-1)*(-1) ) = 2

test1 = torch.tensor([0, np.pi, 0])
test2 = torch.tensor([0, np.pi, 0])

cost_check = cost(test1, test2)
print("Energy for [1,-1,-1] spin configuration:",cost_check)

# Random initialization in PyTorch

p1 = Variable((np.pi * torch.rand(3, dtype=torch.float64)), requires_grad=True)
p2 = Variable((np.pi * torch.rand(3, dtype=torch.float64)), requires_grad=True)
var_init = [p1, p2]
cost_init = cost(p1, p2)
print("Randomly initialized angles:",var_init)
print("Corresponding cost before initialization:",cost_init)

#  Now we use the PyTorch Adam optimizer to minimize this cost

opt = torch.optim.Adam(var_init, lr=0.1)

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


p1_final, p2_final = opt.param_groups[0]["params"]

# the minimum energy is -2  for the spin configuration [1,1,-1] which corresponds to
# (phi,theta,omega = (0,0,0) for spin2 and (0,pi,0) for spin3
# We might not always see this value due to the non-convex cost function

print("Optimized angles:",p1_final, p2_final)

print("Final cost after optimization:",cost(p1_final, p2_final))
