# Qubit Optimization for 3 qubit Ising model using the Pennylane `default.qubit` Plugin.
# Ising models are used in Quantum Annealing to solve Quadratic Unconstrained Binary
# Optimization (QUBO) problems with non-convex cost functions. This example demonstrates how
# gradient descent optimizers can get stuck in local minima when using non-convex cost functions.

# This example uses and compares the PennyLane gradient descent optimizer with the optimizers
# of PyTorch and TensorFlow for this quantum system: see also `Q5_isingmodel.ipynb`

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

# making a Pennylane device using the `default.qubit` Plugin

dev1 = qml.device("default.qubit", wires=3)

# we will be solving the problem with fixed coupling between the three
# qubits and optimizing the phase operator to get the minimum energy
# configuration of the spin system


@qml.qnode(dev1)
def circuit1(p1, p2):
    # Assuming first spin is always up (+1 eigenstate of the Pauli-Z operator),
    # we can optimize the rotation angles for the other two spins
    # so that the energy is minimized for the given couplings

    # We use the general Rot(phi,theta,omega,wires) single-qubit operation

    qml.Rot(p1[0], p1[1], p1[2], wires=1)
    qml.Rot(p2[0], p2[1], p2[2], wires=2)
    return qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliZ(2)


def cost(var):
    # let coupling matrix be J=[1, -1]
    J = np.array([1, -1])
    # circuit1 function returns a numpy array of Pauli-Z exp values
    spins = circuit1(var[0], var[1])
    # the expectation value of Pauli-Z is plus 1 for spin up and -1 for spin down
    energy = -sum(J_ij * spins[i] * spins[i + 1] for i, J_ij in enumerate(J))
    return energy


# Test it for [1,-1,-1] spin configuration. Total energy for this Ising model should be

# H = -1(J1*s1*s2 + J2*s2*s3) = -1 ( 1*1*(-1)  +  (-1)*(-1)*(-1) ) = 2

test1 = np.array([0, np.pi, 0])
test2 = np.array([0, np.pi, 0])
cost_check = cost([test1, test2])
print("Energy for [1,-1,-1] spin configuration:",cost_check)

# Random initialization
t1 = np.pi * (np.random.ranf(3))
t2 = np.pi * (np.random.ranf(3))
var_init = np.array([t1, t2])
cost_init = cost(var_init)
print("Randomly initialized angles:",var_init)
print("Corresponding cost before initialization:",cost_init)

# Now we optimize using the PennyLane gradient-descent optimizer

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


# the minimum energy is -2  for the spin configuration [1,1,-1] which corresponds to
# (phi,theta,omega) = (0,0,0) for spin2 and (0,pi,0) for spin3
# We might not always see this value due to the non-convex cost function

cost_final = cost(var)
print("Optimized angles:",var)
print("Final cost after optimization:",cost_final)
