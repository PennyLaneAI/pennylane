# Qubit Optimization for 3 qubit Ising model using the Pennylane `default.qubit` Plugin.
# Ising models are used in Quantum Annealing to solve Quadratic Unconstrained Binary
# Optimization (QUBO) problems with non-convex cost functions. This example demonstrates how
# gradient descent optimizers can get stuck in local minima when using non-convex cost functions.

# This example uses and compares the PennyLane gradient descent optimizer with the optimizers
# of PyTorch and TensorFlow for this quantum system: see also `Q5_isingmodel.ipynb`

import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Pennylane interfaces with the eager execution of TensorFlow
tf.enable_eager_execution()

# You can check if the eager execution is working by using the following command
# print(tf.executing_eagerly())

import pennylane as qml
from pennylane import numpy as np

dev2 = qml.device("default.qubit", wires=3)


@qml.qnode(dev2, interface="tfe")
def circuit2(p1, p2):
    # Assuming first spin is always up (+1 eigenstate of the Pauli-Z operator),
    # we can optimize the rotation angles for the other two spins
    # so that the energy is minimized for the given couplings

    # We use the general Rot(phi,theta,omega,wires) single-qubit operation
    qml.Rot(p1[0], p1[1], p1[2], wires=1)
    qml.Rot(p2[0], p2[1], p2[2], wires=2)
    return qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliZ(2)


def cost(var):
    # let coupling matrix be J=[1, -1]
    # circuit2 function returns a numpy array of Pauli-Z exp values
    spins = circuit2(var[0], var[1])
    # the expectation value of Pauli-Z is plus 1 for spin up and -1 for spin down
    energy = -(1 * spins[0] * spins[1]) - (-1 * spins[1] * spins[2])
    return energy


# Test it for [1,-1,-1] spin configuration. Total energy for this Ising model should be

# H = -1(J1*s1*s2 + J2*s2*s3) = -1 ( 1*1*(-1)  +  (-1)*(-1)*(-1) ) = 2

test1 = tfe.Variable([0, np.pi, 0], dtype=tf.float64)
test2 = tfe.Variable([0, np.pi, 0], dtype=tf.float64)
cost_check = cost([test1, test2])
print("Energy for [1,-1,-1] spin configuration:",cost_check)

# Random initialization in TensorFlow

t1 = tfe.Variable(tf.random_uniform([3], 0, np.pi, dtype=tf.float64))
t2 = tfe.Variable(tf.random_uniform([3], 0, np.pi, dtype=tf.float64))
var_init = [t1, t2]
cost_init = cost(var_init)
print("Randomly initialized angles:",var_init)
print("Corresponding cost before initialization:",cost_init)

# optimize using the TensorFlow optimizer

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)

steps = 100

var = var_init
var_tf = [var]
cost_tf = [cost_init]

for i in range(steps):
    with tf.GradientTape() as tape:
        loss = cost([t1, t2])
        grads = tape.gradient(loss, [t1, t2])

    opt.apply_gradients(zip(grads, [t1, t2]), global_step=tf.train.get_or_create_global_step())
    if (i + 1) % 5 == 0:
        var_tf.append([t1, t2])
        cost_tf.append(cost([t1, t2]))
        print("Energy after step {:5d}: {: .7f} | Angles: {}".format(i + 1, cost([t1, t2]), [t1, t2]),"\n")

# the minimum energy is -2  for the spin configuration [1,1,-1] which corresponds to
# (phi,theta,omega = (0,0,0) for spin2 and (0,pi,0) for spin3
# We might not always see this value due to the non-convex cost function

print("Optimized angles:",t1, t2)

print("Final cost after optimization:",cost([t1, t2]))
