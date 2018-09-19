"""Quantum Neural Network.

In this demo we implement a simplified version of
the "circuit-centric classifier" of Ref XXX.
"""

import openqml as qm
from openqml import numpy as np


def square_loss(labels, predictions):
    """ Square loss function

    Args:
        labels (array[float]): 1-d array of labels
        predictions (array[float]): 1-d array of predictions
    Returns:
        float: square loss
    """
    loss = 0
    for l, p in zip(labels, predictions):
        loss += (l-p)**2
    loss = loss/len(labels)

    return loss


def layer(W):
    """ Single layer of the quantum neural net
    with CNOT range 1."""

    qm.Rotation(W[0, 0], W[0, 1], W[0, 2], [0])
    qm.Rotation(W[1, 0], W[1, 1], W[1, 2], [1])
    qm.Rotation(W[2, 0], W[2, 1], W[2, 2], [2])
    qm.Rotation(W[3, 0], W[3, 1], W[3, 2], [2])

    qm.CNOT([0, 1])
    qm.CNOT([1, 2])
    qm.CNOT([2, 3])
    qm.CNOT([3, 0])


def make_random_layer(num_modes):
    """ Randomly initialised layer."""
    W = np.random.randn(num_modes, num_modes)
    return W


dev = qm.device('default.qubit', wires=2)

@qm.qfunc(dev)
def quantum_neural_net(weights, x):
    """The quantum neural net variational circuit."""

    # We cheat and initialise the quantum state to
    # encode the input in its amplitudes

    initial_state = x/np.sqrt(sum(x**2))
    qm.QubitStateVector(initial_state, wires=[0, 1, 2, 3])

    for W in weights:
        layer(W)

    qm.expectation.PauliZ(0)


def cost(weights, features, labels):
    """Cost (error) function to be minimized."""

    # Compute prediction for each input in data batch
    predictions = np.zeros((len(features), ))
    for idx, x in enumerate(features):
        predictions[idx] = quantum_neural_net(weights, x)

    return square_loss(labels, predictions)


# load Iris data
data = np.loadtxt("iris.txt")

# initialize weights
num_layers = 2
weights0 = [make_random_layer(num_layers) for _ in range(num_layers)]
o = qm.Optimizer(cost, weights0)

# train the circuit: HOW TO FEED IN DATA?
batch_size = 3
steps = 10
for steps in range(steps):
    batch_index = np.random.integer(batch_size)
    batch = data[batch_index]
    o.step(X=batch[:, 1:], y=batch[:, 0])
    print(o.cost())

# print the results
print('Initial rotation angles:', weights0)
print('Optimized rotation angles:', o.weights)
