"""Quantum Neural Network.

In this demo we implement a simplified version of
the "circuit-centric classifier" of Ref XXX.
"""

import openqml as qm
from openqml import numpy as onp
import numpy as np
from openqml._optimize import GradientDescentOptimizer


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


def regularizer(weights):
    """Regularizer penalty on weights"""
    return onp.abs(weights.flatten()*2)


def layer(W):
    """ Single layer of the quantum neural net
    with CNOT range 1."""

    qm.Rot(W[0, 0], W[0, 1], W[0, 2], [0])
    qm.Rot(W[1, 0], W[1, 1], W[1, 2], [1])
    qm.Rot(W[2, 0], W[2, 1], W[2, 2], [2])
    qm.Rot(W[3, 0], W[3, 1], W[3, 2], [3])

    qm.CNOT([0, 1])
    qm.CNOT([1, 2])
    qm.CNOT([2, 3])
    qm.CNOT([3, 0])


def make_random_layer(num_modes):
    """ Randomly initialised layer."""
    W = onp.random.randn(num_modes, num_modes)
    return W


dev = qm.device('default.qubit', wires=4)


@qm.qfunc(dev)
def quantum_neural_net(weights, x):
    """The quantum neural net variational circuit."""

    # We cheat and initialise the quantum state to
    # encode the input in its amplitudes

    qm.QubitStateVector(x, wires=[0, 1, 2, 3])

    for W in weights:
        layer(W)

    return qm.expectation.PauliZ(0)


def cost(weights, features, labels):
    """Cost (error) function to be minimized."""

    # Compute prediction for each input in data batch
    predictions = onp.zeros((len(features), ))
    for idx, x in enumerate(features):
        x = x / onp.sqrt(np.sum(x ** 2))
        predictions[idx] = quantum_neural_net(weights, x)

    return square_loss(labels, predictions) + regularizer(weights)


# load Iris data
data = np.loadtxt("iris.txt")

# initialize weights
num_qubits = 4
num_layers = 2
weights0 = [make_random_layer(num_qubits) for _ in range(num_layers)]
o = GradientDescentOptimizer(0.1)

# train the circuit: HOW TO FEED IN DATA?
batch_size = 3

weights = weights0
for steps in range(11):
    batch_index = np.random.randint(0, len(data), (batch_size, ))
    batch = data[batch_index]
    X = batch[:, 1:]
    y = batch[:, 0]
    
    weights = o.step(lambda w: cost(w, X, y), weights)
    print(cost(weights, X, y))

