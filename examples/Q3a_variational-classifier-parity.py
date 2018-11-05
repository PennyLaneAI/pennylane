"""Variational quantum classifier

This example shows that a variational quantum classifier
can be optimized to reproduce the parity function.

"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdagradOptimizer

dev = qml.device('default.qubit', wires=4)


def layer(W):
    """ Single layer of the quantum neural net.

    Args:
        W: array of variables
    """

    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=[0])
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=[1])
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=[2])
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=[3])

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])


def statepreparation(x):
    """ Encodes data input x into quantum state.

    Args:
        x: single input vector
    """

    qml.BasisState(x, wires=[0, 1, 2, 3])


@qml.qnode(dev)
def circuit(var, x=None):
    """The circuit of the variational classifier.

    Args:
        var (array[float]): array of variables
        x: single input vector

    Returns:
        expectation of Pauli-Z operator on Qubit 0
    """

    statepreparation(x)

    for V in var:
        layer(V)

    return qml.expval.PauliZ(0)


def variational_classifier(var, x=None):
    """The variational classifier.

    Args:
        var (array[float]): array of variables
        x: single input vector

    Returns:
        continuous output of the model
    """

    weights = var[0]
    bias = var[1]

    return circuit(weights, x=x) + bias


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


def accuracy(labels, predictions):
    """ Share of equal labels and predictions

    Args:
        labels (array[float]): 1-d array of labels
        predictions (array[float]): 1-d array of predictions

    Returns:
        float: accuracy
    """

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l-p) < 1e-5:
            loss += 1
    loss = loss/len(labels)

    return loss


def cost(var, X, Y):
    """Cost (error) function to be minimized."""

    predictions = [variational_classifier(var, x=x) for x in X]

    return square_loss(Y, predictions)


# load parity data
data = np.loadtxt("data/parity.txt")
X = data[:, :-1]
Y = data[:, -1]
Y = Y*2 - np.ones(len(Y))  # shift label from {0, 1} to {-1, 1}

# initialize weight layers
num_qubits = 4
num_layers = 2
var_init = (0.01*np.random.randn(num_layers, num_qubits, 3), 0.0)

# create optimizer
o = AdagradOptimizer(0.5)
batch_size = 5

# train the variational classifier
var = var_init
for it in range(5):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, len(X), (batch_size, ))
    X_batch = X[batch_index]
    Y_batch = Y[batch_index]
    var = o.step(lambda v: cost(v, X, Y), var)

    # Compute accuracy
    predictions = [np.sign(variational_classifier(var, x=x)) for x in X]
    acc = accuracy(Y, predictions)

    print("Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} "
          "".format(it+1, cost(var, X, Y), acc))

