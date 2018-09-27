import openqml as qm
from openqml import numpy as np
from openqml._optimize import GradientDescentOptimizer
import matplotlib.pyplot as plt

dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=10)


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


def layer(W, b):
    """ Single layer of the continuous-variable quantum neural net."""

    U, d, V = np.linalg.grad_svd(W)

    # Matrix multiplication of input layer
    qm.Interferometer(U, [0, 1])
    qm.Squeezing(-np.log(d[0]), [0])
    qm.Interferometer(V, [0, 1])

    # Bias
    qm.Displacement(b[0], [0])

    # Element-wise nonlinear transformation
    qm.Kerr(0.1, [0])


@qm.qfunc(dev)
def quantum_neural_net(weights, x=None):
    """The quantum neural net variational circuit."""

    # Encode 2-d input into quantum state
    qm.Displacement(x[0], [0])

    # execute "layers"
    for W, b in weights:
        layer(W, b)

    return qm.expectation.Homodyne()


def cost(weights, features, labels):
    """Cost (error) function to be minimized."""

    # Compute prediction for each input in data batch
    predictions = np.zeros((len(features), ))
    for idx, x in enumerate(features):
        predictions[idx] = quantum_neural_net(weights, x)

    return square_loss(labels, predictions)


def make_random_layer(num_modes):
    """ Randomly initialised layer."""
    W = np.random.randn(num_modes, num_modes)
    b = np.random.randn(num_modes)
    return W, b