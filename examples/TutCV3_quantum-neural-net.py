"""Continuous-variable quantum neural network example.

In this demo we implement the photonic quantum neural net model
from Killoran et al. (arXiv:1806.06871) with the example
of function fitting.
"""

import openqml as qm
from openqml import numpy as np
from openqml.optimize import AdamOptimizer

dev = qm.device('strawberryfields.fock', wires=1, cutoff_dim=10)


def layer(v):
    """ Single layer of the quantum neural net.

    Args:
        v (array[float]): array of variables for one layer
    """

    # Bias
    qm.Displacement(v[0], v[1], [0])

    # Matrix multiplication of input layer
    qm.Rotation(v[2], [0])
    qm.Squeezing(v[3], v[4], [0])
    qm.Rotation(v[5], [0])

    # Nonlinear transformation
    qm.Kerr(v[6], [0])


@qm.qnode(dev)
def quantum_neural_net(var, x=None):
    """The quantum neural net variational circuit.

    Args:
        var (array[float]): array of variables
        x (array[float]): single input vector

    Returns:
        float: expectation of Homodyne measurement on Mode 0
    """

    # Encode input x into quantum state
    qm.Displacement(x, 0., [0])

    # execute "layers"
    for v in var:
        layer(v)

    return qm.expval.X(0)


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


def cost(var, X, Y):
    """Cost function to be minimized.

    Args:
        var (array[float]): array of variables
        X (array[float]): 2-d array of input vectors
        Y (array[float]): 1-d array of targets

    Returns:
        float: loss
    """

    # Compute prediction for each input in data batch
    preds = [quantum_neural_net(var, x=x) for x in X]

    loss = square_loss(Y, preds)

    return loss


# load function data
data = np.loadtxt("sine.txt")
X = data[:, 0]
Y = data[:, 1]

# initialize weights
num_layers = 4
var_init = 0.05*np.random.randn(num_layers, 7)

# create optimizer
o = AdamOptimizer(0.005, beta1=0.9, beta2=0.999)

# train
var = var_init
for it in range(50):
    var = o.step(lambda v: cost(v, X, Y), var)
    print("Iter: {:5d} | Cost: {:0.7f}".format(it+1, cost(var, X, Y)))

