"""Continuous-variable quantum neural network example.

In this demo we implement the photonic quantum neural net model
from Killoran et al. (arXiv:1806.06871) with the example
of function fitting.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

try:
    dev = qml.device('strawberryfields.fock', wires=1, cutoff_dim=10)    
except:
    print("To run this demo you need to install the strawberryfields plugin...")


def layer(v):
    """ Single layer of the quantum neural net.

    Args:
        v (array[float]): array of variables for one layer
    """
    # Matrix multiplication of input layer
    qml.Rotation(v[0], wires=0)
    qml.Squeezing(v[1], 0., wires=0)
    qml.Rotation(v[2], wires=0)

    # Bias
    qml.Displacement(v[3], 0., wires=0)

    # Element-wise nonlinear transformation
    qml.Kerr(v[4], wires=0)


@qml.qnode(dev)
def quantum_neural_net(var, x=None):
    """The quantum neural net variational circuit.

    Args:
        var (array[float]): array of variables
        x (array[float]): single input vector

    Returns:
        float: expectation of Homodyne measurement on Mode 0
    """
    # Encode input x into quantum state
    qml.Displacement(x, 0., wires=0)

    # "layer" subcircuits
    for v in var:
        layer(v)

    return qml.expval.X(0)


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
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)

    return loss


def cost(var, features, labels):
    """Cost function to be minimized.

    Args:
        var (array[float]): array of variables
        features (array[float]): 2-d array of input vectors
        labels (array[float]): 1-d array of targets

    Returns:
        float: loss
    """
    # Compute prediction for each input in data batch
    preds = [quantum_neural_net(var, x=x) for x in features]

    return square_loss(labels, preds)


# load function data
data = np.loadtxt("data/sine.txt")
X = data[:, 0]
Y = data[:, 1]

# initialize weights
np.random.seed(0)
num_layers = 4
var_init = 0.05 * np.random.randn(num_layers, 5)

# create optimizer
opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)

# train
var = var_init
for it in range(500):
    var = opt.step(lambda v: cost(v, X, Y), var)
    
    print("Iter: {:5d} | Cost: {:0.7f} ".format(it + 1, cost(var, X, Y)))
