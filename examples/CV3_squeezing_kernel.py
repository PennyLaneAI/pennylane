"""Continuous-variable quantum kernel classifier example.

In this demo we implement a variation of the "explicit" quantum kernel classifier
from Schuld and Killoran (arXiv:1803.07128) with a 2-dimensional moons data set.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdagradOptimizer

dev = qml.device('strawberryfields.fock', wires=2, cutoff_dim=13)


def featuremap(x):
    """Encode input x into a squeezed state.

    Args:
        x (array[float]): single input vector
    """

    qml.Squeezing(0.5, x[0], wires=[0])
    qml.Squeezing(0.5, x[1], wires=[1])


def layer(v):
    """ Single layer of the classifier.

    Args:
        v (array[float]): array of variables for one layer
    """

    qml.Beamsplitter(v[0], v[1], wires=[0, 1])

    # linear gates in quadrature
    qml.Displacement(v[2], 0., wires=[0])
    qml.Displacement(v[3], 0., wires=[1])

    # quadratic gates in quadrature
    qml.QuadraticPhase(v[4], wires=[0])
    qml.QuadraticPhase(v[5], wires=[1])

    # cubic gates in quadrature
    qml.CubicPhase(v[6], wires=[0])
    qml.CubicPhase(v[7], wires=[1])


@qml.qnode(dev)
def circuit1(var, x=None):
    """The first variational circuit of the quantum classifier.

    Args:
        var (array[float]): array of variables
        x (array[float]): single input vector

    Returns:
        float: expectation of Fock state |2,0>
    """

    # execute feature map
    featuremap(x)

    # execute linear classifier
    for v in var:
        layer(v)

    return qml.expval.NumberState(np.array([2, 0]), [0, 1])


@qml.qnode(dev)
def circuit2(var, x=None):
    """The second variational circuit of the quantum classifier.

    Args:
        var (array[float]): array of variables
        x (array[float]): single input vector

    Returns:
        float: expectation of Fock state |0,2>
    """

    # execute feature map
    featuremap(x)

    # execute linear classifier
    for v in var:
        layer(v)

    return qml.expval.NumberState(np.array([0, 2]), [0, 1])


@qml.qnode(dev)
def trace_(var, x=None):
    """Returns the trace of the final quantum state of the base circuit.
    This 'virtual' circuit is useful for simulations only. Physically, the trace should always be 1,
    but the `strawberryfields.fock` backend may push gate parameters to a regime
    where simulations are not exact any more, and where the trace is smaller than 1.

    Args:
        var (array[float]): array of variables
        x (array[float]): single input vector

    Returns:
        float: identity expectation or trace of the final state
    """

    # execute feature map
    featuremap(x)

    # execute linear classifier
    for v in var:
        layer(v)

    return qml.expval.Identity([0, 1])


def qclassifier(var, x=None):
    """The variational circuit of the quantum classifier.

    Args:
        var (array[float]): array of variables
        x (array[float]): single input vector

    Returns:
        float: continuous output of the classifier

    """

    p1 = circuit1(var, x=x)
    p2 = circuit2(var, x=x)

    return p1 / (p1 + p2)


def cost(var, X, Y):
    """Cost function to be minimized.

    Args:
        var (array[float]): array of variables

    Returns:
        float: scalar cost
    """

    outpts = np.array([qclassifier(var, x=x) for x in X])
    squareloss = np.mean((Y - outpts) ** 2)
    l2regularization = np.sum(np.inner(var, var))

    return squareloss + 0.01*l2regularization


# set seed to make experiment reproducable
np.random.seed(0)

# load function data
data = np.loadtxt("data/blobs.txt")
X = data[:, 0:2]
Y = data[:, 2]

# split into training and validation set
num_data = len(Y)
num_train = int(0.5*num_data)
index = np.random.permutation(range(num_data))
X_train = X[index[: num_train]]
Y_train = Y[index[: num_train]]
X_val = X[index[num_train: ]]
Y_val = Y[index[num_train: ]]

# initialize weights
num_layers = 4
var_init = 0.1*np.random.randn(num_layers, 8)

# create optimizer
o = AdagradOptimizer(0.01)

# train
batch_size = 5
var = var_init


for it in range(150):

    # select minibatch of training samples
    batch_index = np.random.randint(0, num_train, (batch_size,))
    X_train_batch = X_train[batch_index]
    Y_train_batch = Y_train[batch_index]

    # update weights for one step
    var = o.step(lambda v: cost(v, X_train_batch, Y_train_batch), var)

    # compute accuracy on train and validation set, as well as the trace
    # every 5 steps
    if it % 10 == 0:
        pred_train = [np.round(qclassifier(var, x=x_)) for x_ in X_train]
        pred_val = [np.round(qclassifier(var, x=x_)) for x_ in X_val]
        acc_train = np.mean(Y_train == pred_train)
        acc_val = np.mean(Y_val == pred_val)
        current_trace = trace_(var, x=X_train[0])
        print("Iter: {:5d} | Cost: {:0.7f} | Accuracy train/validation: {:0.2f}/{:0.2f} "
              "| Trace: {:0.7f} ".format(it + 1, cost(var, X_train, Y_train),
                                         acc_train, acc_val, current_trace))
    else:
        print("Iter: {:5d} | Cost: {:0.7f} ".format(it + 1, cost(var, X_train, Y_train)))
