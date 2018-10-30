"""Continuous-variable quantum kernel classifier example.

In this demo we implement a variation of the "explicit" quantum kernel classifier
from Schuld and Killoran (arXiv:1803.07128) with a 2-dimensional moons data set.
"""

import numpy as np
import openqml as qm
from openqml import numpy as onp
from openqml.optimize import AdamOptimizer

dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=20)


def featuremap(x):
    """Encode input x into a squeezed state.

    Args:
        x (array[float]): single input vector
    """

    qm.Squeezing(1.9, x[0], wires=[0])
    qm.Squeezing(1.9, x[1], wires=[1])


def layer(v):
    """ Single layer of the classifier.

    Args:
        v (array[float]): array of variables for one layer
    """

    qm.Beamsplitter(v[0], v[1], wires=[0, 1])

    # linear gates in quadrature
    qm.Displacement(v[2], 0., wires=[0])
    qm.Displacement(v[3], 0., wires=[1])

    # quadratic gates in quadrature
    qm.QuadraticPhase(v[4], wires=[0])
    qm.QuadraticPhase(v[5], wires=[1])

    # cubic gates in quadrature
    qm.CubicPhase(v[6], wires=[0])
    qm.CubicPhase(v[7], wires=[1])


@qm.qnode(dev)
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

    return qm.expval.NumberState(np.array([2, 0]), [0, 1])


@qm.qnode(dev)
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

    return qm.expval.NumberState(np.array([0, 2]), [0, 1])


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
    """ Proportion of equal labels and predictions.

    Args:
        labels (array[float]): 1-d array of labels
        predictions (array[float]): 1-d array of predictions

    Returns:
        float: scalar accuracy
    """

    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l-p) < 1e-5:
            acc += 1
    acc = acc/len(labels)

    return acc


def regularizer(var):
    """ L2 regularizer

    Args:
        var (array[float]): array of variables

    Returns:
        float: regularization penalty
    """

    reg = onp.sum(onp.inner(var, var))

    return reg


def cost(var, X, Y):
    """Cost function to be minimized.

    Args:
        var (array[float]): array of variables

    Returns:
        float: scalar cost
    """

    regul = regularizer(var)
    outpts = [qclassifier(var, x=x) for x in X]
    loss = square_loss(Y, outpts)

    return loss + 0.001*regul


# load function data
data = np.loadtxt("data/moons.txt")
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
var_init = 0.5*np.random.randn(num_layers, 8)

# create optimizer
o = AdamOptimizer(0.01)

# train
batch_size = 5
var = var_init

for it in range(50):

    # select minibatch of training samples
    batch_index = np.random.randint(0, num_train, (batch_size,))
    X_train_batch = X_train[batch_index]
    Y_train_batch = Y_train[batch_index]

    var = o.step(lambda v: cost(v, X_train_batch, Y_train_batch), var)

    # Compute accuracy on train and validation set
    pred_train = [np.round(qclassifier(var, x=x_)) for x_ in X_train]
    pred_val = [np.round(qclassifier(var, x=x_)) for x_ in X_val]
    acc_train = accuracy(Y_train, pred_train)
    acc_val = accuracy(Y_val, pred_val)

    print("Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
         "".format(it+1, cost(var, X_train, Y_train), acc_train, acc_val))
