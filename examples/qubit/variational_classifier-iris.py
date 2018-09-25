"""Quantum Neural Network.

In this demo we implement a variational classifier inspired by
Farhi & Neven 2018 (arXiv:1802.06002) and
Schuld et al. 2018 (arXiv:1804.00633).
"""

import openqml as qm
from openqml import numpy as onp
import numpy as np
from openqml._optimize import AdagradOptimizer
import matplotlib.pyplot as plt

from math import isclose

dev = qm.device('default.qubit', wires=4)


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


def statepreparation(x):
    """ Encodes data input x into a quantum state, using a feature map
    that first projects x -> x \otimes x. This gives the variational classifier
    more power."""

    # We cheat and hard-set the initial quantum state to x \otimes x
    qm.QubitStateVector(onp.kron(x, x), wires=[0, 1, 2, 3])


@qm.qfunc(dev)
def variational_classifier(weights, x=None):
    """The circuit of the variational classifier."""

    statepreparation(x)

    for W in weights:
        layer(W)

    return qm.expectation.PauliZ(0)


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
        if isclose(l, p, abs_tol=1e-5):
            loss += 1
    loss = loss/len(labels)

    return loss


def regularizer(weights):
    """L2 Regularizer penalty on weights

    Args:
        weights (array[float]): The array of trainable weights
    Returns:
        float: regularization penalty
    """
    w_flat = weights.flatten()

    # Compute the l2 norm
    reg = onp.abs(onp.inner(w_flat, w_flat))

    return reg


def cost(weights, features, labels):
    """Cost (error) function to be minimized."""

    predictions = [variational_classifier(weights, x=x) for x in features]

    return square_loss(labels, predictions) #+ regularizer(weights)


# load Iris data and normalise feature vectors
data = np.loadtxt("iris_scaled.txt")
X = data[:, :-1]
normalization = np.sqrt(np.sum(X ** 2, -1))
X = (X.T / normalization).T  # normalize each feature vector
Y = data[:, -1]
Y = Y*2 - np.ones(len(Y))  # shift label from {0, 1} to {-1, 1}

# split into training and validation set
num_data = len(X)
num_train = int(0.75*num_data)
index = np.random.permutation(range(num_data))
X_train = X[index[: num_train]]
Y_train = Y[index[: num_train]]
X_val = X[index[num_train: ]]
Y_val = Y[index[num_train: ]]

# initialize weight layers
num_qubits = 4
num_layers = 2
weights0 = [np.random.randn(num_qubits, num_qubits)] * num_layers

# create optimizer
o = AdagradOptimizer(0.1)

# train the variational classifier
batch_size = 5
weights = np.array(weights0)

for iteration in range(1):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size, ))
    X_train_batch = X_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    weights = o.step(lambda w: cost(w, X_train_batch, Y_train_batch), weights)

    # Compute predictions on train and validation set
    predictions_train = [np.sign(variational_classifier(weights, x=x)) for x in X_train]
    predictions_val = [np.sign(variational_classifier(weights, x=x)) for x in X_val]

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print("Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
          "".format(iteration, cost(weights, X, Y), acc_train, acc_val))


# start plotting
plt.figure()
cm = plt.cm.RdBu

# make data for decision regions
xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 20), np.linspace(-1.1, 1.1, 20))
X_grid = [np.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())]
predictions_grid = [variational_classifier(weights, x=x) for x in X_grid]
Z = np.reshape(predictions_grid, xx.shape)

# plot decision regions
cnt = plt.contourf(xx, yy, Z, levels=np.arange(-1, 1.1, 0.1), cmap=cm, alpha=.8, extend='both')
plt.colorbar(cnt, ticks=[-1, 0, 1])

# plot data
trf0 = [d for i, d in enumerate(X_train) if Y_train[i] == -1]
trf1 = [d for i, d in enumerate(X_train) if Y_train[i] == 1]
plt.scatter([c[0] for c in trf1], [c[1] for c in trf1], c='r', marker='^', edgecolors='k')
plt.scatter([c[0] for c in trf0], [c[1] for c in trf0], c='r', marker='o', edgecolors='k')
tes0 = [d for i, d in enumerate(X_val) if Y_val[i] == -1]
tes1 = [d for i, d in enumerate(X_val) if Y_val[i] == 1]
plt.scatter([c[0] for c in tes1], [c[1] for c in tes1], c='g', marker='^', edgecolors='k')
plt.scatter([c[0] for c in tes0], [c[1] for c in tes0], c='g', marker='o', edgecolors='k')

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()