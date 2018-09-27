"""Variational classifier

We train a variational classifier to learn parity.

REF: Farhi & Neven 2018

"""

import openqml as qm
from openqml import numpy as onp
import numpy as np
from openqml._optimize import GradientDescentOptimizer

from math import isclose


dev = qm.device('default.qubit', wires=4)


def layer(W):
    """ Single layer of the quantum neural net."""

    qm.Rot(W[0, 0], W[0, 1], W[0, 2], [0])
    qm.Rot(W[1, 0], W[1, 1], W[1, 2], [1])
    qm.Rot(W[2, 0], W[2, 1], W[2, 2], [2])
    qm.Rot(W[3, 0], W[3, 1], W[3, 2], [3])

    qm.CNOT([0, 1])
    qm.CNOT([1, 2])
    qm.CNOT([2, 3])
    qm.CNOT([3, 0])


def statepreparation(x):
    """ Encodes data input x into quantum state."""

    for i in range(len(x)):
        if x[i] == 1:
            qm.PauliX([i])


@qm.qfunc(dev)
def quantum_neural_net(weights, x=None):
    """The quantum neural net variational circuit."""

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

    predictions = [quantum_neural_net(weights, x=x) for x in features]

    return square_loss(labels, predictions) # + regularizer


# load Iris data and normalise feature vectors
data = np.loadtxt("parity.txt")
X = data[:, :-1]
Y = data[:, -1]
Y = Y*2 - np.ones(len(Y)) # shift label from {0, 1} to {-1, 1}

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
num_layers = 6
weights0 = [np.random.randn(num_qubits, num_qubits)] * num_layers

# create optimizer
o = GradientDescentOptimizer(0.1)
batch_size = 3

# train the variational classifier
weights = np.array(weights0)
for iteration in range(50):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size, ))
    X_train_batch = X_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    weights = o.step(lambda w: cost(w, X_train_batch, Y_train_batch), weights)

    # Compute predictions on train and validation set
    predictions_train = [np.sign(quantum_neural_net(weights, x=x)) for x in X_train]
    predictions_val = [np.sign(quantum_neural_net(weights, x=x)) for x in X_val]

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print("Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
          "".format(iteration, cost(weights, X, Y), acc_train, acc_val))


    plt.figure(figsize=(5, 4))

    extendblue = (21/float(256), 80/float(256), 141/float(256))
    extendred = (150/float(256), 15/float(256), 39/float(256))
    cm = plt.cm.RdBu
    cm_train = ListedColormap(['#FF0000', '#0000FF'])
    cm_test = ListedColormap(['#e4721aff', '#38176bff'])

    if not grid_labels is None:

        grid_steps = int(np.sqrt(len(grid_labels)))
        xx, yy = np.meshgrid(np.linspace(0, 1.1, grid_steps),
                             np.linspace(0, 1.1, grid_steps))
        Z = np.reshape(grid_labels, xx.shape)
        cnt = plt.contourf(xx, yy, Z, levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], cmap=cm, alpha=.8, extend='both')
        cnt.cmap.set_under(extendred)
        cnt.cmap.set_over(extendblue)
        plt.contour(xx, yy, Z, levels=[0.5], colors=('black',), linestyles=('--',), linewidths=(0.8,))
        plt.colorbar(cnt, ticks=[0, 1])  #[np.amin(Z), np.amax(Z)])


    if not (features is None or labels is None):

        trf0 = [d for i, d in enumerate(features) if labels[i] == 0]
        trf1 = [d for i, d in enumerate(features) if labels[i] == 1]
        plt.scatter([c[0] for c in trf1], [c[1] for c in trf1], cmap=cm_train, marker='^', edgecolors='k')
        plt.scatter([c[0] for c in trf0], [c[1] for c in trf0], cmap=cm_train, marker='o', edgecolors='k')

    if not (testfeats is None or testlabels is None):
        tes0 = [d for i, d in enumerate(testfeats) if testlabels[i] == 0]
        tes1 = [d for i, d in enumerate(testfeats) if testlabels[i] == 1]
        plt.scatter([c[0] for c in tes1], [c[1] for c in tes1], cmap=cm_test, marker='^', edgecolors='k')
        plt.scatter([c[0] for c in tes0], [c[1] for c in tes0], cmap=cm_test, marker='o', edgecolors='k')

    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)