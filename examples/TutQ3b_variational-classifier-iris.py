"""Variational quantum classifier.

In this demo we implement a variational classifier inspired by
Schuld et al. 2018 (arXiv:1804.00633).
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

dev = qml.device('default.qubit', wires=2)


def layer(W):
    """ Single layer of the quantum neural net.

    Args:
        W (array[float]): 2-d array of variables for one layer
    """

    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=[0])
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=[1])

    qml.CNOT(wires=[0, 1])


def alphas(phases):
    """Angles for z rotations in state preparation routine"""

    alpha0 = (phases[1] - phases[0])
    alpha1 = (phases[3] - phases[2])
    alpha2 = 1/2*(phases[2] - phases[0] + phases[3] - phases[1])

    return alpha0, alpha1, alpha2


def betas(ax):
    """Angles for y rotations in state preparation routine"""

    beta0 = 2*np.arcsin(ax[1] / np.sqrt(ax[0]**2 + ax[1]**2))
    beta1 = 2*np.arcsin(ax[3] / np.sqrt(ax[2]**2 + ax[3]**2))
    beta2 = 2*np.arcsin(np.sqrt(ax[2]**2 + ax[3]**2) / np.sqrt(ax[0]**2 + ax[1]**2 + ax[2]**2 + ax[3]**2))

    return beta0, beta1, beta2


def angles(x):
    """Compute coefficients needed to prepare a quantum state
    whose ket vector is `x`"""

    absvals = np.abs(x)
    phases = [0 if np.sign(x_) == 1 else np.pi for x_ in x]

    beta0, beta1, beta2 = betas(absvals)
    alpha0, alpha1, alpha2 = alphas(phases)

    return [beta2, -beta1/2, beta1/2, -beta0/2, beta0/2,
            alpha2, alpha1/2, -alpha1/2, alpha0/2, -alpha0/2]


def statepreparation(angls):

    """Quantum circuit for amplitude encoding in 2 qubits.
    See Mottoennen et al 2004.

    Args:
        angls: feature vector of rotation angles
    """
    qml.QubitStateVector(angls, [0, 1])

    # # encode absolute values
    # qml.RY(angls[0], wires=[0])
    #
    # qml.CNOT(wires=[0, 1])
    # qml.RY(angls[1], wires=[1])
    # qml.CNOT(wires=[0, 1])
    # qml.RY(angls[2], wires=[1])
    #
    # qml.PauliX(wires=[0])
    # qml.CNOT(wires=[0, 1])
    # qml.RY(angls[3], wires=[1])
    # qml.CNOT(wires=[0, 1])
    # qml.RY(angls[4], wires=[1])
    # qml.PauliX(wires=[0])
    #
    # # encode signs up to global phase
    # qml.RZ(angls[5], wires=[0])
    #
    # qml.RZ(angls[6], wires=[1])
    # qml.CNOT(wires=[0, 1])
    # qml.RZ(angls[7], wires=[1])
    # qml.CNOT(wires=[0, 1])
    #
    # qml.PauliX(wires=[0])
    # qml.RZ(angls[8], wires=[1])
    # qml.CNOT(wires=[0, 1])
    # qml.RZ(angls[9], wires=[1])
    # qml.CNOT(wires=[0, 1])
    # qml.PauliX(wires=[0])


@qml.qnode(dev)
def circuit(weights, betas=None):
    """The circuit of the variational classifier."""

    statepreparation(betas)

    for W in weights:
        layer(W)

    return qml.expval.PauliZ(0)


def variational_classifier(var, betas=None):
    """The variational classifier."""

    weights = var[0]
    bias = var[1]

    return circuit(weights, betas=betas) + bias


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


def cost(weights, features, labels):
    """Cost (error) function to be minimized."""

    predictions = [variational_classifier(weights, betas=f) for f in features]

    return square_loss(labels, predictions)


# load Iris data and normalise feature vectors
data = np.loadtxt("data/iris_classes1and2_scaled.txt")
X = data[:, :-1]
normalization = np.sqrt(np.sum(X ** 2, -1))
X = (X.T / normalization).T  # normalize each input
features = X #np.array([angles(x) for x in X])  # angles for state preparation are new features

Y = data[:, -1]
labels = Y*2 - np.ones(len(Y))  # shift from {0, 1} to {-1, 1}

# split into training and validation set
num_data = len(labels)
num_train = int(0.75*num_data)
index = np.random.permutation(range(num_data))
feats_train = features[index[: num_train]]
labels_train = labels[index[: num_train]]
feats_val = features[index[num_train: ]]
labels_val = labels[index[num_train: ]]

# initialize weight layers
num_qubits = 2
num_layers = 6
var_init = (0.01*np.random.randn(num_layers, num_qubits, 3), 0.0)

# create optimizer
o = GradientDescentOptimizer(0.01)
batch_size = 5

# train the variational classifier
var = var_init
for iteration in range(20):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size, ))
    feats_train_batch = feats_train[batch_index]
    labels_train_batch = labels_train[batch_index]
    var = o.step(lambda v: cost(v, feats_train_batch, labels_train_batch), var)

    # Compute predictions on train and validation set
    predictions_train = [np.sign(variational_classifier(var, betas=f)) for f in feats_train]
    predictions_val = [np.sign(variational_classifier(var, betas=f)) for f in feats_val]

    # Compute accuracy on train and validation set
    acc_train = accuracy(labels_train, predictions_train)
    acc_val = accuracy(labels_val, predictions_val)

    print("Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
          "".format(iteration, cost(var, features, Y), acc_train, acc_val))

import matplotlib.pyplot as plt

plt.figure()
cm = plt.cm.RdBu

# make data for decision regions
xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 20), np.linspace(-1.1, 1.1, 20))
X_grid = [np.array([0, 0, x, y]) for x, y in zip(xx.flatten(), yy.flatten())]
predictions_grid = [variational_classifier(var, x=x) for x in X_grid]
Z = np.reshape(predictions_grid, xx.shape)

# plot decision regions
cnt = plt.contourf(xx, yy, Z, levels=np.arange(-1, 1.1, 0.1), cmap=cm, alpha=.8, extend='both')
plt.colorbar(cnt, ticks=[-1, 0, 1])

# plot data
trf0 = [d for i, d in enumerate(feats_train) if labels_train[i] == -1]
trf1 = [d for i, d in enumerate(feats_train) if labels_train[i] == 1]
plt.scatter([c[0] for c in trf1], [c[1] for c in trf1], c='r', marker='^', edgecolors='k')
plt.scatter([c[0] for c in trf0], [c[1] for c in trf0], c='r', marker='o', edgecolors='k')
tes0 = [d for i, d in enumerate(feats_val) if labels_val[i] == -1]
tes1 = [d for i, d in enumerate(feats_val) if labels_val[i] == 1]
plt.scatter([c[0] for c in tes1], [c[1] for c in tes1], c='g', marker='^', edgecolors='k')
plt.scatter([c[0] for c in tes0], [c[1] for c in tes0], c='g', marker='o', edgecolors='k')

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()