"""Continuous-variable quantum neural network.

In this demo we implement the cv-qnn of Ref XXX with the
example of function fitting.
"""

import openqml as qm
from openqml import numpy as np
from openqml._optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt

dev = qm.device('strawberryfields.fock', wires=1, cutoff_dim=10)


def layer(w):
    """ Single layer of the continuous-variable quantum neural net."""

    # Matrix multiplication of input layer
    qm.Rotation(w[0], [0])
    qm.Squeezing(w[1], w[2], [0])
    qm.Rotation(w[3], [0])

    # Bias
    qm.Displacement(w[4], w[5], [0])

    # Element-wise nonlinear transformation
    qm.Kerr(w[6], [0])


@qm.qfunc(dev)
def quantum_neural_net(weights, x=None):
    """The quantum neural net variational circuit."""

    # Encode 2-d input into quantum state
    qm.Displacement(x[0], 0., [0])

    # execute "layers"
    for i in range(6):
        layer(weights[i*7: i*7+7])

    return qm.expectation.X(0)


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
    w_flat = weights.flatten()
    return np.abs(np.sum(w_flat**2))


def cost(weights, features, labels):
    """Cost (error) function to be minimized."""

    # Compute prediction for each input in data batch
    predictions = [quantum_neural_net(weights, x=x) for x in features]

    loss = square_loss(labels, predictions)
    regularization = 0.5*regularizer(weights)

    cost = loss + 0.0 * regularization

    return cost


# load function data
data = np.loadtxt("sine.txt")
X = data[:, :-1]
Y = data[:, -1]

# initialize weights
num_layers = 6
weights0 = 0.05*np.random.randn(num_layers*7)

# create optimizer
o = NesterovMomentumOptimizer(0.1)

# train
weights = weights0
for iteration in range(15):
    weights = o.step(lambda w: cost(w, X, Y), weights)
    print("Iter: {:5d} | Cost: {:0.7f}".format(iteration, cost(weights, X, Y)))

# predict a range of values between -1 and 1
x_axis = np.linspace(-1, 1, 50)
predictions = [quantum_neural_net(weights, x=x) for x in x_axis]

# plot the noisy data (red) and predictions (green)
plt.figure()
plt.plot(x_axis, predictions, color='#3f9b0b', marker='o', zorder=1)
plt.scatter(X, Y, color='#fb2943', marker='o', zorder=2, s=75)
plt.xlabel('Input', fontsize=18)
plt.ylabel('Output', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.show()