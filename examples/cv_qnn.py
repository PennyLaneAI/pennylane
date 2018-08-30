"""Variational quantum eigensolver example.

In this demo we use a variational circuit as an ansatz for
a VQE, and optimize the circuit to lower the energy expectation
of a user-defined Hamiltonian.
"""

import openqml as qm
from openqml import numpy as np

dev1 = qm.device('strawberryfields.fock', wires=2, cutoff=15)


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
    qm.Squeezing(-np.log(d[1]), [1])
    qm.Interferometer(V, [0, 1])

    # Bias
    qm.Displacement(b[0], [0])
    qm.Displacement(b[1], [1])

    # Element-wise nonlinear transformation
    qm.Kerr(0.1, [0])
    qm.Kerr(0.1, [1])


@qm.qfunc(dev1)
def quantum_neural_net(weights, x):
    """QNode"""

    # Encode 2-d input into quantum state
    qm.Displacement(x[0], [0])
    qm.Displacement(x[1], [1])

    # execute "layers"
    for W, b in weights:
        layer(W, b)

    qm.expectation.Homodyne(0)
    qm.expectation.Homodyne(1)


def cost(weights, features, labels):  # Todo: remove batch
    """Cost (error) function to be minimized."""

    # Compute prediction for each input in data batch
    predictions = np.zeros((len(features), ))
    for idx, x in enumerate(features):
        predictions[idx] = quantum_neural_net(weights, x)

    return square_loss(labels, predictions)


# initialize x with random value
num_layers = 2

def make_random_layer(num_modes):
    """ Randomly initialised layer."""
    W = np.random.randn((num_modes, num_modes))
    b = np.random.randn((num_modes,))
    return (W, b)


weights0 = [make_random_layer(2) for _ in range(num_layers)]
o = qm.Optimizer(cost, weights0)

# train the circuit
o.train(max_steps=100)

# print the results
print('Initial rotation angles:', weights0)
print('Optimized rotation angles:', o.weights)

# Does not learn!!!!!!!!!!????????