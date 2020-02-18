"""Entangling layers benchmark.
Creates an immutable QNode using the StronglyEntanglingLayers template,
then evaluates it and its Jacobian."""
import numpy as np

import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.init import strong_ent_layers_uniform

n = 3
n_min = 1
n_max = 5
n_step = 1

n_wires = 3

dev = qml.device("default.qubit", wires=n_wires)


def circuit(weights, features=None):
    AngleEmbedding(features, range(n_wires))
    StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0))


def benchmark(n_layers=3):
    """Entangling layers"""

    # print("circuit: {} layers, {} wires".format(n_layers, n_wires))
    features = np.arange(n_wires)
    init_weights = strong_ent_layers_uniform(n_layers=n_layers, n_wires=n_wires)

    node = qml.qnodes.QubitQNode(circuit, dev, mutable=False)

    res = node(init_weights, features=features)
    # print(node.draw())

    jac = node.jacobian(init_weights, {"features": features})
    return True
