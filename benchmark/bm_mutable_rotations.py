"""Mutable sequence of rotations benchmark.
The benchmark consists of a single mutable QNode evaluated several times,
each evaluation having a larger number of rotations in the circuit.
"""
import numpy as np

import pennylane as qml

n = 30
n_min = 10
n_max = 60
n_step = 10

n_wires = 1

dev = qml.device("default.qubit", wires=n_wires)


def circuit(a, b=1):
    for idx in range(b):
        qml.RY(a[idx], wires=[0])
    return qml.expval(qml.PauliX(0))


try:
    circuit = qml.qnodes.QubitQNode(circuit, dev, mutable=True)
except AttributeError:
    circuit = qml.QNode(circuit, dev, cache=False)


def benchmark(num_evaluations=30):
    """Mutable rotations"""

    # Execution time should grow quadratically in num_evaluations as
    # the circuit size grows linearly with evaluation number.

    wrong_results = 0
    for i in range(num_evaluations):
        params = np.random.rand(i)
        res = circuit(params, b=i)

        expected = np.sin(np.sum(params))
        if np.abs(res - expected) > 1e-6:
            wrong_results += 1

    # print("Wrong results: {}/{}".format(wrong_results, num_evaluations))
    return True
