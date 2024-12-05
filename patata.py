import pennylane as qml
import numpy as np


# P(x) = -1 + 0.2 x^2 + 0.5 x^4
poly = np.array([-0.1, 0, 0.2, 0, 0.5])

A = np.array([[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]])

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():
    qml.qsvt(A, poly, encoding_wires=[0, 1, 2, 3 , 4], block_encoding="fable")
    return qml.state()

matrix = qml.matrix(circuit, wire_order=[0, 1, 2, 3, 4])()

print(np.round(matrix[:4, :4], 4).real)