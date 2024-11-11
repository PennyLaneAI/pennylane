import pennylane as qml
import numpy as np

# P(x) = -0.1 + 0.2j x + 0.3 x^2
poly = [0.1, 0.2j, 0.3]

angles = qml.math.poly_to_angles(poly, "GQSP")

@qml.prod
def unitary(wires):
    qml.RX(0.3, wires)

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit(angles):
    qml.GQSP(unitary(1), angles, control = 0)
    return qml.state()


matrix = qml.matrix(circuit, wire_order=[0, 1])(angles)

print(np.round(matrix, 3)[:2, :2])
