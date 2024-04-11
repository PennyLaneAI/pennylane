import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# define the Hamiltonian of interest

dev = qml.device("default.qubit")

H = qml.dot(
    [0.1, 0.3, -0.3],
    [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0), qml.PauliZ(1)])

@qml.qnode(dev)
def circuit(coeffs):
    H = qml.dot(
        coeffs,
        [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0), qml.PauliZ(1)])


    qml.PauliX(0)
    qml.Qubitization(H, control = [2,3])
    return qml.expval(qml.Z(0))

coeffs = np.array(
    [0.1, 0.3, -0.3],
    requires_grad=True)

print(circuit(coeffs))
print(qml.grad(circuit)(coeffs))
