import pennylane as qml
import numpy as np

A = np.array([[1, 0], [0, -1]])
print(((3 * qml.PauliZ(0) + (qml.PauliZ(1) @ qml.PauliZ(0)) - qml.PauliZ(0)) @ qml.Hamiltonian([1, 1], [qml.PauliX(2), qml.Hermitian(A, 2) @ qml.PauliY(3)])).terms)
print(qml.Hamiltonian([1], [qml.PauliZ(0) @ qml.PauliZ(1) @ qml.Identity(2)]) == qml.PauliZ(0) @ qml.PauliZ(1))