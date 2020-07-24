import pennylane as qml
import numpy as np

H = qml.Hamiltonian([2, 1], [qml.Hermitian(np.array([[0, 1], [1, 0]]), 0) @ qml.Hermitian(np.array([[0, 1], [1, 0]]), 1), qml.PauliX(1)])
#H = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)])

print(H.decompose())
