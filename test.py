import pennylane as qml

hamiltonian = qml.Hamiltonian([1, 1, 1, -1], [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1), qml.PauliX(0) @ qml.PauliX(1)])
print(hamiltonian.is_diagonal)

