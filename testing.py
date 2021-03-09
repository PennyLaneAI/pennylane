import pennylane as qml
qml.enable_tape()

dev = qml.device('default.qubit', wires=3)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    H = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliX(1)])
    return qml.expval(H)

print(circuit())
