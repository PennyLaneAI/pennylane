import pennylane as qml

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit():
    qml.templates.tile([qml.Hadamard, qml.Hadamard], [0, 1, 1], 3)
    return qml.expval(qml.PauliZ(0))

print(circuit())
print(circuit.draw())
