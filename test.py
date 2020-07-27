import pennylane as qml

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit():
    qml.templates.tile([qml.Hadamard, qml.CNOT, qml.PauliX], [0, [0, 1], 0], 3)
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

print(circuit())
print(circuit.draw())
