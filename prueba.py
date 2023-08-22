import pennylane as qml

dev = qml.device("default.qubit", wires = 2)

@qml.qnode(dev)
def circuit():
    qml.ctrl_evolution(qml.PauliX(0), control = 1)
    return qml.state()

print(circuit())