import pennylane as qml

dev = qml.device("sparse.qubit")

@qml.qnode(dev)
def circuit():
    qml.X(0)
    qml.X(1)
    return qml.state()

print(circuit())
