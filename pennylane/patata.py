import pennylane as qml

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():

    return qml.state()

print(circuit())