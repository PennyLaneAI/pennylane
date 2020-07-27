import pennylane as qml

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit(params):
    qml.templates.tile([qml.Hadamard, qml.MultiRZ, qml.MultiRZ], [0, [0, 1], [0, 1]], 3, params)
    return qml.expval(qml.PauliZ(0))

par = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
print(circuit(par))
print(circuit.draw())
