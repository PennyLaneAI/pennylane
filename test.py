from pennylane import qaoa
import pennylane as qml

mixer_h = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(0) @ qml.PauliX(1)])
mixer_layer = qaoa.mixer_layer(mixer_h)

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(alpha):

    mixer_layer(alpha)

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(2)]

circuit(0.5)
print(circuit.draw())