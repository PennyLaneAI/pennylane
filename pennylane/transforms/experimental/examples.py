import pennylane as qml
from pennylane.transforms.experimental import cancel_inverses, param_shift_experimental
dev = qml.device("default.qubit", wires = 2)

qml.enable_return()

@qml.qnode_experimental(device=dev)
def circuit(a, b, c):
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=0)
    qml.RX(a, wires=0)
    qml.RY(b, wires=1)
    qml.CRX(a, wires=[0, 1])
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=1)
    return qml.expval(qml.PauliZ(wires=0))

a = qml.numpy.array(0.1, requires_grad=True)
b = qml.numpy.array(0.2, requires_grad=True)
c = qml.numpy.array(0.3, requires_grad=True)

res = param_shift_experimental(param_shift_experimental(cancel_inverses(circuit)))(a, b, c)
