import pennylane as qml
from pennylane.labs.templates import LeftQuantumIntegerComparator
dev = qml.device("lightning.qubit")


# x > y

# 0: <
# 1: <=
# 2: >=
# 3: >

@qml.qjit
@qml.qnode(dev, shots=1)
def circuit(a, b):
    op = 2
    qml.BasisState(a, wires=[0, 3, 6, 9])
    qml.BasisState(b, wires=[1, 4, 7, 10])
    LeftQuantumIntegerComparator([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], op)
    qml.CNOT([11, 12])
    qml.adjoint(
        lambda: LeftQuantumIntegerComparator([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], op)
    )()
    return qml.sample(wires=[12])


print(bool(circuit(3, 3)))