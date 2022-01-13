import pennylane as qml
import pennylane.numpy as np

"""Tests the continuous variable based operations."""

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def runtime_circuit():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)

    m0 = qml.Measure(0)
    print(m0)
    m1 = qml.Measure(1)
    m2 = qml.Measure(2)

    @qml.apply_to_outcome
    def fun_1(x, y):
        return x * y

    out1 = fun_1(m0, m1)

    rot = qml.apply_to_outcome(lambda x, y, z: np.sin(x) + y + z)(out1, m1, m2)

    qml.RuntimeOp(qml.RZ, rot, wires=1)

    return qml.probs(wires=1)

value = runtime_circuit()