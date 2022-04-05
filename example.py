import pennylane as qml
from pennylane.script.ast import script


@script
def circuit(x, y):
    while y > 0 and x == 0:
        if x > 3:
            if y < 10:
                qml.RX(x + 10, wires=0)
    return qml.expval(qml.PauliX(wires=0))