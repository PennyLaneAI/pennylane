import pennylane as qml
from pennylane.templates.subroutines.arithmetic.semi_adder import _controlled_semi_adder

dev = qml.device("default.qubit", shots = 1)

op = qml.SemiAdder([0,1,2], [3,4,5], [6,7])
@qml.qnode(dev)
def circuit():
    qml.BasisState(1, [0,1,2])
    qml.BasisState(1, [3,4,5])
    qml.X(8)
    _controlled_semi_adder(qml.SemiAdder([0,1,2], [3,4,5], [6,7]), 8, 0)
    return qml.sample(wires = [3,4,5])

print(circuit())