import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", shots = 1)

from functools import partial

qml.decomposition.enable_graph()

from pennylane.templates.subroutines.semi_adder import _semiadder_log_depth
@partial(qml.transforms.decompose, fixed_decomps = {qml.SemiAdder: _semiadder_log_depth}, max_expansion = 1)
@qml.qnode(dev)
def circuit(x_wires, y_wires):
    #qml.BasisState(64, wires=x_wires)
    #qml.BasisState(32 + 11, wires=y_wires)
    qml.adjoint(qml.ctrl(qml.adjoint(qml.X(0)), control = [1,2]))
    #qml.SemiAdder(x_wires, y_wires, None)

    return qml.sample(wires = [3,4,5])

print(circuit([0,1,2], [3,4,5]))

print(qml.draw(circuit)(1,1))
#qml.draw_mpl(circuit)([0,1,2], [3,4,5])
#plt.show()
