import pennylane as qml
import numpy as np
from pennylane.templates.subroutines.select import _unary_select
import matplotlib.pyplot as plt



dev = qml.device('default.qubit')
ops = [qml.BasisEmbedding(1, wires = range(9, 11)) for i in range(25)]
@qml.qnode(dev)
def circuit():
     qml.BasisEmbedding(1, wires = range(5))
     _unary_select(ops, control=[0,1,2,3,4], work_wires=[5,6,7,8])
     return qml.probs(range(9,11))

#print(circuit())

qml.draw_mpl(circuit)()
plt.show()