import pennylane as qml
import numpy as np
from pennylane.templates.subroutines.select import _unary_select
import matplotlib.pyplot as plt


for n in range(11):

    dev = qml.device('default.qubit')
    ops = [qml.BasisEmbedding(i, wires = range(7, 7+4)) for i in range(11)]
    @qml.qnode(dev)
    def circuit():
         qml.BasisEmbedding(n, wires = range(4))
         _unary_select(ops, control=[0,1,2,3], work_wires=[4,5,6])
         return qml.probs(range(7, 7+4))

    print(circuit())