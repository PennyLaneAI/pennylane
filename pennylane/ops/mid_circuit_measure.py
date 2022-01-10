import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import AnyWires, Operation


class MidCircuitMeasure(Operation):
    num_wires = 1



class Eval(Operation):

    def __init__(self, measured, fun):
        self._measured = measured
        self._fun = fun
        super().__init__()

