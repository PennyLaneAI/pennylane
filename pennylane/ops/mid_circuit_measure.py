import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import AnyWires, Operation


class MidCircuitMeasure(Operation):
    num_wires = 1

    def __init__(self, classical_var_name, *args, **kwargs):
        self.classical_var_name = classical_var_name
        super().__init__(classical_var_name, *args, **kwargs)


class RuntimeOp(Operation):
    num_wires = AnyWires

    def __init__(self, op, *params, **kwargs):
        self.op = op
        super().__init__(op, *params, **kwargs)


class Eval(Operation):

    def __init__(self, measured, fun):
        self._measured = measured
        self._fun = fun
        super().__init__()

