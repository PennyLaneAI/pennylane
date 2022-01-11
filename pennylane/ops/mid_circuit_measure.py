import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import AnyWires, Operation


class MidCircuitMeasure(Operation):
    num_wires = 1

    def __init__(self, *args, **kwargs):
        self.runtime_value = None
        super().__init__(*args, **kwargs)


class RuntimeOp(Operation):
    num_wires = AnyWires

    def __init__(self, op, *args, **kwargs):
        self.op = op
        self.args = args
        self.kwargs = kwargs
        super().__init__(op, *args, **kwargs)

    def create_op(self):
        return self.op(*self.unwrap_args(self.args), do_queue=False, **self.kwargs)


    @staticmethod
    def unwrap_args(args):
        unwrapped = []
        for arg in args:
            if isinstance(arg, MidCircuitMeasure):
                unwrapped.append(arg.runtime_value)
            else:
                unwrapped.append(arg)
        return unwrapped



class RuntimeFunc(Operation):
    num_wires = None

    def __init__(self, fun, args, kwargs):
        self._fun = fun
        self._args = args
        self._kwargs = kwargs

class Eval(Operation):

    def __init__(self, measured, fun):
        self._measured = measured
        self._fun = fun
        super().__init__()

