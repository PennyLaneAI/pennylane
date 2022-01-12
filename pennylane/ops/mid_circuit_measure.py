from sympy import Symbol
# from sympy.core import Symbol
# from ast import literal_eval


from pennylane.operation import AnyWires, Operation


# class RuntimeResult(Symbol):
#     __slots__ = tuple()
#
#     count = 0
#
#     def __new__(cls, name):
#         # name = str(hash(name)) + "_measured" + "_"
#         obj = super().__new__(Symbol, name)
#         obj.__class__ = cls
#         return obj

# def Measure(wire):
#     # runtime_var = RuntimeResult(wire)
#     runtime_var = Symbol(f"measurement_{Measure._count}")
#     Measure._count += 1
#     MidCircuitMeasure(runtime_var, wires=wire)
#     return runtime_var
#
# Measure._count = 0

class MeasureClass:

    def __init__(self):
        self._count = 0

    def _get_count(self):
        count = self._count
        self._count += 1
        return count

    def __call__(self, wire):
        runtime_var = Symbol(f"measurement_{self._get_count()}")
        MidCircuitMeasure(runtime_var, wires=wire)
        return runtime_var

Measure = MeasureClass()




class MidCircuitMeasure(Operation):
    num_wires = 1

    def __init__(self, measure_var, wires=None):
        self.measure_var = measure_var
        self.runtime_value = None
        super().__init__(wires=wires)


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


class IfOp(Operation):
    num_wires = AnyWires

    def __init__(self, runtime_exp, then_op, *args, **kwargs):
        self.runtime_exp = runtime_exp
        self.then_op = then_op(*args, do_queue=False, **kwargs)
        super().__init__(runtime_exp, then_op, *args, **kwargs)


# class RuntimeFunc(Operation):
#     num_wires = None
#
#     def __init__(self, fun, args, kwargs):
#         self._fun = fun
#         self._args = args
#         self._kwargs = kwargs

class Eval(Operation):

    def __init__(self, measured, fun):
        self._measured = measured
        self._fun = fun
        super().__init__()

