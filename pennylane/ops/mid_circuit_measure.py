from sympy import Symbol


from pennylane.operation import AnyWires, Operation

class _MeasureClass:

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

Measure = _MeasureClass()


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
        super().__init__(*args, **kwargs)

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


class If(Operation):
    num_wires = AnyWires

    def __init__(self, runtime_exp, then_op, *args, **kwargs):
        self.runtime_exp = runtime_exp
        self.then_op = then_op(*args, do_queue=False, **kwargs)
        super().__init__(*args, **kwargs)


