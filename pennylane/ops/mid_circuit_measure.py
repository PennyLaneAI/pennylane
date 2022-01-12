import sympy as sp

from pennylane.operation import AnyWires, Operation

class _MeasureClass:

    def __init__(self):
        self._count = 0

    def _get_count(self):
        count = self._count
        self._count += 1
        return count

    def __call__(self, wire):
        runtime_var = sp.Symbol(f"measurement_{self._get_count()}")
        MidCircuitMeasure(runtime_var, wires=wire)
        return runtime_var

Measure = _MeasureClass()


class MidCircuitMeasure(Operation):
    num_wires = 1

    def __init__(self, measure_var, wires=None):
        self.measure_var = measure_var
        self.runtime_value = None
        super().__init__(wires=wires)


class If(Operation):
    num_wires = AnyWires

    def __init__(self, runtime_exp, then_op, *args, **kwargs):
        self.runtime_exp = runtime_exp
        self.then_op = then_op(*args, do_queue=False, **kwargs)
        super().__init__(*args, **kwargs)


