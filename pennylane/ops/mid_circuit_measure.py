import uuid

from pennylane.operation import AnyWires, Operation

def Measure(wire):
    name = str(uuid.uuid4())[:8]  # might need to use more characters
    _MidCircuitMeasure(name, wire)
    return PossibleOutcomes(name)

class RuntimeOp(Operation):
    num_wires = AnyWires

    def __init__(self, op, *args, wires=None, **kwargs):
        self.op = op
        self.unknown_ops = apply_to_outcome(
            lambda *unwrapped: self.op(*unwrapped, do_queue=False, wires=wires, **kwargs)
        )(*args)
        super().__init__(wires=wires)


class If(Operation):
    num_wires = AnyWires

    def __init__(self, runtime_exp, then_op, *args, **kwargs):
        self.runtime_exp = runtime_exp
        self.then_op = then_op(*args, do_queue=False, **kwargs)
        super().__init__(*args, **kwargs)


class _MidCircuitMeasure(Operation):
    num_wires = 1

    def __init__(self, measure_var, wires=None):
        self.measure_var = measure_var
        self.runtime_value = None
        super().__init__(wires=wires)

def apply_to_outcome(fun):
    def wrapper(*args, **kwargs):
        partial = OutcomeValue()
        for arg in args:
            partial = partial._merge(arg)
        return partial._transform_leaves(lambda *unwrapped: fun(*unwrapped, **kwargs))
    return wrapper

class OutcomeValue:

    def __init__(self, *args):
        self.values = args

    def _merge(self, other):
        if isinstance(other, PossibleOutcomes):
            new_node = PossibleOutcomes(None)
            new_node.measurement_id = other.measurement_id
            new_node.zero = self._merge(other.zero)
            new_node.one = self._merge(other.one)
            return new_node
        elif isinstance(other, OutcomeValue):
            return OutcomeValue(*self.values, *other.values)
        else:
            return OutcomeValue(*self.values, other)

    def _transform_leaves(self, fun):
        return OutcomeValue(fun(*self.values))

    def get_computation(self, runtime_measurements):
        if len(self.values) == 1:
            return self.values[0]
        return self.values

    def _str_builder(self):
        return [f"=> {', '.join(str(v) for v in self.values)}"]


class PossibleOutcomes:

    def __init__(self, name):
        self.zero = OutcomeValue(0)
        self.one = OutcomeValue(1)
        self.measurement_id = name

    def __add__(self, other):
        return apply_to_outcome(lambda x, y: x + y)(self, other)

    def __radd__(self, other):
        return apply_to_outcome(lambda x, y: y + x)(self, other)

    def __mul__(self, other):
        return apply_to_outcome(lambda x, y: x*y)(self, other)

    def __rmul__(self, other):
        return apply_to_outcome(lambda x, y: y*x)(self, other)

    def _str_builder(self):
        build = []
        if isinstance(self.zero, PossibleOutcomes):
            for v in self.zero._str_builder():
                build.append(f"{self.measurement_id}=0,{v}")
            for v in self.one._str_builder():
                build.append(f"{self.measurement_id}=1,{v}")
        else:
            for v in self.zero._str_builder():
                build.append(f"{self.measurement_id}=0 {v}")
            for v in self.one._str_builder():
                build.append(f"{self.measurement_id}=1 {v}")
        return build

    def __str__(self):
        return "\n".join(self._str_builder())


    def _merge(self, other):
        if isinstance(other, PossibleOutcomes):
            new_node = PossibleOutcomes(None)
            if self.measurement_id == other.measurement_id:
                new_node.measurement_id = self.measurement_id
                new_node.zero = self.zero._merge(other.zero)
                new_node.one = self.one._merge(other.one)
            elif self.measurement_id < other.measurement_id:
                new_node.measurement_id = self.measurement_id
                new_node.zero = self.zero._merge(other)
                new_node.one = self.one._merge(other)
            elif self.measurement_id > other.measurement_id:
                new_node.measurement_id = other.measurement_id
                new_node.zero = self._merge(other.zero)
                new_node.one = self._merge(other.one)
            return new_node
        elif isinstance(other, OutcomeValue):
            new_node = PossibleOutcomes(None)
            new_node.measurement_id = self.measurement_id
            new_node.zero = self.zero._merge(other)
            new_node.one = self.one._merge(other)
            return new_node
        else:
            leaf = OutcomeValue(other)
            new_node = PossibleOutcomes(None)
            new_node.measurement_id = self.measurement_id
            new_node.zero = self.zero._merge(leaf)
            new_node.one = self.one._merge(leaf)
            return new_node

    def _transform_leaves(self, fun):
        new_node = PossibleOutcomes(self.measurement_id)
        new_node.zero = self.zero._transform_leaves(fun)
        new_node.one = self.one._transform_leaves(fun)
        return new_node

    def get_computation(self, runtime_measurements):
        if self.measurement_id in runtime_measurements:
            result = runtime_measurements[self.measurement_id]
            if result == 0:
                return self.zero.get_computation(runtime_measurements)
            else:
                return self.one.get_computation(runtime_measurements)




