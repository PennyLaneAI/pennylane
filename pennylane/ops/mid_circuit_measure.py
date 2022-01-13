import uuid

from pennylane.operation import AnyWires, Operation

def Measure(wire):
    name = uuid.uuid4()
    _MidCircuitMeasure(name, wire)
    return PossibleOutcomes(name)

class _MidCircuitMeasure(Operation):
    num_wires = 1

    def __init__(self, measure_var, wires=None):
        self.measure_var = measure_var
        self.runtime_value = None
        super().__init__(wires=wires)


class OutcomeValue:

    def __init__(self, *args):
        self.values = args

    def __add__(self, other):
        if isinstance(other, PossibleOutcomes):
            return other.__radd__(self)
        if not isinstance(other, OutcomeValue):
            other = OutcomeValue(other)
        return OutcomeValue(*self.values, *other.values)

    def __radd__(self, other):
        if isinstance(other, PossibleOutcomes):
            return other.__add__(self)
        if not isinstance(other, OutcomeValue):
            other = OutcomeValue([other])
        return OutcomeValue(*other.values, *self.values)

    def transform_leaves(self, fun):
        return OutcomeValue(fun(*self.values))

    def get_computation(self, runtime_measurements):
        if len(self.values) == 1:
            return self.values[0]
        return self.values


class PossibleOutcomes:

    def __init__(self, name):
        self.zero = OutcomeValue(0)
        self.one = OutcomeValue(1)
        self.measurement_id = name

    def __add__(self, other):
        new_node = PossibleOutcomes(None)
        if isinstance(other, OutcomeValue):
            new_node.measurement_id = self.measurement_id
            new_node.zero = self.zero.__add__(other)
            new_node.one = self.one.__add__(other)
        elif not isinstance(other, PossibleOutcomes):
            leaf = OutcomeValue(other)
            new_node.measurement_id = self.measurement_id
            new_node.zero = self.zero.__add__(leaf)
            new_node.one = self.one.__add__(leaf)
        elif self.measurement_id == other.measurement_id:
            new_node.measurement_id = self.measurement_id
            new_node.zero = self.zero.__add__(other.zero)
            new_node.one = self.one.__add__(other.zero)
        elif self.measurement_id < other.measurement_id:
            new_node.measurement_id = self.measurement_id
            new_node.zero = other.__radd__(self.zero)
            new_node.one = other.__radd__(self.one)
        elif self.measurement_id > other.measurement_id:
            new_node.measurement_id = other.measurement_id
            new_node.zero = self.__add__(other.zero)
            new_node.one = self.__add__(other.one)
        return new_node

    def __radd__(self, other):
        new_node = PossibleOutcomes(None)
        if isinstance(other, OutcomeValue):
            new_node.measurement_id = self.measurement_id
            new_node.zero = self.zero.__radd__(other)
            new_node.one = self.one.__radd__(other)
        elif not isinstance(other, PossibleOutcomes):
            leaf = OutcomeValue([other])
            new_node.measurement_id = self.measurement_id
            new_node.zero = self.zero.__radd__(leaf)
            new_node.one = self.one.__radd__(leaf)
        elif self.measurement_id == other.measurement_id:
            new_node.measurement_id = self.measurement_id
            new_node.zero = other.zero.__radd__(self.zero)
            new_node.one = other.zero.__radd__(self.one)
        elif self.measurement_id < other.measurement_id:
            new_node.measurement_id = self.measurement_id
            new_node.zero = other.__add__(self.zero)
            new_node.one = other.__add__(self.one)
        elif self.measurement_id > other.measurement_id:
            new_node.measurement_id = other.measurement_id
            new_node.zero = self.__radd__(other.zero)
            new_node.one = self.__radd__(other.one)
        return new_node

    def _transform_leaves(self, fun):
        new_node = PossibleOutcomes(self.measurement_id)
        new_node.zero = self.zero.transform_leaves(fun)
        new_node.one = self.one.transform_leaves(fun)
        return new_node

    @classmethod
    def apply_to_all(cls, fun):
        def wrapper(*args, **kwargs):
            partial = OutcomeValue()
            for arg in args:
                partial = partial + arg
            return partial._transform_leaves(lambda *unwrapped: fun(*unwrapped, **kwargs))
        return wrapper

    def get_computation(self, runtime_measurements):
        if self.measurement_id in runtime_measurements:
            result = runtime_measurements[self.measurement_id]
            if result == 0:
                return self.zero.get_computation(runtime_measurements)
            else:
                return self.one.get_computation(runtime_measurements)

class RuntimeOp(Operation):
    num_wires = AnyWires

    def __init__(self, op, *args, wires=None, **kwargs):
        self.op = op
        self.unknown_ops = PossibleOutcomes.apply_to_all(
            lambda *unwrapped: self.op(*unwrapped, do_queue=False, wires=wires, **kwargs)
        )(*args)
        super().__init__(wires=wires)


class If(Operation):
    num_wires = AnyWires

    def __init__(self, runtime_exp, then_op, *args, **kwargs):
        self.runtime_exp = runtime_exp
        self.then_op = then_op(*args, do_queue=False, **kwargs)
        super().__init__(*args, **kwargs)




