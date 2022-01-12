import uuid
from functools import reduce

from pennylane.operation import AnyWires, Operation

def Measure(wire):
    name = uuid.uuid4()
    MidCircuitMeasure(uuid.uuid4(), wire)
    # return ScenarioValue(name)

class MidCircuitMeasure(Operation):
    num_wires = 1

    def __init__(self, measure_var, wires=None):
        self.measure_var = measure_var
        self.runtime_value = None
        super().__init__(wires=wires)


class Outcome:

    @classmethod
    def measure(cls, name):
        m = cls()
        m.zero = 0
        m.one = 1
        m.value = name
        m.leaf = False
        return m


    def __init__(self):
        self.one = None
        self.zero = None
        self.value = None
        self.leaf = True

    def __radd__(self, other):
        return other + self

    def __add__(self, other):

        if not isinstance(other, Outcome):
            wrap = Outcome()
            wrap.value = [other]
            other = wrap

        new_node = Outcome()
        if self.leaf and other.leaf:
            new_node.value = [*self.value, *other.value]
        elif self.leaf and not other.leaf:
            new_node.value = other.value
            new_node.zero = self + other.zero
            new_node.one = self + other.one
            new_node.leaf = False
        elif not self.leaf and other.leaf:
            new_node.value = self.value
            new_node.zero = other + self.zero
            new_node.one = other + self.one
            new_node.leaf = False
        elif self.value == other.value:
            new_node.value = self.value
            new_node.zero = self.zero
            new_node.one = self.one
            new_node.leaf = False
        elif self.value < other.value:
            new_node.value = self.value
            new_node.zero = other + self.zero
            new_node.one = other + self.one
            new_node.leaf = False
        elif self.value > other.value:
            new_node.value = other.value
            new_node.zero = self + other.zero
            new_node.one = self + other.one
            new_node.leaf = False
        return new_node

    def apply(self, fun):
        def wrapper(*args, **kwargs):






class If(Operation):
    num_wires = AnyWires

    def __init__(self, runtime_exp, then_op, *args, **kwargs):
        self.runtime_exp = runtime_exp
        self.then_op = then_op(*args, do_queue=False, **kwargs)
        super().__init__(*args, **kwargs)


