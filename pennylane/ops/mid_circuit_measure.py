import uuid

from pennylane.operation import AnyWires, Operation

def Measure(wire):
    name = uuid.uuid4()
    MidCircuitMeasure(uuid.uuid4(), wire)
    return Node(name)

class MidCircuitMeasure(Operation):
    num_wires = 1

    def __init__(self, measure_var, wires=None):
        self.measure_var = measure_var
        self.runtime_value = None
        super().__init__(wires=wires)


class Leaf:

    def __init__(self, *args):
        self.values = args

    def __add__(self, other):
        if not isinstance(other, Leaf):
            other = Leaf(other)
        return Leaf(*self.values, *other.values)

    def __radd__(self, other):
        if not isinstance(other, Leaf):
            other = Leaf([other])
        return Leaf(*other.values, *self.values)


class Node:

    def __init__(self, name):
        self.zero = Leaf(0)
        self.one = Leaf(1)
        self.name = name

    def __add__(self, other):
        new_node = Node(None)
        if isinstance(other, Leaf):
            new_node.name = self.name
            new_node.zero = self.zero.__add__(other)
            new_node.one = self.one.__add__(other)
        elif not isinstance(other, Node):
            leaf = Leaf(other)
            new_node.name = self.name
            new_node.zero = self.zero.__add__(leaf)
            new_node.one = self.one.__add__(leaf)
        elif self.name == other.name:
            new_node.name = self.name
            new_node.zero = self.zero.__add__(other.zero)
            new_node.one = self.one.__add__(other.zero)
        elif self.name < other.name:
            new_node.name = self.name
            new_node.zero = other.__radd__(self.zero)
            new_node.one = other.__radd__(self.one)
        elif self.name > other.name:
            new_node.name = other.name
            new_node.zero = self.__add__(other.zero)
            new_node.one = self.__add__(other.one)
        return new_node

    def __radd__(self, other):
        new_node = Node(None)
        if isinstance(other, Leaf):
            new_node.name = self.name
            new_node.zero = self.zero.__radd__(other)
            new_node.one = self.one.__radd__(other)
        elif not isinstance(other, Node):
            leaf = Leaf([other])
            new_node.name = self.name
            new_node.zero = self.zero.__radd__(leaf)
            new_node.one = self.one.__radd__(leaf)
        elif self.name == other.name:
            new_node.name = self.name
            new_node.zero = other.zero.__radd__(self.zero)
            new_node.one = other.zero.__radd__(self.one)
        elif self.name < other.name:
            new_node.name = self.name
            new_node.zero = other.__add__(self.zero)
            new_node.one = other.__add__(self.one)
        elif self.name > other.name:
            new_node.name = other.name
            new_node.zero = self.__radd__(other.zero)
            new_node.one = self.__radd__(other.one)
        return new_node

    def transform_leaves(self, fun):
        new_node = Node(self.name)
        if isinstance(self.zero, Leaf):
            new_node.zero = fun(self.zero)
            new_node.one = fun(self.one)
        else:
            new_node.zero = self.zero.transform_leaves(fun)
            new_node.one = self.one.transform_leaves(fun)


test, what = Node("test"), Node("what")

ok = test + what

print("new")



class If(Operation):
    num_wires = AnyWires

    def __init__(self, runtime_exp, then_op, *args, **kwargs):
        self.runtime_exp = runtime_exp
        self.then_op = then_op(*args, do_queue=False, **kwargs)
        super().__init__(*args, **kwargs)


