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

# class ScenarioValue:
#
#     def __init__(self):
#         self.scenario = frozenset()
#         self.value = None
#
# class Possibilities:
#
#     def __init__(self):

class Node:
    def __init__(self):
        self.one = None
        self.zero = None
        self.value = None
        self.leaf = True

    def __add__(self, other):

        if not isinstance(other, Node):
            wrap = Node()
            wrap.value = other
            other = wrap

        new_node = Node()
        if self.leaf and other.leaf:
            new_node.value = [self.value, other.value]
        elif self.leaf and not other.leaf:
            new_node.value = other.value
            new_node.zero = self + other.zero
            new_node.one = self + other.one
            new_node.leaf = False
        elif not self.leaf and other.leaf:
            new_node.value = self.value
            new_node.zero = self.zero + other
            new_node.one = self.one + other
            new_node.leaf = False
        elif self.value == other.value:
            new_node.value = self.value
            new_node.zero = self.zero + other.zero
            new_node.one = self.one + other.one
            new_node.leaf = False
        elif self.value < other.value:
            new_node.value = self.value
            new_node.zero = self.zero + other
            new_node.one = self.one + other
            new_node.leaf = False
        elif self.value > other.value:
            new_node.value = other.value
            new_node.zero = other.zero + self
            new_node.one = other.one + self
            new_node.leaf = False
        return new_node

class Outcome:

    def __init__(self):
        self.versions = Node()
        self.depends_on = []

    def __add__(self, other):
        depends_on = sorted(set(self.depends_on) | set(other.depends_on))
        for measure in depends_on:
            if

    @classmethod
    def measurement_outcome(cls, name):
        outcome = cls()
        outcome.possible = {
            frozenset([(name, 0)]): 0,
            frozenset([(name, 1)]): 1
        }
        outcome.depends_on = frozenset([name])
        return outcome

    # @classmethod
    # def apply(cls, fun):
    #
    #     def wrapper(*args, **kwargs):
    #         measurements = set(reduce(lambda x, y: x.depends_on | y.depends_on, filter(lambda arg: isinstance(arg, cls), args)))
    #         def scenarios(ok):
    #             if not ok:
    #                 return frozenset()
    #             for measurement in ok:
    #                 exclude = ok.remove(measurement)
    #                 remaining = scenarios(exclude)
    #                 yield frozenset((measurement, 0)) | remaining
    #                 yield frozenset((measurement, 1)) | remaining
    #
    #         for scenario in scenarios(measurements):
    #             print(scenario)
    #     return wrapper



    @classmethod
    def apply(cls, func, scenario=frozenset()):
        def wrapper(*args, **kwargs):

            if not any([isinstance(arg, cls) for arg in args]):
                return func(*args, **kwargs)

            for i, arg in enumerate(args):
                if isinstance(arg, cls):
                    out = cls()
                    partial_possible = {}
                    partial_depends_on = set(arg.depends_on)
                    for name, value in arg.possible.items():
                        outcome_args = list(args)
                        outcome_args[i] = value
                        result = cls.apply(func)(*outcome_args, **kwargs)
                        if isinstance(result, cls):
                            for result_name, result_value in result.possible.items():
                                partial_possible[frozenset(set(name) | set(result_name))] = result_value
                            partial_depends_on |= result.depends_on
                        else:
                            partial_possible[name] = result
                    out.possible = partial_possible
                    out.depends_on = partial_depends_on
                    return out
        return wrapper


m1 = Outcome.measurement_outcome("what")
ok = Outcome.apply(lambda x, y: x * y)(m1, m1)

print(ok)





class If(Operation):
    num_wires = AnyWires

    def __init__(self, runtime_exp, then_op, *args, **kwargs):
        self.runtime_exp = runtime_exp
        self.then_op = then_op(*args, do_queue=False, **kwargs)
        super().__init__(*args, **kwargs)


