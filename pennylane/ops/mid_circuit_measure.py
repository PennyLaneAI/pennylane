import uuid

from typing import Union, Any, Dict

from pennylane.operation import AnyWires, Operation

def Measure(wire):
    """
    Create a mid-circuit measurement and return an outcome.

    m0 = qml.Measure(0)
    """
    name = str(uuid.uuid4())[:8]  # might need to use more characters
    _MidCircuitMeasure(name, wire)
    return MeasurementDependantValue(name)

class RuntimeOp(Operation):
    """
    Run an operation with parameters being outcomes of mid-circuit measurements.

    ex:

    m0 = qml.Measure(0)
    qml.RuntimeOp(qml.RZ, m0, wires=1)
    """
    num_wires = AnyWires

    def __init__(self, op, *args, wires=None, **kwargs):
        self.op = op
        self.unknown_ops = apply_to_outcome(
            lambda *unwrapped: self.op(*unwrapped, do_queue=False, wires=wires, **kwargs)
        )(*args)
        super().__init__(wires=wires)


class If(Operation):
    """
    Run an operation conditionally on the outcome of mid-circuit measurements.

    ex:

    m0 = qml.Measure(0)
    qml.If(m0, qml.RZ, 1.2, wires=1)
    """
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
    """
    Apply an arbitrary function to a `MeasurementDependantValue` or set of `MeasurementDependantValue`s.

    ex:
    m0 = qml.Measure(0)
    m0_sin = qml.apply_to_outcome(np.sin)(m0)
    """
    def wrapper(*args, **kwargs):
        partial = _Value()
        for arg in args:
            partial = partial._merge(arg)
        return partial._transform_leaves(lambda *unwrapped: fun(*unwrapped, **kwargs))
    return wrapper


class MeasurementDependantValue:
    """
    A class representing unknown measurement outcomes. Since we don't know the actual outcomes at circuit creation time,
    consider all scenarios.

    supports python __dunder__ mathematical operations. as well as qml.apply_to_outcome to perform arbitrary function.
    """

    def __init__(self, measurement_id: str):
        self.zero_case: Union[MeasurementDependantValue, _Value] = _Value(0)
        self.one_case: Union[MeasurementDependantValue, _Value] = _Value(1)
        self.dependent_on: str = measurement_id

    def __add__(self, other: Any):
        return apply_to_outcome(lambda x, y: x + y)(self, other)

    def __radd__(self, other: Any):
        return apply_to_outcome(lambda x, y: y + x)(self, other)

    def __mul__(self, other: Any):
        return apply_to_outcome(lambda x, y: x*y)(self, other)

    def __rmul__(self, other: Any):
        return apply_to_outcome(lambda x, y: y*x)(self, other)

    def _str_builder(self):
        build = []
        if isinstance(self.zero_case, MeasurementDependantValue):
            for v in self.zero_case._str_builder():
                build.append(f"{self.dependent_on}=0,{v}")
            for v in self.one_case._str_builder():
                build.append(f"{self.dependent_on}=1,{v}")
        else:
            for v in self.zero_case._str_builder():
                build.append(f"{self.dependent_on}=0 {v}")
            for v in self.one_case._str_builder():
                build.append(f"{self.dependent_on}=1 {v}")
        return build

    def __str__(self):
        return "\n".join(self._str_builder())


    def _merge(self, other: Union["MeasurementDependantValue", "_Value", Any]):
        """
        Merge this MeasurementDependantValue with `other`.

        Ex: Merging a MeasurementDependantValue such as:

        df3jff4t:0 => 3.4
        df3jff4t:1 => 1

        with another MeasurementDependantValue:

        f93fjdj3=0 => 100
        f93fjdj3=1 => 67

        will result in:

        df3jff4t=0,f93fjdj3=0 => 3.4,100
        df3jff4t=0,f93fjdj3=1 => 3.4,67
        df3jff4t=1,f93fjdj3=0 => 1,100
        df3jff4t=1,f93fjdj3=1 => 1,67

        """
        if isinstance(other, MeasurementDependantValue):
            new_node = MeasurementDependantValue(None)
            if self.dependent_on == other.dependent_on:
                new_node.dependent_on = self.dependent_on
                new_node.zero_case = self.zero_case._merge(other.zero_case)
                new_node.one_case = self.one_case._merge(other.one_case)
            elif self.dependent_on < other.dependent_on:
                new_node.dependent_on = self.dependent_on
                new_node.zero_case = self.zero_case._merge(other)
                new_node.one_case = self.one_case._merge(other)
            elif self.dependent_on > other.dependent_on:
                new_node.dependent_on = other.dependent_on
                new_node.zero_case = self._merge(other.zero_case)
                new_node.one_case = self._merge(other.one_case)
            return new_node
        elif isinstance(other, _Value):
            new_node = MeasurementDependantValue(None)
            new_node.dependent_on = self.dependent_on
            new_node.zero_case = self.zero_case._merge(other)
            new_node.one_case = self.one_case._merge(other)
            return new_node
        else:
            leaf = _Value(other)
            new_node = MeasurementDependantValue(None)
            new_node.dependent_on = self.dependent_on
            new_node.zero_case = self.zero_case._merge(leaf)
            new_node.one_case = self.one_case._merge(leaf)
            return new_node

    def _transform_leaves(self, fun: callable):
        """
        Transform the leaves of a MeasurementDependantValue with `fun`.
        """
        new_node = MeasurementDependantValue(self.dependent_on)
        new_node.zero_case = self.zero_case._transform_leaves(fun)
        new_node.one_case = self.one_case._transform_leaves(fun)
        return new_node

    def get_computation(self, runtime_measurements: Dict[str, int]):
        """
        Given a list of measurement outcomes get the correct computation.
        """
        if self.dependent_on in runtime_measurements:
            result = runtime_measurements[self.dependent_on]
            if result == 0:
                return self.zero_case.get_computation(runtime_measurements)
            else:
                return self.one_case.get_computation(runtime_measurements)

class _Value:
    """
    Leaf node for a MeasurementDependantValue tree structure.
    """
    def __init__(self, *args):
        self.values = args

    def _merge(self, other):
        if isinstance(other, MeasurementDependantValue):
            new_node = MeasurementDependantValue(None)
            new_node.dependent_on = other.dependent_on
            new_node.zero_case = self._merge(other.zero_case)
            new_node.one_case = self._merge(other.one_case)
            return new_node
        elif isinstance(other, _Value):
            return _Value(*self.values, *other.values)
        else:
            return _Value(*self.values, other)

    def _transform_leaves(self, fun):
        return _Value(fun(*self.values))

    def get_computation(self, runtime_measurements):
        if len(self.values) == 1:
            return self.values[0]
        return self.values

    def _str_builder(self):
        return [f"=> {', '.join(str(v) for v in self.values)}"]