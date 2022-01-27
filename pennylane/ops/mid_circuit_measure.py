"""
Mid-circuit measurements and associated operations.
"""

import uuid
import functools

from typing import Union, Any, Dict, TypeVar, Generic, Callable, Type

from pennylane.operation import Operation


def mid_measure(wire):
    """
    Create a mid-circuit measurement and return an outcome.

    .. code-block:: python

        m0 = qml.Measure(0)
    """
    wire_id = wire
    _MidCircuitMeasure(wire_id, wire)
    return MeasurementDependantValue(wire_id, 0, 1)


class _MidCircuitMeasure(Operation):
    """
    Operation to perform mid-circuit measurement.
    """

    num_wires = 1

    def __init__(self, measure_var, wires=None):
        self.measure_var = measure_var
        self.runtime_value = None
        super().__init__(wires=wires)


def apply_to_measurement_dependant_values(fun):
    """
    Apply an arbitrary function to a `MeasurementDependantValue` or set of `MeasurementDependantValue`s.

    Ex:

    .. code-block:: python

        m0 = qml.mid_measure(0)
        m0_sin = qml.apply_to_outcome(np.sin)(m0)
    """

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        partial = _Value()
        for arg in args:
            partial = partial._merge(arg)
        partial._transform_leaves_inplace(
            lambda *unwrapped: fun(*unwrapped, **kwargs)
        )
        return partial

    return wrapper


T = TypeVar("T")


# pylint: disable=protected-access
class MeasurementDependantValue(Generic[T]):
    """
    A class representing unknown measurement outcomes.
    Since we don't know the actual outcomes at circuit creation time,
    consider all scenarios.

    supports python __dunder__ mathematical operations. As well as arbitrary functions using qml.apply_to_outcome
    """

    __slots__ = ("_dependent_on", "_zero_case", "_one_case")

    def __init__(
        self,
        measurement_id: str,
        zero_case: Union["MeasurementDependantValue[T]", "_Value[T]", T],
        one_case: Union["MeasurementDependantValue[T]", "_Value[T]", T],
    ):
        self._dependent_on = measurement_id
        if isinstance(zero_case, (MeasurementDependantValue, _Value)):
            self._zero_case = zero_case
        else:
            self._zero_case = _Value(zero_case)
        if isinstance(one_case, (MeasurementDependantValue, _Value)):
            self._one_case = one_case
        else:
            self._one_case = _Value(one_case)

    @property
    def branches(self):
        branch_dict = {}
        if isinstance(self._zero_case, MeasurementDependantValue):
            for k, v in self._zero_case.branches.items():
                branch_dict[(0, *k)] = v
            for k, v in self._one_case.branches.items():
                branch_dict[(1, *k)] = v
        else:
            branch_dict[(0,)] = self._zero_case.values
            branch_dict[(1,)] = self._one_case.values
        return branch_dict

    @property
    def measurements(self):
        if isinstance(self._zero_case, MeasurementDependantValue):
            return [self._dependent_on, *self._zero_case.measurements]
        return [self._dependent_on]

    # define all mathematical __dunder__ methods https://docs.python.org/3/library/operator.html
    def __add__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: x + y)(self, other)

    def __radd__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: y + x)(self, other)

    def __mul__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: x * y)(self, other)

    def __rmul__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: y * x)(self, other)

    def __str__(self):
        measurements = self.measurements
        lines = []
        for k, v in self.branches.items():
            lines.append(",".join([f"{measurements[i]}={k[i]}" for i in range(len(measurements))]) + " => " + str(v))
        return "\n".join(lines)


    def _merge(self, other: Union["MeasurementDependantValue", "_Value", Any]):
        """
        Merge this MeasurementDependantValue with `other`.

        Ex: Merging a MeasurementDependantValue such as

        .. code-block:: python

            df3jff4t=0 => 3.4
            df3jff4t=1 => 1

        with another MeasurementDependantValue:

        .. code-block:: python

            f93fjdj3=0 => 100
            f93fjdj3=1 => 67

        will result in:

        .. code-block:: python

            df3jff4t=0,f93fjdj3=0 => 3.4,100
            df3jff4t=0,f93fjdj3=1 => 3.4,67
            df3jff4t=1,f93fjdj3=0 => 1,100
            df3jff4t=1,f93fjdj3=1 => 1,67

        (note the uuids in the example represent distinct measurements of different qubit.)

        """
        if isinstance(other, MeasurementDependantValue):
            if self._dependent_on == other._dependent_on:
                return MeasurementDependantValue(
                    self._dependent_on,
                    self._zero_case._merge(other._zero_case),
                    self._one_case._merge(other._one_case),
                )
            if self._dependent_on < other._dependent_on:
                return MeasurementDependantValue(
                    self._dependent_on,
                    self._zero_case._merge(other),
                    self._one_case._merge(other),
                )
            return MeasurementDependantValue(
                other._dependent_on,
                self._merge(other._zero_case),
                self._merge(other._one_case),
            )
        return MeasurementDependantValue(
            self._dependent_on,
            self._zero_case._merge(other),
            self._one_case._merge(other),
        )

    def _transform_leaves_inplace(self, fun: Callable):
        """
        Transform the leaves of a MeasurementDependantValue with `fun`.
        """
        self._zero_case._transform_leaves_inplace(fun)
        self._one_case._transform_leaves_inplace(fun)


# pylint: disable=too-few-public-methods,protected-access
class _Value(Generic[T]):
    """
    Leaf node for a MeasurementDependantValue tree structure.
    """

    __slots__ = ("_values",)

    def __init__(self, *values):
        self._values = values

    def _merge(self, other):
        """
        Works with MeasurementDependantValue._merge
        """
        if isinstance(other, MeasurementDependantValue):
            return MeasurementDependantValue(
                other._dependent_on, self._merge(other._zero_case), self._merge(other._one_case)
            )
        if isinstance(other, _Value):
            return _Value(*self._values, *other._values)
        return _Value(*self._values, other)

    def _transform_leaves_inplace(self, fun):
        """
        Works with MeasurementDependantValue._transform_leaves
        """
        self._values = (fun(*self._values),)

    @property
    def values(self):
        if len(self._values) == 1:
            return self._values[0]
        return self._values


def if_then(expr: MeasurementDependantValue[bool], then_op: Type[Operation]):
    """
    Run an operation conditionally on the outcome of mid-circuit measurements.

    .. code-block:: python

        m0 = qml.mid_measure(0)
        qml.if_then(m0, qml.RZ)(1.2, wires=1)
    """

    class _IfOp(Operation):
        """
        Helper private class for `if_then` function.
        """

        num_wires = then_op.num_wires
        op: Type[Operation] = then_op
        branches = expr.branches
        required_measurements = expr.measurements

        def __init__(self, *args, **kwargs):
            self.then_op = then_op(*args, do_queue=False, **kwargs)
            super().__init__(*args, **kwargs)

    return _IfOp


def condition(condition_op: Type[Operation]):
    """
    Run an operation with parameters being outcomes of mid-circuit measurements.

    .. code-block:: python

        m0 = qml.mid_measure(0)
        qml.condition(qml.RZ)(m0, wires=1)
    """

    class _ConditionOp(Operation):
        """
        Helper private class for `condition` function.
        """

        num_wires = condition_op.num_wires
        op: Type[Operation] = condition_op

        def __init__(self, *args, **kwargs):
            measurement_dependant_op = apply_to_measurement_dependant_values(
                lambda *unwrapped: self.op(*unwrapped, do_queue=False, **kwargs)
            )(
                *args
            )
            self.branches = measurement_dependant_op.branches
            self.dependant_measurements = measurement_dependant_op.measurements
            super().__init__(*args, **kwargs)

    return _ConditionOp
