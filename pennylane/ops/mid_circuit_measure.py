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
    measurement_id = str(uuid.uuid4())[:8]  # might need to use more characters
    _MidCircuitMeasure(measurement_id, wire)
    return MeasurementDependantValue(measurement_id, 0, 1)


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
            partial = partial._merge(arg)  # pylint: disable=protected-access
        return partial._transform_leaves(  # pylint: disable=protected-access
            lambda *unwrapped: fun(*unwrapped, **kwargs)  # pylint: disable=unnecessary-lambda
        )

    return wrapper


T = TypeVar("T")


class MeasurementDependantValue(Generic[T]):
    """
    A class representing unknown measurement outcomes.
    Since we don't know the actual outcomes at circuit creation time,
    consider all scenarios.

    supports python __dunder__ mathematical operations. As well as arbitrary functions using qml.apply_to_outcome
    """

    def __init__(
        self,
        measurement_id: str,
        zero_case: Union["MeasurementDependantValue[T]", "_Value[T]", T],
        one_case: Union["MeasurementDependantValue[T]", "_Value[T]", T],
    ):
        self.dependent_on = measurement_id
        if isinstance(zero_case, (MeasurementDependantValue, _Value)):
            self.zero_case = zero_case
        else:
            self.zero_case = _Value(zero_case)
        if isinstance(one_case, (MeasurementDependantValue, _Value)):
            self.one_case = one_case
        else:
            self.one_case = _Value(one_case)

    # define all mathematical __dunder__ methods https://docs.python.org/3/library/operator.html
    def __add__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: x + y)(self, other)

    def __radd__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: y + x)(self, other)

    def __mul__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: x * y)(self, other)

    def __rmul__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: y * x)(self, other)

    def _str_builder(self):
        """
        Helper method for __str__
        """
        build = []
        if isinstance(self.zero_case, MeasurementDependantValue):
            for v in self.zero_case._str_builder():  # pylint: disable=protected-access
                build.append(f"{self.dependent_on}=0,{v}")
            for v in self.one_case._str_builder():  # pylint: disable=protected-access
                build.append(f"{self.dependent_on}=1,{v}")
        else:
            for v in self.zero_case._str_builder():  # pylint: disable=protected-access
                build.append(f"{self.dependent_on}=0 {v}")
            for v in self.one_case._str_builder():  # pylint: disable=protected-access
                build.append(f"{self.dependent_on}=1 {v}")
        return build

    def __str__(self):
        return "\n".join(self._str_builder())

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
            if self.dependent_on == other.dependent_on:
                return MeasurementDependantValue(
                    self.dependent_on,
                    self.zero_case._merge(other.zero_case),  # pylint: disable=protected-access
                    self.one_case._merge(other.one_case),  # pylint: disable=protected-access
                )
            if self.dependent_on < other.dependent_on:
                return MeasurementDependantValue(
                    self.dependent_on,
                    self.zero_case._merge(other),  # pylint: disable=protected-access
                    self.one_case._merge(other),  # pylint: disable=protected-access
                )
            return MeasurementDependantValue(
                other.dependent_on,
                self._merge(other.zero_case),  # pylint: disable=protected-access
                self._merge(other.one_case),  # pylint: disable=protected-access
            )
        return MeasurementDependantValue(
            self.dependent_on,
            self.zero_case._merge(other),  # pylint: disable=protected-access
            self.one_case._merge(other),  # pylint: disable=protected-access
        )

    def _transform_leaves(self, fun: Callable):
        """
        Transform the leaves of a MeasurementDependantValue with `fun`.
        """
        return MeasurementDependantValue(
            self.dependent_on,
            self.zero_case._transform_leaves(fun),  # pylint: disable=protected-access
            self.one_case._transform_leaves(fun),  # pylint: disable=protected-access
        )

    def get_computation(self, runtime_measurements: Dict[str, int]):
        """
        Given a list of measurement outcomes get the correct computation.
        """
        if self.dependent_on in runtime_measurements:
            result = runtime_measurements[self.dependent_on]
            if result == 0:
                return self.zero_case.get_computation(runtime_measurements)
            return self.one_case.get_computation(runtime_measurements)
        raise ValueError


# pylint: disable=too-few-public-methods
class _Value(Generic[T]):
    """
    Leaf node for a MeasurementDependantValue tree structure.
    """

    def __init__(self, *args):
        self.values = args

    def _merge(self, other):
        """
        Works with MeasurementDependantValue._merge
        """
        if isinstance(other, MeasurementDependantValue):
            return MeasurementDependantValue(
                other.dependent_on, self._merge(other.zero_case), self._merge(other.one_case)
            )
        if isinstance(other, _Value):
            return _Value(*self.values, *other.values)
        return _Value(*self.values, other)

    def _transform_leaves(self, fun):
        """
        Works with MeasurementDependantValue._transform_leaves
        """
        return _Value(fun(*self.values))

    def get_computation(self, _):
        """
        Works with MeasurementDependantValue.get_computation
        """
        if len(self.values) == 1:
            return self.values[0]
        return self.values

    def _str_builder(self):
        return [f"=> {', '.join(str(v) for v in self.values)}"]


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
        if_expr: MeasurementDependantValue[bool] = expr

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
            self.measurement_dependant_op: Union[
                MeasurementDependantValue[Operation], _Value[Operation]
            ] = apply_to_measurement_dependant_values(
                lambda *unwrapped: self.op(*unwrapped, do_queue=False, **kwargs)
            )(
                *args
            )
            super().__init__(*args, **kwargs)

    return _ConditionOp
