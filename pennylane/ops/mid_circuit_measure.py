"""
Mid-circuit measurements and associated operations.
"""

import functools
import uuid
from typing import Union, Any, TypeVar, Generic, Callable, Type, Iterable, Sequence

from pennylane.wires import Wires
from pennylane.operation import Operation, Operator, AnyWires, AllWires, Observable


class MeasurementIdentifier:

    def __init__(self, wire, measurement_id=None):
        if not measurement_id:
            self.id = str(uuid.uuid4())[:8]
        else:
            self.id = measurement_id
        self.wire = wire

    def __str__(self):
        return f"{type(self).__name__}({repr(self.wire)}, {repr(self.id)})"

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.wire)}, {repr(self.id)})"

    def __hash__(self):
        return hash((MeasurementIdentifier, self.wire, self.id))

    def __eq__(self, other):
        if self.id == other.id:
            if self.wire != other.wire:
                raise ValueError
            return True
        return False


def mid_measure(wire):
    """
    Create a mid-circuit measurement and return an outcome.

    .. code-block:: python

        m0 = qml.mid_measure(0)
    """
    mc_op = _MidCircuitMeasure(wire)
    return MeasurementDependantValue(mc_op.measurement, _Value(0), _Value(1))


class _MidCircuitMeasure(Operation):
    """
    Operation to perform mid-circuit measurement.
    """

    num_wires = 1

    def __init__(self, wires):
        if isinstance(wires, Sequence):
            self.measurement = MeasurementIdentifier(wires[0])
        else:
            self.measurement = MeasurementIdentifier(wires)
        super().__init__(wires=wires)

    @property
    def wires(self):
        return Wires([self.measurement.wire])

    def __str__(self):
        return f"{type(self).__name__}({self.wires[0]})"


# pylint: disable=protected-access
def apply_to_measurement_dependant_values(fun):
    """
    Apply an arbitrary function to a `MeasurementDependantValue` or set of `MeasurementDependantValue`s.
    (fun should be a "pure" function)

    Ex:

    .. code-block:: python

        m0 = qml.mid_measure(0)
        m0_sin = qml.apply_to_measurement_dependant_value(np.sin)(m0)
    """

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        partial = _Value()
        for arg in args:
            if not isinstance(arg, MeasurementDependantValue):
                arg = _Value(arg)
            partial = partial._merge(arg)
        partial._transform_leaves_inplace(
            lambda *unwrapped: fun(*unwrapped, **kwargs)  # pylint: disable=unnecessary-lambda
        )
        return partial
    return wrapper


T = TypeVar("T")


# pylint: disable=protected-access
class MeasurementDependantValue(Generic[T]):
    """A class representing unknown measurement outcomes.
    Since we don't know the actual outcomes at circuit creation time,
    consider all scenarios.

    supports python __dunder__ mathematical operations. As well as arbitrary functions using
    qml.apply_to_measurement_dependant_value.
    """

    __slots__ = ("_depends_on", "_zero_case", "_one_case")

    def __init__(
        self,
        measurement: MeasurementIdentifier,
        zero_case: Union["MeasurementDependantValue[T]", "_Value[T]"],
        one_case: Union["MeasurementDependantValue[T]", "_Value[T]"],
    ):
        self._depends_on = measurement
        self._zero_case = zero_case
        self._one_case = one_case

    @property
    def branches(self):
        """A dictionary representing all the possible outcomes of the MeasurementDependantValue."""
        branch_dict = {}
        if isinstance(self._zero_case, MeasurementDependantValue):
            for k, v in self._zero_case.branches.items():
                branch_dict[(0, *k)] = v
            # if _zero_case is a MeaurementDependantValue, then _one_case is too.
            for k, v in self._one_case.branches.items():
                branch_dict[(1, *k)] = v
        else:
            branch_dict[(0,)] = self._zero_case.values
            branch_dict[(1,)] = self._one_case.values
        return branch_dict

    @property
    def measurements(self):
        """List of all measurements this MeasurementDependantValue depends on."""
        if isinstance(self._zero_case, MeasurementDependantValue):
            return [self._depends_on, *self._zero_case.measurements]
        return [self._depends_on]

    def logical_not(self):
        """Perform a logical not on the MeasurementDependantValue.
        """
        return apply_to_measurement_dependant_values(lambda x: not x)(self)

    # define all mathematical __dunder__ methods https://docs.python.org/3/library/operator.html
    def __add__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: x + y)(self, other)

    def __radd__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: y + x)(self, other)

    def __mul__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: x * y)(self, other)

    def __rmul__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: y * x)(self, other)

    def __and__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: x and y)(self, other)

    def __rand__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: y and x)(self, other)

    def __or__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: x or y)(self, other)

    def __ror__(self, other: Any):
        return apply_to_measurement_dependant_values(lambda x, y: y or x)(self, other)

    def __str__(self):
        return f"{type(self).__name__}({repr(self._depends_on)}, {repr(self._zero_case)}, {repr(self._one_case)}"

    def __repr__(self):
        return f"{type(self).__name__}({repr(self._depends_on)}, {repr(self._zero_case)}, {repr(self._one_case)}"


    def _merge(self, other: Union["MeasurementDependantValue", "_Value"]):
        """
        Merge this MeasurementDependantValue with `other`.

        Ex: Merging a MeasurementDependantValue such as:
        .. code-block:: python

            if wire_0=0 => 3.4
            if wire_0=1 => 1

        with another MeasurementDependantValue:

        .. code-block:: python

            if wire_1=0 => 100
            if wire_1=1 => 67

        will result in:

        .. code-block:: python

            wire_0=0,wire_1=0 => 3.4,100
            wire_0=0,wire_1=1 => 3.4,67
            wire_0=1,wire_1=0 => 1,100
            wire_0=1,wire_1=1 => 1,67

        (note the uuids in the example represent distinct measurements of different qubit.)

        """
        if isinstance(other, MeasurementDependantValue):
            if self._depends_on.id == other._depends_on.id:
                return MeasurementDependantValue(
                    self._depends_on,
                    self._zero_case._merge(other._zero_case),
                    self._one_case._merge(other._one_case),
                )
            if self._depends_on.id < other._depends_on.id:
                return MeasurementDependantValue(
                    self._depends_on,
                    self._zero_case._merge(other),
                    self._one_case._merge(other),
                )
            return MeasurementDependantValue(
                other._depends_on,
                self._merge(other._zero_case),
                self._merge(other._one_case),
            )
        return MeasurementDependantValue(
            self._depends_on,
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

    def _merge(self, other: Union["MeasurementDependantValue", "_Value"]):
        """
        Works with MeasurementDependantValue._merge
        """
        if isinstance(other, MeasurementDependantValue):
            return MeasurementDependantValue(
                other._depends_on, self._merge(other._zero_case), self._merge(other._one_case)
            )
        return _Value(*self._values, *other._values)

    def _transform_leaves_inplace(self, fun):
        """
        Works with MeasurementDependantValue._transform_leaves
        """
        self._values = (fun(*self._values),)

    @property
    def values(self):
        """Values this Leaf node is holding."""
        if len(self._values) == 1:
            return self._values[0]
        return self._values

    def __str__(self):
        return f"{type(self).__name__}({', '.join(str(v) for v in self._values)})"

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(str(v) for v in self._values)})"


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
        dependant_measurements = expr.measurements

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
            )(*args)
            self.branches = measurement_dependant_op.branches
            self.dependant_measurements = measurement_dependant_op.measurements
            super().__init__(*args, **kwargs)

    return _ConditionOp


class RandomVariable(Observable):

    @classmethod
    def _eigvals(cls, *params):
        pass

    def diagonalizing_gates(self):
        return []

    def __init__(self, measurement_dependant_value):
        self.measurement_dependant_value = measurement_dependant_value
        super().__init__(wires=AnyWires)

    @classmethod
    def _matrix(cls, *params):
        raise NotImplemented

    @property
    def num_wires(self):
        return AllWires

    @staticmethod
    def decomposition(*params, wires):
        raise NotImplemented