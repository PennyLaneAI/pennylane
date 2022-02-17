# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mid-circuit measurements and associated operations.
"""

import functools

from typing import Union, Any, TypeVar, Generic, Callable, Type
import uuid
from pennylane.operation import Operation

def mid_measure(wire):
    """
    Create a mid-circuit measurement and return an outcome.

    .. code-block:: python

        m0 = qml.mid_measure(0)
    """
    measurement_id = str(uuid.uuid4())[:8]
    _MidCircuitMeasure(wire, measurement_id)
    return MeasurementDependantValue(measurement_id, _Value(0), _Value(1))


class _MidCircuitMeasure(Operation):
    """
    Operation to perform mid-circuit measurement.
    """

    num_wires = 1

    def __init__(self, wires, measurement_id):
        self.measurement_id = measurement_id
        super().__init__(wires=wires)


T = TypeVar("T")


# pylint: disable=protected-access
class MeasurementDependantValue(Generic[T]):
    """A class representing unknown measurement outcomes.
    Since we don't know the actual outcomes at circuit creation time,
    consider all scenarios.

    supports python __dunder__ mathematical operations.
    """

    __slots__ = ("_depends_on", "_zero_case", "_one_case")

    def __init__(
        self,
        measurement_id: str,
        zero_case: Union["MeasurementDependantValue[T]", "_Value[T]"],
        one_case: Union["MeasurementDependantValue[T]", "_Value[T]"],
    ):
        self._depends_on = measurement_id
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
        measurements = self.measurements
        lines = []
        for k, v in self.branches.items():
            lines.append("if " + ",".join([f"{measurements[i]}={k[i]}" for i in range(len(measurements))]) + " => " + str(v))
        return "\n".join(lines)


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
            if self._depends_on == other._depends_on:
                return MeasurementDependantValue(
                    self._depends_on,
                    self._zero_case._merge(other._zero_case),
                    self._one_case._merge(other._one_case),
                )
            if self._depends_on < other._depends_on:
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
