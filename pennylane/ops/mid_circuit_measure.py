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
    return MeasurementDependantValue(measurement_id, 0, 1)


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
        zero_case: int,
        one_case: int,
    ):
        self._depends_on = measurement_id
        self._zero_case = zero_case
        self._one_case = one_case

    @property
    def branches(self):
        """A dictionary representing all the possible outcomes of the MeasurementDependantValue."""
        branch_dict = {}
        branch_dict[(0,)] = self._zero_case
        branch_dict[(1,)] = self._one_case
        return branch_dict

    @property
    def measurements(self):
        """List of all measurements this MeasurementDependantValue depends on."""
        return [self._depends_on]

    def __str__(self):
        measurements = self.measurements
        lines = []
        for k, v in self.branches.items():
            lines.append(
                "if "
                + ",".join([f"{measurements[i]}={k[i]}" for i in range(len(measurements))])
                + " => "
                + str(v)
            )
        return "\n".join(lines)


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

            # TODO: make dynamically inspected or otherwise refactor
            op_kwargs = {k: v for k, v in kwargs.items() if k in ("wires", "do_queue", "id")}
            super().__init__(*args, **op_kwargs)

    return _IfOp
