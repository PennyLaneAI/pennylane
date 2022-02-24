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
from typing import TypeVar, Generic, Type
import uuid
from pennylane.operation import Operation, AnyWires
from pennylane.queuing import QueuingContext


def measure(wire):
    """
    Create a mid-circuit measurement and return an outcome.

    .. code-block:: python

        m0 = qml.measure(0)
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


class If(Operation):
    """
    If conditional operation wrapper class.
    """

    num_wires = AnyWires

    def __init__(self, expr: MeasurementDependantValue[bool], then_op: Type[Operation], do_queue=True, id=None):
        self.branches = expr.branches
        self.dependant_measurements = expr.measurements
        self.then_op = then_op
        QueuingContext.remove(then_op)
        super().__init__(wires=then_op.wires, do_queue=do_queue, id=id)
