# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the qml.measure measurement.
"""
import uuid
from typing import Generic, TypeVar

import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import Wires

from .measurements import MeasurementProcess, MidMeasure


def measure(wires):  # TODO: Change name to mid_measure
    """Perform a mid-circuit measurement in the computational basis on the
    supplied qubit.

    Measurement outcomes can be obtained and used to conditionally apply
    operations.

    If a device doesn't support mid-circuit measurements natively, then the
    QNode will apply the :func:`defer_measurements` transform.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def func(x, y):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            m_0 = qml.measure(1)

            qml.cond(m_0, qml.RY)(y, wires=0)
            return qml.probs(wires=[0])

    Executing this QNode:

    >>> pars = np.array([0.643, 0.246], requires_grad=True)
    >>> func(*pars)
    tensor([0.90165331, 0.09834669], requires_grad=True)

    Mid circuit measurements can be manipulated using the following dunder methods
    `+`, `-`, `*`, `/`, `~` (not), `&` (and), `|` (or), `==`, `<=`, `>=`, `<`, `>`. With other mid-circuit measurements or scalars.

    Note:
        python `not`, `and`, `or`, do not work since these do not have dunder methods. Instead use `~`, `&`, `|`.

    Args:
        wires (Wires): The wire of the qubit the measurement process applies to.

    Returns:
        MidMeasureMP: measurement process instance

    Raises:
        QuantumFunctionError: if multiple wires were specified
    """
    wire = Wires(wires)
    if len(wire) > 1:
        raise qml.QuantumFunctionError(
            "Only a single qubit can be measured in the middle of the circuit"
        )

    # Create a UUID and a map between MP and MV to support serialization
    measurement_id = str(uuid.uuid4())[:8]
    MidMeasureMP(wires=wire, id=measurement_id)
    return MeasurementValue([measurement_id], processing_fn=lambda v: v)


T = TypeVar("T")


class MidMeasureMP(MeasurementProcess):
    """Mid-circuit measurement.

    Please refer to :func:`measure` for detailed documentation.

    Args:
        obs (.Observable): The observable that is to be measured as part of the
            measurement process. Not all measurement processes require observables (for
            example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    @property
    def return_type(self):
        return MidMeasure

    @property
    def _queue_category(self):
        return "_ops"


class MeasurementValueError(ValueError):
    """Error raised when an unknown measurement value is being used."""


class MeasurementValue(Generic[T]):
    """A class representing unknown measurement outcomes in the qubit model.

    Measurements on a single qubit in the computational basis are assumed.

    Args:
        measurement_ids (list[str]): The id of the measurement that this object depends on.
        processing_fn (callable): A lazily transformation applied to the measurement values.
    """

    def __init__(self, measurement_ids, processing_fn):
        self.measurement_ids = measurement_ids
        self.processing_fn = processing_fn

    def _items(self):
        """A generator representing all the possible outcomes of the MeasurementValue."""
        for i in range(2 ** len(self.measurement_ids)):
            branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurement_ids)))
            yield branch, self.processing_fn(*branch)

    @property
    def branches(self):
        """A dictionary representing all possible outcomes of the MeasurementValue."""
        ret_dict = {}
        for i in range(2 ** len(self.measurement_ids)):
            branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurement_ids)))
            ret_dict[branch] = self.processing_fn(*branch)
        return ret_dict

    def _transform_bin_op(self, base_bin, other):
        """Helper function for defining dunder binary operations."""
        if isinstance(other, MeasurementValue):
            # pylint: disable=protected-access
            return self._merge(other)._apply(lambda t: base_bin(t[0], t[1]))
        # if `other` is not a MeasurementValue then apply it to each branch
        return self._apply(lambda v: base_bin(v, other))

    def __invert__(self):
        """Return a copy of the measurement value with an inverted control
        value."""
        return self._apply(lambda v: not v)

    def __eq__(self, other):
        return self._transform_bin_op(lambda a, b: a == b, other)

    def __ne__(self, other):
        return self._transform_bin_op(lambda a, b: a != b, other)

    def __add__(self, other):
        return self._transform_bin_op(lambda a, b: a + b, other)

    def __radd__(self, other):
        return self._apply(lambda v: other + v)

    def __sub__(self, other):
        return self._transform_bin_op(lambda a, b: a - b, other)

    def __rsub__(self, other):
        return self._apply(lambda v: other - v)

    def __mul__(self, other):
        return self._transform_bin_op(lambda a, b: a * b, other)

    def __rmul__(self, other):
        return self._apply(lambda v: other * v)

    def __truediv__(self, other):
        return self._transform_bin_op(lambda a, b: a / b, other)

    def __rtruediv__(self, other):
        return self._apply(lambda v: other / v)

    def __lt__(self, other):
        return self._transform_bin_op(lambda a, b: a < b, other)

    def __le__(self, other):
        return self._transform_bin_op(lambda a, b: a <= b, other)

    def __gt__(self, other):
        return self._transform_bin_op(lambda a, b: a > b, other)

    def __ge__(self, other):
        return self._transform_bin_op(lambda a, b: a >= b, other)

    def __and__(self, other):
        return self._transform_bin_op(lambda a, b: a and b, other)

    def __or__(self, other):
        return self._transform_bin_op(lambda a, b: a or b, other)

    def _apply(self, fn):
        """Apply a post computation to this measurement"""
        return MeasurementValue(self.measurement_ids, lambda *x: fn(self.processing_fn(*x)))

    def _merge(self, other: "MeasurementValue"):
        """Merge two measurement values"""

        # create a new merged list with no duplicates and in lexical ordering
        merged_measurement_ids = list(set(self.measurement_ids).union(set(other.measurement_ids)))
        merged_measurement_ids.sort()

        # create a new function that selects the correct indices for each sub function
        def merged_fn(*x):
            sub_args_1 = (
                x[i] for i in [merged_measurement_ids.index(m) for m in self.measurement_ids]
            )
            out_1 = self.processing_fn(*sub_args_1)

            sub_args_2 = (
                x[i] for i in [merged_measurement_ids.index(m) for m in other.measurement_ids]
            )
            out_2 = other.processing_fn(*sub_args_2)

            return out_1, out_2

        return MeasurementValue(merged_measurement_ids, merged_fn)

    def __getitem__(self, i):
        branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurement_ids)))
        return self.processing_fn(*branch)

    def __str__(self):
        lines = []
        for i in range(2 ** (len(self.measurement_ids))):
            branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurement_ids)))
            id_branch_mapping = [
                f"{self.measurement_ids[j]}={branch[j]}" for j in range(len(branch))
            ]
            lines.append(
                "if " + ",".join(id_branch_mapping) + " => " + str(self.processing_fn(*branch))
            )
        return "\n".join(lines)
