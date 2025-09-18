# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Defines the MeasurementValue class
"""
from collections.abc import Callable
from typing import Generic, TypeVar

from pennylane import math
from pennylane.wires import Wires

T = TypeVar("T")


def no_processing(results):
    """A postprocessing function with no effect."""
    return results


class MeasurementValue(Generic[T]):
    """A class representing unknown measurement outcomes in the qubit model.

    Measurements on a single qubit in the computational basis are assumed.

    Args:
        measurements (list[.MidMeasureMP]): The measurement(s) that this object depends on.
        processing_fn (callable | None): A lazy transformation applied to the measurement values.
    """

    name = "MeasurementValue"

    def __init__(self, measurements: list, processing_fn: Callable | None = None):
        self.measurements = measurements
        self._processing_fn = processing_fn

    @property
    def has_processing(self) -> bool:
        """Whether or not classical processing is applied to the measurement value."""
        return self._processing_fn is not None

    @property
    def processing_fn(self) -> Callable:
        """A lazy transformation applied to the measurement values."""
        if self._processing_fn is None:
            return no_processing
        return self._processing_fn

    def items(self):
        """A generator representing all the possible outcomes of the MeasurementValue."""
        num_meas = len(self.measurements)
        for i in range(2**num_meas):
            branch = tuple(int(b) for b in f"{i:0{num_meas}b}")
            yield branch, self.processing_fn(*branch)

    def postselected_items(self):
        """A generator representing all the possible outcomes of the MeasurementValue,
        taking postselection into account."""
        # pylint: disable=stop-iteration-return
        ps = {i: p for i, m in enumerate(self.measurements) if (p := m.postselect) is not None}
        num_non_ps = len(self.measurements) - len(ps)
        if num_non_ps == 0:
            yield (), self.processing_fn(*ps.values())
            return
        for i in range(2**num_non_ps):
            # Create the branch ignoring postselected measurements
            non_ps_branch = tuple(int(b) for b in f"{i:0{num_non_ps}b}")
            # We want a consumable iterable and the static tuple above
            non_ps_branch_iter = iter(non_ps_branch)
            # Extend the branch to include postselected measurements
            full_branch = tuple(
                ps[j] if j in ps else next(non_ps_branch_iter)
                for j in range(len(self.measurements))
            )
            # Return the reduced non-postselected branch and the procesing function
            # evaluated on the full branch
            yield non_ps_branch, self.processing_fn(*full_branch)

    @property
    def wires(self):
        """Returns a list of wires corresponding to the mid-circuit measurements."""
        return Wires.all_wires([m.wires for m in self.measurements])

    @property
    def branches(self):
        """A dictionary representing all possible outcomes of the MeasurementValue."""
        ret_dict = {}
        num_meas = len(self.measurements)
        for i in range(2**num_meas):
            branch = tuple(int(b) for b in f"{i:0{num_meas}b}")
            ret_dict[branch] = self.processing_fn(*branch)
        return ret_dict

    def map_wires(self, wire_map):
        """Returns a copy of the current ``MeasurementValue`` with the wires of each measurement changed
        according to the given wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            MeasurementValue: new ``MeasurementValue`` instance with measurement wires mapped
        """
        mapped_measurements = [m.map_wires(wire_map) for m in self.measurements]
        return MeasurementValue(mapped_measurements, self._processing_fn)

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
        return self._apply(math.logical_not)

    def __bool__(self) -> bool:
        raise ValueError(
            "The truth value of a MeasurementValue is undefined. To condition on a MeasurementValue, please use qml.cond instead."
        )

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
        return self._apply(lambda v: other * math.cast_like(v, other))

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
        return self._transform_bin_op(math.logical_and, other)

    def __or__(self, other):
        return self._transform_bin_op(math.logical_or, other)

    def __mod__(self, other):
        return self._transform_bin_op(math.mod, other)

    def __xor__(self, other):
        return self._transform_bin_op(math.logical_xor, other)

    def _apply(self, fn):
        """Apply a post computation to this measurement"""
        return MeasurementValue(self.measurements, lambda *x: fn(self.processing_fn(*x)))

    def concretize(self, measurements: dict):
        """Returns a concrete value from a dictionary of hashes with concrete values."""
        values = tuple(measurements[meas] for meas in self.measurements)
        return self.processing_fn(*values)

    def _merge(self, other: "MeasurementValue"):
        """Merge two measurement values"""

        # create a new merged list with no duplicates and in lexical ordering
        merged_measurements = list(set(self.measurements).union(set(other.measurements)))
        merged_measurements.sort(key=lambda m: m.id)

        # create a new function that selects the correct indices for each sub function
        def merged_fn(*x):
            sub_args_1 = (x[i] for i in [merged_measurements.index(m) for m in self.measurements])
            sub_args_2 = (x[i] for i in [merged_measurements.index(m) for m in other.measurements])

            out_1 = self.processing_fn(*sub_args_1)
            out_2 = other.processing_fn(*sub_args_2)

            return out_1, out_2

        return MeasurementValue(merged_measurements, merged_fn)

    def __getitem__(self, i):
        branch = tuple(int(b) for b in f"{i:0{len(self.measurements)}b}")
        return self.processing_fn(*branch)

    def __str__(self):
        lines = []
        num_meas = len(self.measurements)
        for i in range(2**num_meas):
            branch = tuple(int(b) for b in f"{i:0{num_meas}b}")
            id_branch_mapping = [
                f"{self.measurements[j].id}={branch[j]}" for j in range(len(branch))
            ]
            lines.append(
                "if " + ",".join(id_branch_mapping) + " => " + str(self.processing_fn(*branch))
            )
        return "\n".join(lines)

    def __repr__(self):
        return f"MeasurementValue(wires={self.wires.tolist()})"
