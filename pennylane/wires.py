# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module contains the :class:`Wires` class.
"""
from collections import Set, Sequence

# tolerance for wire indices
TOLERANCE = 1e-8

class WireError(Exception):
    """Exception raised by a :class:`~.pennylane.wires.Wire` when it is unable to process wire objects.
    """


class Wires(Sequence, Set):

    def __init__(self, wires):
        """
        A bookkeeping class for wires, which are ordered collections of unique non-negative integers that
        represent the index of a wire.

        Args:
             wires (iterable): Ordered collection of unique wire indices. Takes common types of iterables,
                such as lists, tuples and numpy arrays. The element of the iterable must be a
                non-negative integer. If elements are floats, they are internally converted to integers,
                throwing an error if the rounding error exceeds TOLERANCE.
        """

        if wires is not None:
            self._wires = list(wires)

        # Turn elements into integers, if possible
        for idx, w in enumerate(self._wires):

            if not isinstance(w, int):
                # Try to parse to int
                try:
                    # This works for floats and numpy.int
                    difference = abs(w - round(w))
                    if difference < TOLERANCE:
                        self._wires[idx] = int(w)
                    else:
                        raise TypeError
                except TypeError:
                    raise WireError("Wire indices must be integers; got type {}.".format(type(w)))

        # Check that indices are non-negative
        for w in self._wires:
            if w < 0:
                raise WireError("Wire indices must be non-negative; got index {}.".format(w))

        # Check that indices are unique
        if len(set(wires)) != len(wires):
            raise WireError("Each wire must be represented by a unique index; got {}.".format(wires))

    def __getitem__(self, idx):
        return self._wires[idx]

    def __len__(self):
        return len(self._wires)

    def injective_map_exists(self, wires):
        """
        Checks that there is an injective mapping between ``wires`` and this object.

        Since :class:`pennylane.wires.Wires`` objects are by definition collections of unique elements, we
        only need to check the length of both sequences.
        """

        if not isinstance(wires, Wires):
            raise WireError("Expected a pennylane.wires.Wires object; got input of type {}.".format(type(wires)))

        if len(self._wires) == len(wires):
            return True
        else:
            return False

