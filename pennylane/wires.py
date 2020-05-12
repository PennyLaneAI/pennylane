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
This module contains the :class:`Wires` class, which takes care of wire bookkeeping.
"""
from collections import Sequence, Iterable

# tolerance for wire indices
TOLERANCE = 1e-8


class WireError(Exception):
    """Exception raised by a :class:`~.pennylane.wires.Wire` object when it is unable to process wires.
    """


class Wires(Sequence):
    def __init__(self, wires):
        """
        A bookkeeping class for wires, which are ordered collections of unique non-negative integers that
        represent the index of a quantum subsystem such as a qubit or qmode.

        Args:
             wires (int or iterable): Single wire index represented by a non-negative integer,
                or a ordered collection of unique wire indices represented by any common type of iterables,
                such as list, tuple, range and numpy array. The elements of the iterable must be
                non-negative integers. If elements are floats, they are internally converted to integers,
                throwing an error if the rounding error exceeds TOLERANCE.
        """

        if isinstance(wires, Iterable):
            self._wires = list(wires)
        else:
            self._wires = [wires]

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
        if len(set(self._wires)) != len(self._wires):
            raise WireError(
                "Each wire must be represented by a unique index; got {}.".format(self._wires)
            )

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
            raise WireError(
                "Expected a pennylane.wires.Wires object; got input of type {}.".format(type(wires))
            )

        if len(self._wires) == len(wires):
            return True
        else:
            return False
