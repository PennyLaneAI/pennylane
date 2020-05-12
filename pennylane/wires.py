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


def _clean(iterable):
    """Turns 'iterable' into list of non-negative integers, and checks that all integers are unique.

        Args:
            iterable (Iterable): iterable of wire indices

        Returns:
            list[int]: cleaned wire index sequence

        Raises:
            WireError if elements cannot be turned into integers, if they are negative or not unique.
        """

    iterable = list(iterable)

    # Turn elements into integers, if possible
    for idx, w in enumerate(iterable):

        if not isinstance(w, int):
            # Try to parse integer-like elements to int
            try:
                # This works for floats and numpy.int
                difference = abs(w - round(w))
                if difference < TOLERANCE:
                    iterable[idx] = int(w)
                else:
                    raise TypeError
            except TypeError:
                raise WireError(
                    "Wire indices must be integers; got type {}.".format(type(w))
                )

    # Check that indices are non-negative
    for w in iterable:
        if w < 0:
            raise WireError("Wire indices must be non-negative; got index {}.".format(w))

    # Check that indices are unique
    if len(set(iterable)) != len(iterable):
        raise WireError(
            "Each wire must be represented by a unique index; got {}.".format(
                iterable
            )
        )

    return iterable


class Wires(Sequence):
    def __init__(self, wires, accept_integer=True):
        """
        A bookkeeping class for wires, which are ordered collections of unique non-negative integers that
        represent the index of a quantum subsystem such as a qubit or qmode.

        Args:
             wires (int or iterable): Integer specifying the number of consecutive wire indices,
                or a ordered collection of unique wire indices represented by any common type of iterable
                (such as list, tuple, range and numpy array). The elements of the iterable must be
                non-negative integers. If elements are floats, they are internally converted to integers,
                throwing an error if the rounding error exceeds TOLERANCE.
             accept_integer (bool): If False, only an iterable is accepted as argument
        """

        if isinstance(wires, Iterable):
            # If input is an iterable, interpret as a collection of wire indices
            self.wire_list = _clean(wires)

        elif accept_integer and isinstance(wires, int):
            if wires >= 0:
                # If input is non-negative integer, interpret it as the
                # number of consecutive wires
                self.wire_list = list(range(wires))
            else:
                raise WireError("Number of wires cannot be negative; got {}.".format(wires))
        else:
            raise WireError(
                "Unexpected wires input; got {} of type {}.".format(wires, type(wires))
            )

    def __getitem__(self, idx):
        return self.wire_list[idx]

    def __len__(self):
        return len(self.wire_list)

    def extend(self, wires):
        """Extend this ``Wires`` object by the indices of another.

        Args:
            wires (Wires): A Wires object whose unique indices are supposed to be added to this one.
        """

        if not isinstance(wires, Wires):
            raise WireError(
                "Expected wires object to extend this Wires object; "
                "got {} of type {}".format(wires, type(wires))
            )

        self.wire_list.extend(wires.wire_list)

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

        if len(self.wire_list) == len(wires):
            return True
        else:
            return False
