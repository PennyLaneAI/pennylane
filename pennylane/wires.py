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
import numpy as np  # for random functions
from copy import deepcopy

# tolerance for integer-like wire indices
TOLERANCE = 1e-8


class WireError(Exception):
    """Exception raised by a :class:`~.pennylane.wires.Wire` object when it is unable to process wires.
    """


def _clean(iterable):
    """Converts 'iterable' into list of integers, and checks that all integers are unique and non-negative.

        Args:
            iterable (Iterable): some iterable object that represents a sequence of wire indices

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
                    "Wire indices must be integers; got {} of type {}.".format(w, type(w))
                )

    # Check that indices are non-negative
    for w in iterable:
        if w < 0:
            raise WireError("Wire indices must be non-negative; got index {}.".format(w))

    # Check that indices are unique
    if len(set(iterable)) != len(iterable):
        raise WireError("Each wire must be represented by a unique index; got {}.".format(iterable))

    return iterable


class Wires(Sequence):
    """
    A bookkeeping class for wires, which are ordered collections of unique non-negative integers that
    represent the index of a quantum subsystem such as a qubit or qmode.

    Args:
         wires (int or iterable): Iterable representing an ordered collection of unique wire indices.
            The iterable can be of any common type such as list, tuple, range or numpy array.
            The elements of the iterable must be non-negative integers. If elements are floats,
            they are internally converted to integers, throwing an error if the rounding error exceeds TOLERANCE.
    """

    def __init__(self, wires):

        if isinstance(wires, Iterable):
            # If input is an iterable, interpret as a collection of wire indices
            self.wire_list = _clean(wires)

        elif isinstance(wires, int) and wires >= 0:
            # If input is non-negative integer, interpret as single wire index
            # and wrap as a list
            self.wire_list = [wires]

        else:
            raise WireError(
                "received unexpected wires input {} of type {}.".format(wires, type(wires))
            )

    def __getitem__(self, idx):
        """Method to support indexing."""
        return self.wire_list[idx]

    def __len__(self):
        """Method to support ``len()``."""
        return len(self.wire_list)

    def __repr__(self):
        """Method defining the string representation of this class."""
        return "<Wires = {}>".format(self.wire_list)

    def __eq__(self, other):
        """Method to support the '==' operator."""
        if isinstance(other, self.__class__):
            return self.wire_list == other.wire_list
        
        return False

    def __ne__(self, other):
        """Method to support the '!=' operator."""
        return not self.__eq__(other)

    def combine(self, wires):
        """Return a copy of this ``Wires`` object extended by the unique wires of ``wires``.

        For example:

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([3, 0, 4])
        >>> wires1.combine(wires2)
        <Wires [4, 0, 1, 3]>

        Args:
            wires (Wires): A Wires object whose unique indices are combined with this Wires object.

        Returns:
            Wires: combined wires
        """

        if not isinstance(wires, Wires):
            raise WireError(
                "expected a `pennylane.wires.Wires` object; got {} of type {}".format(
                    wires, type(wires)
                )
            )

        additional_wires = [w for w in wires if w not in self.wire_list]
        combined_wire_list = self.wire_list + additional_wires
        return Wires(combined_wire_list)

    def intersect(self, wires):
        """
        Creates a wires object that is the intersection of wires between 'wires' and this object.

        For example:

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([0, 4, 3])
        >>> wires1.intersect(wires2)
        <Wires = [4, 0]>

        The returned wire sequence is ordered according to the order of this Wires object.

        Args:
            wires (Wires): A Wires class object

        Returns:
            Wires: intersection of this Wires object and 'wires'
        """

        if not isinstance(wires, Wires):
            raise WireError(
                "expected a `pennylane.wires.Wires` object; got {} of type {}.".format(
                    wires, type(wires)
                )
            )

        intersect = [w for w in self.wire_list if w in wires.wire_list]
        return Wires(intersect)

    def difference(self, wires):
        """
        Creates a wires object that is the difference of wires between 'wires' and this object.

        For example:

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([0, 2, 3])
        >>> wires1.difference(wires2)
        <Wires = [4, 1]>

        The returned wire sequence is ordered according to the order of this Wires object.

        Args:
            wires (Wires): A Wires class object

        Returns:
            Wires: difference of this Wires object and 'wires'
        """

        if not isinstance(wires, Wires):
            raise WireError(
                "expected a `pennylane.wires.Wires` object; got {} of type {}.".format(
                    wires, type(wires)
                )
            )

        diff = [w for w in self.wire_list if w not in wires.wire_list]
        return Wires(diff)

    def get_indices(self, wires):
        """
        Return the indices of the wires in this Wires object.

        For example,

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([1, 4])
        >>> wires1.get_indices(wires2)
        [2, 0]

        Args:
            wires (Wires): A Wires class object

        Returns:
            List: index list
        """

        if not isinstance(wires, Wires):
            raise WireError(
                "expected a `pennylane.wires.Wires` object; got {} of type {}.".format(
                    wires, type(wires)
                )
            )

        return [self.index(w) for w in wires]

    def injective_map_exists(self, wires):
        """
        Checks that there is an injective mapping between ``wires`` and this object.

        Since :class:`pennylane.wires.Wires`` objects are by definition collections of unique elements, we
        only need to check the length of both sequences.

        Args:
            wires (Wires): A Wires class object

        Returns:
            bool: whether injective mapping exists or not
        """

        if not isinstance(wires, Wires):
            raise WireError(
                "expected a `pennylane.wires.Wires` object; got {} of type {}.".format(
                    wires, type(wires)
                )
            )

        if len(self.wire_list) == len(wires):
            return True
        
        return False

    def select(self, indices, periodic_boundary=False):
        """
        Returns a subset of the Wires specified by the 'indices'.

        For example:

        >>> wires = Wires([4, 0, 1, 5, 6])
        >>> wires.select([2, 3, 0]) == Wires([1, 5, 4])
        True

        >>> wires.select(1) == Wires([0])
        True

        Args:
            indices (List[int] or int): indices or index of the wires we want to select

        Returns:
            Wires: subset of wires
        """

        if isinstance(indices, int):
            indices = [indices]

        if periodic_boundary:
            # replace indices by their modulo
            indices = [i % len(self) for i in indices]

        for i in indices:
            if i > len(self):
                raise WireError(
                    "cannot select wire at index {} from {} wires.".format(i, len(self))
                )

        subset = [self.wire_list[i] for i in indices]
        return Wires(subset)

    def select_random(self, n_samples, seed=None):
        """
        Returns a randomly sampled subset of Wires of length 'n_samples'.

        Args:
            n_samples (int): number of subsampled wires
            seed (int): optional random seed used for selecting the wires

        Returns:
            Wires: random subset of wires
        """

        if n_samples > len(self):
            raise WireError("cannot sample {} wires from {} wires.".format(n_samples, len(self)))

        if seed is not None:
            np.random.seed(seed)

        indices = np.random.choice(len(self), size=n_samples, replace=False)
        subset = [self.wire_list[i] for i in indices]
        return Wires(subset)
