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
from numbers import Number
import numpy as np  # for random functions
TOLERANCE = 1e-8 # tolerance for integer-like wire indices


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
            they are internally converted to integers, throwing an error if the rounding error exceeds 1e-8.
    """

    def __init__(self, wires):

        if isinstance(wires, Number):
            # If input is a number, interpret as single wire index and wrap as a list
            wires = [wires]

        if isinstance(wires, Wires):
            # If input is already a Wires object, just copy its wire list
            self.wire_list = wires.wire_list
        elif isinstance(wires, Iterable):
            # If input is an iterable, interpret as a collection of wire indices
            self.wire_list = _clean(wires)
        else:
            raise WireError(
                "Received unexpected wires input {} of type {}.".format(wires, type(wires))
            )

    def __getitem__(self, idx):
        """Method to support indexing."""
        return Wires(self.wire_list[idx])

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

    def __lt__(self, other):
        """Implements the '<' operator for Wires objects of length 1."""
        if len(self.wire_list) == 1 and len(other.wire_list) == 1:
            return self.wire_list[0] < other.wire_list[0]
        else:
            raise WireError("Cannot compare Wires objects of length larger than 1 with '<' operator.")

    def __le__(self, other):
        """Implements the '<=' operator for Wires objects of length 1."""
        if len(self.wire_list) == 1 and len(other.wire_list) == 1:
            return self.wire_list[0] <= other.wire_list[0]
        else:
            raise WireError("Cannot compare Wires objects of length larger than 1 with '<=' operator.")

    def __gt__(self, other):
        """Implements the '>' operator for Wires objects of length 1."""
        if len(self.wire_list) == 1 and len(other.wire_list) == 1:
            return self.wire_list[0] > other.wire_list[0]
        else:
            raise WireError("Cannot compare Wires objects of length larger than 1 with '>' operator.")

    def __ge__(self, other):
        """Implements the '>=' operator for Wires objects of length 1."""
        if len(self.wire_list) == 1 and len(other.wire_list) == 1:
            return self.wire_list[0] < other.wire_list[0]
        else:
            raise WireError("Cannot compare Wires objects of length larger than 1 with '<' operator.")

    def index(self, wire):
        """Overwrites the ``index()`` function which returns the index of 'wire'.

        Args:
            Wire (int): Wires object of a single wire

        Returns:
            int: index of wire
        """
        if not isinstance(wire, Wires):
            raise WireError(
                "expected a `pennylane.wires.Wires` object; got {} of type {}.".format(
                    wire, type(wire)
                )
            )

        if len(wire) != 1:
            raise WireError("Can only retrieve an index of a Wires object of length 1.")

        try:
            idx = self.wire_list.index(wire.wire_list[0])
        # if wire is not in wire_list of this object
        except ValueError:
            idx = None
        return idx

    def as_ndarray(self):
        """Returns a numpy array representation of the Wires object.

        Returns:
            ndarray: array representing Wires object
        """
        return np.array(self.wire_list)

    def as_list(self):
        """Returns a list representation of the Wires object.

        Returns:
            List: list representing Wires object
        """
        return self.wire_list

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

        additional_wires = [w for w in wires.wire_list if w not in self.wire_list]
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

    def subset(self, indices, periodic_boundary=False):
        """
        Returns a new Wires object which is a subset of this Wires object. The wires of the new
        object are the wires at positions specified by 'indices'.

        For example:

        >>> wires = Wires([4, 0, 1, 5, 6])
        >>> wires.subset([2, 3, 0]) == Wires([1, 5, 4])
        True

        >>> wires.subset(1) == Wires([0])
        True

        Args:
            indices (List[int] or int): indices or index of the wires we want to select
            periodic_boundary (bool): whether the modulo of the number of wires of an index is used instead of an index.
                implements periodic boundary conditions in the indexing,
                so that for example ``wires.select(len(wires)) == wires.select(0)``.

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
                    "cannot subset wire at index {} from {} wires.".format(i, len(self))
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

    @staticmethod
    def merge(list_of_wires):
        """Merge Wires objects in list to one

        Args:
            list_of_wires (List[Wires]): list of Wires objects

        Return:
            Wires: new Wires object that contains all wires of the list's Wire objects

        Raises:
            WireError: for incorrect input or if Wires objects contain the same wires
        """

        merged_wires = []
        for wires in list_of_wires:

            if not isinstance(wires, Wires):
                raise WireError("Expected list of Wires objects; got {}.".format(list_of_wires))

            if any([w in merged_wires for w in wires.wire_list]):
                raise WireError("Cannot merge Wires objects that contain the same wires; got {}.".format(list_of_wires))
            else:
                merged_wires += wires.wire_list

        return Wires(merged_wires)