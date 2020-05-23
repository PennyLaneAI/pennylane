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


class WireError(Exception):
    """Exception raised by a :class:`~.pennylane.wires.Wire` object when it is unable to process wires.
    """


class Wires(Sequence):
    """
    A bookkeeping class for wires, which are ordered collections of unique objects. The i'th object represents
    the i'th quantum subsystem.

    There is no conceptual difference between registers of multiple wires and single wires,
    which are just wire registers of length one.

    Indexing and slicing this sequence will return another ``Wires`` object.

    Args:
         wires (Any): If iterable, interpreted as an ordered collection of unique objects representing wires.
            Any other input type is converted into an iterable of a single entry, and hence interpreted as a
            single wire.
    """

    def __init__(self, wires):

        if isinstance(wires, Wires):
            # If input is already a Wires object, just copy its wire list
            self.wire_list = wires.wire_list

        elif isinstance(wires, Iterable):
            # If input is an iterable, check that entries are unique and convert to a list
            try:
                if len(set(wires)) != len(wires):
                    raise WireError(
                        "Wires must be unique; got {}.".format(wires)
                    )
            except TypeError:
                raise WireError("Cannot create wires from elements of type {}.".format(type(wires[0]))) # TODO: edge case that iterable contains different types

            self.wire_list = list(wires)

        else:
            # If input is not an iterable, interpret it as an object representing a single wire
            self.wire_list = [wires]

    def __getitem__(self, idx):
        """Method to support indexing. Returns a Wires object representing a register with a single wire."""
        return Wires(self.wire_list[idx])

    def __len__(self):
        """Method to support ``len()``."""
        return len(self.wire_list)

    def __contains__(self, item):
        """Method checking if Wires object contains an object."""
        if isinstance(item, Wires):
            # If all wires can be found in this object, return True
            if all(wire in self.wire_list for wire in item.wire_list):
                return True
        return False

    def __repr__(self):
        """Method defining the string representation of this class."""
        return "<Wires = {}>".format(self.wire_list)

    def __eq__(self, other):
        """Method to support the '==' operator. This will also implicitly define the '!=' operator."""
        if isinstance(other, self.__class__):
            return self.wire_list == other.wire_list
        return False

    def __hash__(self):
        """Implements the hash function, used for example by ``set()``."""
        # hash the string representation
        return hash(str(repr(self)))

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

    def index(self, wire):
        """Overwrites a Sequences ``index()`` function which returns the index of ``wire``.

        Args:
            wire (Any): Object whose index is to be found. If this is a Wires object of length 1, look for the object
                representing the wire.

        Returns:
            int: index of the input
        """

        if isinstance(wire, Wires):
            if len(wire) != 1:
                raise WireError("Can only retrieve index of a Wires object of length 1.")

            return self.wire_list.index(wire.wire_list[0])

        return self.wire_list.index(wire)

    def indices(self, wires):
        """
        Return the indices of the wires in this Wires object.

        For example,

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([1, 4])
        >>> wires1.indices(wires2)
        [2, 0]
        >>> wires1.indices([1, 4])
        [2, 0]

        Args:
            wires (Wires or Iterable[Any]): Iterable containing the wires whose indices are to be found

        Returns:
            List: index list
        """

        return [self.index(w) for w in wires]

    def subset(self, indices, periodic_boundary=False):
        """
        Returns a new Wires object which is a subset of this Wires object. The wires of the new
        object are the wires at positions specified by 'indices'. Also accepts a single index as input.

        For example:

        >>> wires = Wires([4, 0, 1, 5, 6])
        >>> wires.subset([2, 3, 0])
        <Wires =[1, 5, 4]>

        >>> wires.subset(1)
        <Wires = [0]>

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
    def shared_wires(list_of_wires):
        """Return only the wires that appear in each Wires object in the list.

        This is similar to a set intersection method, but keeps the order of wires as they appear in the list.

        For example:

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([3, 0, 4])
        >>> wires3 = Wires([4, 0])
        >>> Wires.shared_wires([wires1, wires2, wires3])
        <Wires [4, 0]>

        >>> Wires.shared_wires([wires2, wires1, wires3])
        <Wires [0, 4]>

        Args:
            list_of_wires (List[Wires]): List of Wires objects

        Returns:
            Wires: shared wires
        """

        for wires in list_of_wires:
            if not isinstance(wires, Wires):
                raise WireError(
                    "expected a `pennylane.wires.Wires` object; got {} of type {}".format(
                        wires, type(wires)
                    )
                )

        shared = []
        # only need to iterate through the first object,
        # since any wire not in this object will also not be shared
        for wire in list_of_wires[0]:
            if all(wire in wires_ for wires_ in list_of_wires):
                shared.append(wire)

        return Wires.merge(shared)

    @staticmethod
    def all_wires(list_of_wires):
        """Return the wires that appear in any of the Wires objects in the list.

        This is similar to a set combine method, but keeps the order of wires as they appear in the list.

        For example:

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([3, 0, 4])
        >>> wires3 = Wires([5, 3])
        >>> list_of_wires = [wires1, wires2, wires3]
        >>> Wires.all_wires(list_of_wires)
        <Wires = [4, 0, 1, 3, 5]>

        Args:
            list_of_wires (List[Wires]): List of Wires objects

        Returns:
            Wires: combined wires
        """

        combined = []
        for wires in list_of_wires:
            if not isinstance(wires, Wires):
                raise WireError(
                    "expected a `pennylane.wires.Wires` object; got {} of type {}".format(
                        wires, type(wires)
                    )
                )

            combined.extend(wire for wire in wires.wire_list if wire not in combined)

        return Wires(combined)

    @staticmethod
    def unique_wires(list_of_wires):
        """Return the wires that are unique to any Wire object in the list.

        For example:

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([0, 2, 3])
        >>> wires3 = Wires([5, 3])
        >>> list_of_wires = [wires1, wires2, wires3]
        >>> Wires.unique_wires(list_of_wires)
        <Wires = [4, 2, 5]>

        Args:
            list_of_wires (List[Wires]): List of Wires objects

        Returns:
            Wires: unique wires
        """

        for wires in list_of_wires:
            if not isinstance(wires, Wires):
                raise WireError(
                    "expected a `pennylane.wires.Wires` object; got {} of type {}".format(
                        wires, type(wires)
                    )
                )

        unique = []
        for wires in list_of_wires:
            for wire in wires:
                # check that wire is only contained in one of the Wires objects
                if sum([1 for wires_ in list_of_wires if wire in wires_]) == 1:
                    unique.append(wire)

        return Wires.merge(unique)

    @staticmethod
    def merge(list_of_wires):
        """Merge Wires objects in list to one. All Wires objects in the list must contain unique wires.

        Args:
            list_of_wires (List[Wires]): list of Wires objects

        Return:
            Wires: new Wires object that contains all wires of the list's Wire objects

        Raises:
            WireError: for incorrect input or if any two Wires objects contain the same wires
        """

        merged_wires = []
        for wires in list_of_wires:

            if not isinstance(wires, Wires):
                raise WireError(
                    "Expected list of Wires objects; got entry {} of type {}.".format(
                        wires, type(wires)
                    )
                )

            if any([w in merged_wires for w in wires.wire_list]):
                raise WireError(
                    "Cannot merge Wires objects that contain the same wires; got {}.".format(
                        list_of_wires
                    )
                )
            else:
                merged_wires += wires.wire_list

        return Wires(merged_wires)

    @staticmethod
    def all_unique(list_of_wires):
        """Check whether all wires in the Wire objects in the list contain only unique wires.

        For example:

        >>> list_of_wires = [Wires([4, 0]), Wires([2, 5])]
        >>> Wires.all_unique(list_of_wires)
        True
        >>> list_of_wires = [Wires([4, 0, 1]), Wires([4, 2])]
        >>> Wires.all_unique(list_of_wires)
        False

        Args:
            list_of_wires (List[Wires]): List of Wires objects

        Returns:
            bool: whether list only contains unique wires
        """

        all_wires = []
        for wires in list_of_wires:
            if not isinstance(wires, Wires):
                raise WireError(
                    "expected a `pennylane.wires.Wires` object; got {} of type {}".format(
                        wires, type(wires)
                    )
                )
            all_wires.append(wires.as_list())

        for wires in list_of_wires:
            for wire in wires:
                # check that wire is only contained in one of the Wires objects
                if sum([1 for wires_ in list_of_wires if wire in wires_]) > 1:
                    return False

        return True