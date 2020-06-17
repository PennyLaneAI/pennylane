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
import numpy as np
from numbers import Number


class WireError(Exception):
    """Exception raised by a :class:`~.pennylane.wires.Wire` object when it is unable to process wires.
    """


def _process(wires):
    """Converts the input to a tuple of numbers or strings."""

    if isinstance(wires, Wires):
        # if input is already a Wires object, just return its wire tuple
        return wires.wire_tuple

    elif isinstance(wires, (Number, str)):
        # interpret as a single wire
        return (wires,)

    elif isinstance(wires, Iterable) and all(isinstance(w, Wires) for w in wires):
        # if the elements are themselves Wires objects, merge them to a new one
        return tuple(w for wires_ in wires for w in wires_.tolist())

    elif isinstance(wires, Iterable) and all(
        isinstance(w, str) or isinstance(w, Number) for w in wires
    ):
        # if the elements are strings or numbers, turn iterable into tuple
        return tuple(wires)

    else:
        raise WireError(
            "Wires must be represented by a number or string; got {} of type {}.".format(
                wires, type(wires)
            )
        )


class Wires(Sequence):
    r"""
    A bookkeeping class for wires, which are ordered collections of unique objects. The :math:`i\mathrm{th}` object
    addresses the :math:`i\mathrm{th}` quantum subsystem.

    There is no conceptual difference between registers of multiple wires and single wires,
    which are just wire registers of length one.

    Indexing and slicing this sequence will return another ``Wires`` object.

    Args:
         wires (Iterable[Number,str], Number): If iterable, interpreted as an ordered collection of unique objects
            representing wires. If a Number, the input is converted into an iterable of a single entry,
            and hence interpreted as a single wire.
    """

    def __init__(self, wires):

        self.wire_tuple = _process(wires)

        # check that all wires are unique
        if len(set(self.wire_tuple)) != len(self.wire_tuple):
            raise WireError("Wires must be unique; got {}.".format(wires))

    def __getitem__(self, idx):
        """Method to support indexing. Returns a Wires object representing a register with a single wire."""
        return Wires(self.wire_tuple[idx])

    def __len__(self):
        """Method to support ``len()``."""
        return len(self.wire_tuple)

    def __contains__(self, item):
        """Method checking if Wires object contains an object."""
        if isinstance(item, Wires):
            item = item.tolist()
        # if all wires can be found in tuple, return True, else False
        return all(wire in self.wire_tuple for wire in item)

    def __repr__(self):
        """Method defining the string representation of this class."""
        return "<Wires = {}>".format(list(self.wire_tuple))

    def __eq__(self, other):
        """Method to support the '==' operator. This will also implicitly define the '!=' operator."""
        # The order is respected in comparison, so that ``assert Wires([0, 1]) != Wires([1,0])``
        if isinstance(other, self.__class__):
            return self.wire_tuple == other.wire_tuple
        return False

    def __hash__(self):
        """Implements the hash function."""
        return hash(repr(self.wire_tuple))

    def toarray(self):
        """Returns a numpy array representation of the Wires object.

        Returns:
            ndarray: array representing Wires object
        """
        return np.array(self.wire_tuple)

    def tolist(self):
        """Returns a list representation of the Wires object.

        Returns:
            List: list representing Wires object
        """
        return list(self.wire_tuple)

    def get_label(self, idx):
        """Returns the wire label at the given position in the wires object.

        >>> w = Wires([0, 'q1', 16])
        >>> w.get_label(1)
        'q1'
        >>> w.get_label(2)
        16

        Args:
            int: index of wire to return

        Returns:
            Number or str: label of the wire
        """
        return self.wire_tuple[idx]

    def index(self, wire):
        """Overwrites a Sequence's ``index()`` function which returns the index of ``wire``.

        Args:
            wire (Any): Object whose index is to be found. If this is a Wires object of length 1, look for the object
                representing the wire.

        Returns:
            int: index of the input
        """

        if isinstance(wire, Wires):
            if len(wire) != 1:
                raise WireError("Can only retrieve index of a Wires object of length 1.")

            return self.wire_tuple.index(wire.wire_tuple[0])

        return self.wire_tuple.index(wire)

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
        <Wires = [1, 5, 4]>
        >>> wires.subset(1)
        <Wires = [0]>

        If ``periodic_boundary`` is True, the modulo of the number of wires of an index is used instead of an index,
        so that  ``wires.subset(i) == wires.subset(i % n_wires)`` where ``n_wires`` is the number of wires of this
        object.

        For example:

        >>> wires = Wires([4, 0, 1, 5, 6])
        >>> wires.subset([5, 1, 7])
        <Wires = [4, 0, 1]>

        Args:
            indices (List[int] or int): indices or index of the wires we want to select
            periodic_boundary (bool): controls periodic boundary conditions in the indexing

        Returns:
            Wires: subset of wires
        """

        if isinstance(indices, int):
            indices = [indices]

        if periodic_boundary:
            # replace indices by their modulo
            indices = [i % len(self.wire_tuple) for i in indices]

        for i in indices:
            if i > len(self.wire_tuple):
                raise WireError(
                    "Cannot subset wire at index {} from {} wires.".format(i, len(self.wire_tuple))
                )

        subset = [self.wire_tuple[i] for i in indices]
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

        if n_samples > len(self.wire_tuple):
            raise WireError(
                "Cannot sample {} wires from {} wires.".format(n_samples, len(self.wire_tuple))
            )

        if seed is not None:
            np.random.seed(seed)

        indices = np.random.choice(len(self.wire_tuple), size=n_samples, replace=False)
        subset = [self.wire_tuple[i] for i in indices]
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
        <Wires = [4, 0]>
        >>> Wires.shared_wires([wires2, wires1, wires3])
        <Wires = [0, 4]>

        Args:
            list_of_wires (List[Wires]): list of Wires objects

        Returns:
            Wires: shared wires
        """

        for wires in list_of_wires:
            if not isinstance(wires, Wires):
                raise WireError(
                    "Expected a Wires object; got {} of type {}.".format(wires, type(wires))
                )

        shared = []
        # only need to iterate through the first object,
        # since any wire not in this object will also not be shared
        for wire in list_of_wires[0]:
            if all(wire in wires_ for wires_ in list_of_wires):
                shared.append(wire)

        return Wires(shared)

    @staticmethod
    def all_wires(list_of_wires):
        """Return the wires that appear in any of the Wires objects in the list.

        This is similar to a set combine method, but keeps the order of wires as they appear in the list.

        For example:

        >>> wires1 = Wires([4, 0, 1])
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
                    "Expected a Wires object; got {} of type {}".format(wires, type(wires))
                )

            combined.extend(wire for wire in wires.wire_tuple if wire not in combined)

        return Wires(combined)

    @staticmethod
    def unique_wires(list_of_wires):
        """Return the wires that are unique to any Wire object in the list.

        For example:

        >>> wires1 = Wires([4, 0, 1])
        >>> wires2 = Wires([0, 2, 3])
        >>> wires3 = Wires([5, 3])
        >>> Wires.unique_wires([wires1, wires2, wires3])
        <Wires = [4, 2, 5]>

        Args:
            list_of_wires (List[Wires]): list of Wires objects

        Returns:
            Wires: unique wires
        """

        for wires in list_of_wires:
            if not isinstance(wires, Wires):
                raise WireError(
                    "Expected a Wires object; got {} of type {}.".format(wires, type(wires))
                )

        unique = []
        for wires in list_of_wires:
            for wire in wires:
                # check that wire is only contained in one of the Wires objects
                if sum([1 for wires_ in list_of_wires if wire in wires_]) == 1:
                    unique.append(wire)

        return Wires(unique)
