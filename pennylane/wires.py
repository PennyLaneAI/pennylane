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
This module contains the :class:`Wires` class, which takes care of wire bookkeeping.
"""
import functools
import itertools
from collections.abc import Iterable, Sequence
import numpy as np
from pennylane.pytrees import register_pytree


class WireError(Exception):
    """Exception raised by a :class:`~.pennylane.wires.Wire` object when it is unable to process wires."""


def _process(wires):
    """Converts the input to a tuple of wire labels.

    If `wires` can be iterated over, its elements are interpreted as wire labels
    and turned into a tuple. Otherwise, `wires` is interpreted as a single wire label.

    The only exception to this are strings, which are always interpreted as a single
    wire label, so users can address wires with labels such as `"ancilla"`.

    Any type can be a wire label, as long as it is hashable. We need this to establish
    the uniqueness of two labels. For example, `0` and `0.` are interpreted as
    the same wire label because `hash(0.) == hash(0)` evaluates to true.

    Note that opposed to numpy arrays, `pennylane.numpy` 0-dim array are hashable.
    """

    if isinstance(wires, str):
        # Interpret string as a non-iterable object.
        # This is the only exception to the logic
        # of considering the elements of iterables as wire labels.
        wires = [wires]

    try:
        # Use tuple conversion as a check for whether `wires` can be iterated over.
        # Note, this is not the same as `isinstance(wires, Iterable)` which would
        # pass for 0-dim numpy arrays that cannot be iterated over.
        tuple_of_wires = tuple(wires)
    except TypeError:
        # if not iterable, interpret as single wire label
        try:
            hash(wires)
        except TypeError as e:
            # if object is not hashable, cannot identify unique wires
            if str(e).startswith("unhashable"):
                raise WireError(f"Wires must be hashable; got object of type {type(wires)}.") from e
        return (wires,)

    try:
        # We need the set for the uniqueness check,
        # so we can use it for hashability check of iterables.
        set_of_wires = set(wires)
    except TypeError as e:
        if str(e).startswith("unhashable"):
            raise WireError(f"Wires must be hashable; got {wires}.") from e

    if len(set_of_wires) != len(tuple_of_wires):
        raise WireError(f"Wires must be unique; got {wires}.")

    return tuple_of_wires


class Wires(Sequence):
    r"""
    A bookkeeping class for wires, which are ordered collections of unique objects.

    If the input `wires` can be iterated over, it is interpreted as a sequence of wire labels that have to be
    unique and hashable. Else it is interpreted as a single wire label that has to be hashable. The
    only exception are strings which are interpreted as wire labels.

    The hash function of a wire label is considered the source of truth when deciding whether
    two wire labels are the same or not.

    Indexing an instance of this class will return a wire label.

    .. warning::

        In order to support wire labels of any hashable type, integers and 0-d arrays are considered different.
        For example, running ``qml.RX(1.1, qml.numpy.array(0))`` on a device initialized with ``wires=[0]``
        will fail because ``qml.numpy.array(0)`` does not exist in the device's wire map.

    Args:
         wires (Any): the wire label(s)
    """

    def _flatten(self):
        """Serialize Wires into a flattened representation according to the PyTree convension."""
        return self._labels, ()

    @classmethod
    def _unflatten(cls, data, _metadata):
        """De-serialize flattened representation back into the Wires object."""
        return cls(data, _override=True)

    def __init__(self, wires, _override=False):
        if _override:
            self._labels = wires
        else:
            self._labels = _process(wires)

        self._hash = None

    def __getitem__(self, idx):
        """Method to support indexing. Returns a Wires object if index is a slice,
        or a label if index is an integer."""
        if isinstance(idx, slice):
            return Wires(self._labels[idx])
        return self._labels[idx]

    def __iter__(self):
        return self._labels.__iter__()

    def __len__(self):
        """Method to support ``len()``."""
        return len(self._labels)

    def contains_wires(self, wires):
        """Method to determine if Wires object contains wires in another Wires object."""
        if isinstance(wires, Wires):
            return set(wires.labels).issubset(set(self._labels))
        return False

    def __contains__(self, item):
        """Method checking if Wires object contains an object."""
        return item in self._labels

    def __repr__(self):
        """Method defining the string representation of this class."""
        return f"<Wires = {list(self._labels)}>"

    def __eq__(self, other):
        """Method to support the '==' operator.
        This will also implicitly define the '!=' operator."""
        # The order is respected in comparison, so that ``assert Wires([0, 1]) != Wires([1,0])``
        if isinstance(other, Wires):
            return self._labels == other.labels
        return self._labels == other

    def __hash__(self):
        """Implements the hash function."""
        if self._hash is None:
            self._hash = hash(self._labels)
        return self._hash

    def __add__(self, other):
        """Defines the addition to return a Wires object containing all wires of the two terms.

        Args:
            other (Iterable[Number,str], Number, Wires): object to add from the right

        Returns:
            Wires: all wires appearing in either object

        **Example**

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([1, 2])
        >>> wires1 + wires2
        Wires([4, 0, 1, 2])
        """
        other = Wires(other)
        return Wires.all_wires([self, other])

    def __radd__(self, other):
        """Defines addition according to __add__ if the left object has no addition defined.

        Args:
            other (Iterable[Number,str], Number, Wires): object to add from the left

        Returns:
            Wires: all wires appearing in either object
        """
        other = Wires(other)
        return Wires.all_wires([other, self])

    def __array__(self):
        """Defines a numpy array representation of the Wires object.

        Returns:
            ndarray: array representing Wires object
        """
        return np.array(self._labels)

    @property
    def labels(self):
        """Get a tuple of the labels of this Wires object."""
        return self._labels

    def toarray(self):
        """Returns a numpy array representation of the Wires object.

        Returns:
            ndarray: array representing Wires object
        """
        return np.array(self._labels)

    def tolist(self):
        """Returns a list representation of the Wires object.

        Returns:
            List: list of wire labels
        """
        return list(self._labels)

    def toset(self):
        """Returns a set representation of the Wires object.

        Returns:
            Set: set of wire labels
        """
        return set(self.labels)

    def index(self, wire):
        """Overwrites a Sequence's ``index()`` function which returns the index of ``wire``.

        Args:
            wire (Any): Object whose index is to be found. If this is a Wires object of length 1, look for the object
                representing the wire.

        Returns:
            int: index of the input
        """
        # pylint: disable=arguments-differ

        if isinstance(wire, Wires):
            if len(wire) != 1:
                raise WireError("Can only retrieve index of a Wires object of length 1.")

            wire = wire[0]

        try:
            return self._labels.index(wire)
        except ValueError as e:
            raise WireError(f"Wire with label {wire} not found in {self}.") from e

    def indices(self, wires):
        """
        Return the indices of the wires in this Wires object.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): Wire(s) whose indices are to be found

        Returns:
            List: index list

        **Example**

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([1, 4])
        >>> wires1.indices(wires2)
        [2, 0]
        >>> wires1.indices([1, 4])
        [2, 0]
        """
        if not isinstance(wires, Iterable):
            return [self.index(wires)]

        return [self.index(w) for w in wires]

    def map(self, wire_map):
        """Returns a new Wires object with different labels, using the rule defined in mapping.

        Args:
            wire_map (dict): Dictionary containing all wire labels used in this object as keys, and unique
                             new labels as their values
        **Example**

        >>> wires = Wires(['a', 'b', 'c'])
        >>> wire_map = {'a': 4, 'b':2, 'c': 3}
        >>> wires.map(wire_map)
        <Wires = [4, 2, 3]>
        """
        # Make sure wire_map has `Wires` keys and values so that the `in` operator always works

        for w in self:
            if w not in wire_map:
                raise WireError(f"No mapping for wire label {w} specified in wire map {wire_map}.")

        new_wires = [wire_map[w] for w in self]

        try:
            new_wires = Wires(new_wires)
        except WireError as e:
            raise WireError(
                f"Failed to implement wire map {wire_map}. Make sure that the new labels "
                f"are unique and valid wire labels."
            ) from e

        return new_wires

    def subset(self, indices, periodic_boundary=False):
        """
        Returns a new Wires object which is a subset of this Wires object. The wires of the new
        object are the wires at positions specified by 'indices'. Also accepts a single index as input.

        Args:
            indices (List[int] or int): indices or index of the wires we want to select
            periodic_boundary (bool): controls periodic boundary conditions in the indexing

        Returns:
            Wires: subset of wires

        **Example**

        >>> wires = Wires([4, 0, 1, 5, 6])
        >>> wires.subset([2, 3, 0])
        <Wires = [1, 5, 4]>
        >>> wires.subset(1)
        <Wires = [0]>

        If ``periodic_boundary`` is True, the modulo of the number of wires of an index is used instead of an index,
        so that  ``wires.subset(i) == wires.subset(i % n_wires)`` where ``n_wires`` is the number of wires of this
        object.

        >>> wires = Wires([4, 0, 1, 5, 6])
        >>> wires.subset([5, 1, 7], periodic_boundary=True)
        <Wires = [4, 0, 1]>

        """

        if isinstance(indices, int):
            indices = [indices]

        if periodic_boundary:
            # replace indices by their modulo
            indices = [i % len(self._labels) for i in indices]

        for i in indices:
            if i > len(self._labels):
                raise WireError(f"Cannot subset wire at index {i} from {len(self._labels)} wires.")

        subset = tuple(self._labels[i] for i in indices)
        return Wires(subset, _override=True)

    def select_random(self, n_samples, seed=None):
        """
        Returns a randomly sampled subset of Wires of length 'n_samples'.

        Args:
            n_samples (int): number of subsampled wires
            seed (int): optional random seed used for selecting the wires

        Returns:
            Wires: random subset of wires
        """

        if n_samples > len(self._labels):
            raise WireError(f"Cannot sample {n_samples} wires from {len(self._labels)} wires.")

        rng = np.random.default_rng(seed)

        indices = rng.choice(len(self._labels), size=n_samples, replace=False)
        subset = tuple(self[i] for i in indices)
        return Wires(subset, _override=True)

    @staticmethod
    def shared_wires(list_of_wires):
        """Return only the wires that appear in each Wires object in the list.

        This is similar to a set intersection method, but keeps the order of wires as they appear in the list.

        Args:
            list_of_wires (List[Wires]): list of Wires objects

        Returns:
            Wires: shared wires

        **Example**

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([3, 0, 4])
        >>> wires3 = Wires([4, 0])
        >>> Wires.shared_wires([wires1, wires2, wires3])
        <Wires = [4, 0]>
        >>> Wires.shared_wires([wires2, wires1, wires3])
        <Wires = [0, 4]>
        """

        for wires in list_of_wires:
            if not isinstance(wires, Wires):
                raise WireError(f"Expected a Wires object; got {wires} of type {type(wires)}.")

        sets_of_wires = [wire.toset() for wire in list_of_wires]
        # find the intersection of the labels of all wires in O(n) time.
        intersecting_wires = functools.reduce(lambda a, b: a & b, sets_of_wires)
        shared = []
        # only need to iterate through the first object,
        # since any wire not in this object will also not be shared
        for wire in list_of_wires[0]:
            if wire in intersecting_wires:
                shared.append(wire)

        return Wires(tuple(shared), _override=True)

    @staticmethod
    def all_wires(list_of_wires, sort=False):
        """Return the wires that appear in any of the Wires objects in the list.

        This is similar to a set combine method, but keeps the order of wires as they appear in the list.

        Args:
            list_of_wires (List[Wires]): List of Wires objects
            sort (bool): Toggle for sorting the combined wire labels. The sorting is based on
                value if all keys are int, else labels' str representations are used.

        Returns:
            Wires: combined wires

        **Example**

        >>> wires1 = Wires([4, 0, 1])
        >>> wires2 = Wires([3, 0, 4])
        >>> wires3 = Wires([5, 3])
        >>> list_of_wires = [wires1, wires2, wires3]
        >>> Wires.all_wires(list_of_wires)
        <Wires = [4, 0, 1, 3, 5]>
        """
        converted_wires = (
            wires if isinstance(wires, Wires) else Wires(wires) for wires in list_of_wires
        )
        all_wires_list = itertools.chain(*(w.labels for w in converted_wires))
        combined = list(dict.fromkeys(all_wires_list))

        if sort:
            if all(isinstance(w, int) for w in combined):
                combined = sorted(combined)
            else:
                combined = sorted(combined, key=str)

        return Wires(tuple(combined), _override=True)

    @staticmethod
    def unique_wires(list_of_wires):
        """Return the wires that are unique to any Wire object in the list.

        Args:
            list_of_wires (List[Wires]): list of Wires objects

        Returns:
            Wires: unique wires

        **Example**

        >>> wires1 = Wires([4, 0, 1])
        >>> wires2 = Wires([0, 2, 3])
        >>> wires3 = Wires([5, 3])
        >>> Wires.unique_wires([wires1, wires2, wires3])
        <Wires = [4, 1, 2, 5]>
        """

        for wires in list_of_wires:
            if not isinstance(wires, Wires):
                raise WireError(f"Expected a Wires object; got {wires} of type {type(wires)}.")

        label_sets = [wire.toset() for wire in list_of_wires]
        seen_ever = set()
        seen_once = set()

        # Find unique set in O(n) time.
        for labels in label_sets:
            # (seen_once ^ labels) finds all of the unique labels seen once
            # (seen_ever - seen_once) is the set of labels already seen more than once
            # Subtracting these two sets makes a set of labels only seen once so far.
            seen_once = (seen_once ^ labels) - (seen_ever - seen_once)
            # Update seen labels with all new seen labels
            seen_ever.update(labels)

        # Get unique values in order they appear.
        unique = []
        for wires in list_of_wires:
            for wire in wires.tolist():
                # check that wire is only contained in one of the Wires objects
                if wire in seen_once:
                    unique.append(wire)

        return Wires(tuple(unique), _override=True)


# Register Wires as a PyTree-serializable class
register_pytree(Wires, Wires._flatten, Wires._unflatten)  # pylint: disable=protected-access
