# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Fermionic representation classes"""
from copy import copy
from pennylane.pauli import PauliWord, PauliSentence
from numbers import Number
from pennylane.numpy.tensor import tensor


class FermiWord(dict):
    """Immutable dictionary used to represent a Fermi word, a product of fermionic creation and
    annihilation operators, associating wires with their respective operators. Can be constructed
    from a standard dictionary.

    >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w
    '0+ 1-'
    """

    def __init__(self, mapping):
        self.sorted_dic = dict(sorted(mapping.items()))
        super().__init__(mapping)

    def __copy__(self):
        """Copy the FermiWord instance."""
        return FermiWord(dict(self.items()))

    @property
    def wires(self):
        """Wires in a FermiWord."""
        return [i[1] for i in self.sorted_dic.keys()]

    def name(self):
        return self.__class__.__name__

    def to_string(self):
        """String representation of the fermionic operator."""
        string = " ".join(
            [
                i + j
                for i, j in zip(
                    [str(i[1]) for i in self.sorted_dic.keys()], self.sorted_dic.values()
                )
            ]
        )
        return string

    def update(self, item):
        """Restrict updating FermiWord after instantiation."""
        raise TypeError("FermiWord object does not support assignment")

    def __setitem__(self, key, item):
        """Restrict setting items after instantiation."""
        raise TypeError("FermiWord object does not support assignment")

    def __hash__(self):
        """Hash value of a FermiWord."""
        return hash(frozenset(self.items()))

    def __str__(self):
        """String representation of a FermiWord."""
        return f"<Operator = '{self.to_string()}', Wire: {self.wires}, Rep: {self.name()}>"

    def __repr__(self):
        """Terminal representation of a FermiWord"""
        return str(self)

    def __mul__(self, other):
        """Multiply two Ferm words together."""

        if isinstance(other, FermiWord) or isinstance(other, FermiC) or isinstance(other, FermiA):
            if len(self) == 0:
                return copy(other)

            if len(other) == 0:
                return copy(self)

            order_other = [i[0] for i in other.sorted_dic.keys()]
            order_final = [
                i[0] + max([i[0] for i in self.sorted_dic.keys()]) + 1
                for i in other.sorted_dic.keys()
            ]

            dict_other = dict(
                zip([(order_final[o], other.wires[o]) for o in order_other], other.values())
            )
            dict_self = dict(zip(self.keys(), self.values()))

            dict_self.update(dict_other)

            return FermiWord(dict_self)

        elif isinstance(other, Number) or isinstance(other, tensor):
            return FermiSentence({self: other})

        raise TypeError("Cannot multiply FermiWord by ...")

    def __rmul__(self, other):
        """Multiply two Ferm words together."""

        if isinstance(other, Number):
            return FermiSentence({self: other})

        if isinstance(other, tensor):
            return FermiSentence({self: other})

        raise TypeError("__rmul__ Cannot multiply FermiWord by ...")

    def to_qubit(self):
        """Map to qubit."""
        return mapping(self)

    def __add__(self, other):
        """Add two Fermi two Ferm words together."""
        if self == other:
            return FermiSentence({self: 2.0})
        return FermiSentence({self: 1.0, other: 1.0})

    def __pow__(self, value):
        if not isinstance(value, int):
            raise TypeError("The exponent must be integer.")

        operator = FermiWord({})

        for _ in range(value):
            operator *= self

        return operator
