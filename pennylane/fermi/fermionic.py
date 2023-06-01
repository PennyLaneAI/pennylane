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
"""The Fermionic representation classes."""
from copy import copy


class FermiWord(dict):
    r"""Immutable dictionary used to represent a Fermi word, a product of fermionic creation and
    annihilation operators, associating wires with their respective operators. Can be constructed
    from a standard dictionary.

    >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w
    <FermiWord = '0+ 1-'>
    """

    def __init__(self, operator):
        self.sorted_dic = dict(sorted(operator.items()))
        super().__init__(operator)

    @property
    def wires(self):
        r"""Return wires in a FermiWord."""
        return [i[1] for i in self.sorted_dic.keys()]

    def __missing__(self, key):
        r"""Return empty string for a missing jey in FermiWord."""
        return ""

    def update(self, item):
        r"""Restrict updating FermiWord after instantiation."""
        raise TypeError("FermiWord object does not support assignment")

    def __setitem__(self, key, item):
        r"""Restrict setting items after instantiation."""
        raise TypeError("FermiWord object does not support assignment")

    def __reduce__(self):
        r"""Defines how to pickle and unpickle a FermiWord. Otherwise, un-pickling
        would cause __setitem__ to be called, which is forbidden on PauliWord.
        For more information, see: https://docs.python.org/3/library/pickle.html#object.__reduce__
        """
        return FermiWord, (dict(self),)

    def __copy__(self):
        r"""Copy the FermiWord instance."""
        return FermiWord(dict(self.items()))

    def __deepcopy__(self, memo):
        r"""Deep copy the FermiWord instance."""
        res = self.__copy__()
        memo[id(self)] = res
        return res

    def __hash__(self):
        r"""Hash value of a FermiWord."""
        return hash(frozenset(self.items()))

    def to_string(self):
        r"""Return a compact string representation of a FermiWord.

        >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
        >>> w.to_string()
        '0+ 1-'
        """
        string = " ".join(
            [
                i + j
                for i, j in zip(
                    [str(i[1]) for i in self.sorted_dic.keys()], self.sorted_dic.values()
                )
            ]
        )
        return string

    def __str__(self):
        r"""String representation of a FermiWord."""
        return f"<FermiWord = '{self.to_string()}'>"

    def __repr__(self):
        r"""Terminal representation of a FermiWord"""
        return str(self)

    def __mul__(self, other):
        r"""Multiply two Fermi words together.

        >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
        >>> w * w
        <FermiWord = '0+ 1- 0+ 1-'>
        """

        if isinstance(other, FermiWord):
            if len(self) == 0:
                return copy(other)

            if len(other) == 0:
                return copy(self)

            order_other = [i[0] for i in other.sorted_dic.keys()]
            order_final = [
                i[0] + max(i[0] for i in self.sorted_dic.keys()) + 1
                for i in other.sorted_dic.keys()
            ]

            dict_other = dict(
                zip([(order_final[o], other.wires[o]) for o in order_other], other.values())
            )
            dict_self = dict(zip(self.keys(), self.values()))

            dict_self.update(dict_other)

            return FermiWord(dict_self)

        raise TypeError(f"Cannot multiply FermiWord by {type(other)}.")

    def __pow__(self, value):
        r"""Exponentiate a Fermi word to an integer power.

        >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
        >>> w**3
        <FermiWord = '0+ 1- 0+ 1- 0+ 1-'>
        """

        if value < 0 or not isinstance(value, int):
            raise ValueError("The exponent must be a positive integer.")

        operator = FermiWord({})

        for _ in range(value):
            operator *= self

        return operator


# TODO: create __add__ method when PauliSentence is merged.
# TODO: support multiply by number in __mul__ when PauliSentence is merged.
# TODO: create mapping method when the ne jordan_wigner function is added.


class FermiC(FermiWord):
    r"""FermiC(wire)
    The fermionic creation operator

    .. math:: c^\dagger \ket{n} = \sqrt(n + 1) \ket{n+1}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wire(int): the positve integer indicating the energy level the operator acts on. The lowest
            energy level is designated 0, and subsequent energy levels are numbered chronologically.

    .. note:: While the `FermiC` class represents a mathematical operator, it is not a PennyLane `Operator`

    **Example**

    >>> FermiC(0)
    <FermiWord = '0+'>
    """

    def __init__(self, wire):

        if not isinstance(wire, int) or wire < 0:
            raise ValueError(
                f"FermiC: expected a single, positive integer value for wire, but received {wire}"
            )
        operator = {(0, wire): "+"}
        self.sorted_dic = dict(sorted(operator.items()))
        super().__init__(operator)

    # TODO: create mapping method when the tne jordan_wigner function is added.
    # def to_qubit(self):
    #     return mapping(self)


class FermiA(FermiWord):
    r"""FermiA(wire)
    The fermionic annihilation operator

    .. math:: c \ket{n} = \sqrt(n) \ket{n-1}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wire(int): the positve integer indicating the energy level the operator acts on. The lowest
            energy level is designated 0, and subsequent energy levels are numbered chronologically.

    .. note:: While the `FermiA` class represents a mathematical operator, it is not a PennyLane `Operator`

    **Example**

    >>> FermiA(0)
    <FermiWord = '0-'>
    """

    def __init__(self, wire):

        if not isinstance(wire, int) or wire < 0:
            raise ValueError(
                f"FermiA: expected a single, positive integer value for wire, but received {wire}"
            )
        operator = {(0, wire): "-"}
        self.sorted_dic = dict(sorted(operator.items()))
        super().__init__(operator)

    # TODO: create mapping method when the tne jordan_wigner function is added.
    # def to_qubit(self):
    #     return mapping(self)
