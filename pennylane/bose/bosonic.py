# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The bosonic representation classes and functions."""
from copy import copy

from pennylane import math
from pennylane.typing import TensorLike


class BoseWord(dict):
    r"""Dictionary used to represent a Bose word, a product of bosonic creation and
    annihilation operators, that can be constructed from a standard dictionary.

    The keys of the dictionary are tuples of two integers. The first integer represents the
    position of the creation/annihilation operator in the Bose word and the second integer
    represents the mode it acts on. The values of the dictionary are one of ``'+'`` or ``'-'``
    symbols that denote creation and annihilation operators, respectively. The operator
    :math:`b^{\dagger}_0 b_1` can then be constructed as

    >>> w = qml.BoseWord({(0, 0) : '+', (1, 1) : '-'})
    >>> print(w)
    b⁺(0) b(1)
    """

    # override the arithmetic dunder methods for numpy arrays so that the
    # methods defined on this class are used instead
    # (i.e. ensure `np.array + BoseWord` uses `BoseWord.__radd__` instead of `np.array.__add__`)
    __numpy_ufunc__ = None
    __array_ufunc__ = None

    def __init__(self, operator):
        self.sorted_dic = dict(sorted(operator.items()))
        indices = [i[0] for i in self.sorted_dic.keys()]

        if indices:
            if list(range(max(indices) + 1)) != indices:
                raise ValueError(
                    "The operator indices must belong to the set {0, ..., len(operator)-1}."
                )

        super().__init__(operator)

    def adjoint(self):
        r"""Return the adjoint of BoseWord."""
        n = len(self.items())
        adjoint_dict = {}
        for key, value in reversed(self.items()):
            position = n - key[0] - 1
            orbital = key[1]
            bose = "+" if value == "-" else "-"
            adjoint_dict[(position, orbital)] = bose

        return BoseWord(adjoint_dict)

    def items(self):
        """Returns the dictionary items in sorted order."""
        return self.sorted_dic.items()

    @property
    def wires(self):
        r"""Return wires in a BoseWord."""
        return {i[1] for i in self.sorted_dic.keys()}

    def __missing__(self, key):
        r"""Return empty string for a missing key in BoseWord."""
        return ""

    def update(self, item):
        r"""Restrict updating BoseWord after instantiation."""
        raise TypeError("BoseWord object does not support assignment")

    def __setitem__(self, key, item):
        r"""Restrict setting items after instantiation."""
        raise TypeError("BoseWord object does not support assignment")

    def __reduce__(self):
        r"""Defines how to pickle and unpickle a BoseWord. Otherwise, un-pickling
        would cause __setitem__ to be called, which is forbidden on PauliWord.
        For more information, see: https://docs.python.org/3/library/pickle.html#object.__reduce__
        """
        return BoseWord, (dict(self),)

    def __copy__(self):
        r"""Copy the BoseWord instance."""
        return BoseWord(dict(self.items()))

    def __deepcopy__(self, memo):
        r"""Deep copy the BoseWord instance."""
        res = self.__copy__()
        memo[id(self)] = res
        return res

    def __hash__(self):
        r"""Hash value of a BoseWord."""
        return hash(frozenset(self.items()))

    def to_string(self):
        r"""Return a compact string representation of a BoseWord. Each operator in the word is
        represented by the number of the wire it operates on, and a `+` or `-` to indicate either
        a creation or annihilation operator.

        >>> w = qml.BoseWord({(0, 0) : '+', (1, 1) : '-'})
        >>> w.to_string()
        'b⁺(0) b(1)'
        """
        if len(self) == 0:
            return "I"

        symbol_map = {"+": "\u207a", "-": ""}

        string = " ".join(
            [
                "b" + symbol_map[j] + "(" + i + ")"
                for i, j in zip(
                    [str(i[1]) for i in self.sorted_dic.keys()], self.sorted_dic.values()
                )
            ]
        )
        return string

    def __str__(self):
        r"""String representation of a BoseWord."""
        return f"{self.to_string()}"

    def __repr__(self):
        r"""Terminal representation of a BoseWord"""
        return f"BoseWord({self.sorted_dic})"

    def __add__(self, other):
        """Add a BoseSentence, BoseWord or constant to a BoseWord. Converts both
        elements into BoseSentences, and uses the BoseSentence __add__
        method"""

        self_bs = BoseSentence({self: 1.0})

        if isinstance(other, BoseSentence):
            return self_bs + other

        if isinstance(other, BoseWord):
            return self_bs + BoseSentence({other: 1.0})

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot add {type(other)} to a BoseWord.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Bose operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        return self_bs + BoseSentence({BoseWord({}): other})

    def __radd__(self, other):
        """Add a BoseWord to a constant, i.e. `2 + BoseWord({...})`"""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract a BoseSentence, BoseWord or constant from a BoseWord. Converts both
        elements into BoseSentences (with negative coefficient for `other`), and
        uses the BoseSentence __add__  method"""

        self_bs = BoseSentence({self: 1.0})

        if isinstance(other, BoseWord):
            return self_bs + BoseSentence({other: -1.0})

        if isinstance(other, BoseSentence):
            other_bs = BoseSentence(dict(zip(other.keys(), [-v for v in other.values()])))
            return self_bs + other_bs

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot subtract {type(other)} from a BoseWord.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Bose operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        return self_bs + BoseSentence({BoseWord({}): -1 * other})  # -constant * I

    def __rsub__(self, other):
        """Subtract a BoseWord to a constant, i.e. `2 - BoseWord({...})`"""
        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot subtract a BoseWord from {type(other)}.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Bose operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )
        self_bs = BoseSentence({self: -1.0})
        other_bs = BoseSentence({BoseWord({}): other})
        return self_bs + other_bs

    def __mul__(self, other):
        r"""Multiply a BoseWord with another BoseWord, a BoseSentence, or a constant.

        >>> w = qml.BoseWord({(0, 0) : '+', (1, 1) : '-'})
        >>> print(w * w)
        b⁺(0) b(1) b⁺(0) b(1)
        """

        if isinstance(other, BoseWord):
            if len(self) == 0:
                return copy(other)

            if len(other) == 0:
                return copy(self)

            order_final = [i[0] + len(self) for i in other.sorted_dic.keys()]
            other_wires = [i[1] for i in other.sorted_dic.keys()]

            dict_other = dict(
                zip(
                    [(order_idx, other_wires[i]) for i, order_idx in enumerate(order_final)],
                    other.values(),
                )
            )
            dict_self = dict(zip(self.keys(), self.values()))

            dict_self.update(dict_other)

            return BoseWord(dict_self)

        if isinstance(other, BoseSentence):
            return BoseSentence({self: 1}) * other

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot multiply BoseWord by {type(other)}.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Bose operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        return BoseSentence({self: other})

    def __rmul__(self, other):
        r"""Reverse multiply a BoseWord

        Multiplies a BoseWord "from the left" with an object that can't be modified
        to support __mul__ for BoseWord. Will be defaulted in for example
        ``2 * BoseWord({(0, 0): "+"})``, where the ``__mul__`` operator on an integer
        will fail to multiply with a BoseWord"""

        return self.__mul__(other)

    def __pow__(self, value):
        r"""Exponentiate a Bose word to an integer power.

        >>> w = qml.BoseWord({(0, 0) : '+', (1, 1) : '-'})
        >>> print(w**3)
        b⁺(0) b(1) b⁺(0) b(1) b⁺(0) b(1)
        """
        if value < 0 or not isinstance(value, int):
            raise ValueError("The exponent must be a positive integer.")

        operator = BoseWord({})

        for _ in range(value):
            operator *= self

        return operator

    def normal_order(self):
        r"""Convert a BoseWord to its normal-ordered form.

        >>> bw = qml.BoseWord({(0, 0): "-", (1, 0): "-", (2, 0): "+", (3, 0): "+"})
        >>> print(bw.normal_order())
        2.0 * I
        + 4.0 * b⁺(0) b(0)
        + 1.0 * b⁺(0) b⁺(0) b(0) b(0)
        """
        bw_terms = sorted(self)
        len_op = len(bw_terms)
        bw_comm = BoseSentence({BoseWord({}): 0.0})

        if len_op == 0:
            return 1 * BoseWord({})

        bw = self

        left_pointer = 0
        # The right pointer iterates through all operators in the BoseWord
        for right_pointer in range(len_op):
            # The right pointer finds the leftmost creation operator
            if self[bw_terms[right_pointer]] == "+":
                # This ensures that the left pointer starts at the leftmost annihilation term
                if left_pointer == right_pointer:
                    left_pointer += 1
                    continue

                # We shift the leftmost creation operator to the position of the left pointer
                bs = bw.shift_operator(right_pointer, left_pointer)
                bs_as_list = sorted(list(bs.items()), key=lambda x: len(x[0].keys()), reverse=True)
                bw = bs_as_list[0][0]

                for i in range(1, len(bs_as_list)):
                    bw_comm += bs_as_list[i][0] * bs_as_list[i][1]

                # Left pointer now points to the new leftmost annihilation term
                left_pointer += 1

        # Sort BoseWord by indice
        plus_terms = list(bw.items())[:left_pointer]
        minus_terms = list(bw.items())[left_pointer:]

        sorted_plus_terms = dict(sorted(plus_terms, key=lambda x: (x[0][1], x[0][0])))
        sorted_minus_terms = dict(sorted(minus_terms, key=lambda x: (x[0][1], x[0][0])))

        sorted_dict = {**sorted_plus_terms, **sorted_minus_terms}

        bw_sorted_by_index = {}
        for i, (k, v) in enumerate(sorted_dict.items()):
            bw_sorted_by_index[(i, k[1])] = v

        ordered_op = BoseWord(bw_sorted_by_index) + bw_comm.normal_order()
        ordered_op.simplify(tol=1e-8)
        return ordered_op

    def shift_operator(self, initial_position, final_position):
        r"""Shifts an operator in the BoseWord from ``initial_position`` to ``final_position`` by applying the bosonic commutation relations.

        Args:
            initial_position (int): the position of the operator to be shifted
            final_position (int): the desired position of the operator

        Returns:
            BoseSentence: The ``BoseSentence`` obtained after applying the commutator relations.

        Raises:
            TypeError: if ``initial_position`` or ``final_position`` is not an integer
            ValueError: if ``initial_position`` or ``final_position`` are outside the range ``[0, len(BoseWord) - 1]``
                        where ``len(BoseWord)`` is the number of operators in the BoseWord.
        """

        if not isinstance(initial_position, int) or not isinstance(final_position, int):
            raise TypeError("Positions must be integers.")

        if initial_position < 0 or final_position < 0:
            raise ValueError("Positions must be positive integers.")

        if initial_position > len(self.sorted_dic) - 1 or final_position > len(self.sorted_dic) - 1:
            raise ValueError("Positions are out of range.")

        if initial_position == final_position:
            return BoseSentence({self: 1})

        bw = self
        bs = BoseSentence({bw: 1})
        delta = 1 if initial_position < final_position else -1
        current = initial_position

        while current != final_position:
            indices = list(bw.sorted_dic.keys())
            next = current + delta
            curr_idx, curr_val = indices[current], bw[indices[current]]
            next_idx, next_val = indices[next], bw[indices[next]]

            # commuting identical terms
            if curr_idx[1] == next_idx[1] and curr_val == next_val:
                current += delta
                continue

            coeff = bs.pop(bw)

            bw = dict(bw)
            bw[(current, next_idx[1])] = next_val
            bw[(next, curr_idx[1])] = curr_val

            if curr_idx[1] != next_idx[1]:
                del bw[curr_idx], bw[next_idx]

            bw = BoseWord(bw)

            # commutator is 0
            if curr_val == next_val or curr_idx[1] != next_idx[1]:
                current += delta
                bs += coeff * bw
                continue

            # commutator is 1
            _min = min(current, next)
            _max = max(current, next)
            items = list(bw.sorted_dic.items())

            left = BoseWord({(i, key[1]): value for i, (key, value) in enumerate(items[:_min])})
            middle = BoseWord(
                {(i, key[1]): value for i, (key, value) in enumerate(items[_min : _max + 1])}
            )
            right = BoseWord(
                {(i, key[1]): value for i, (key, value) in enumerate(items[_max + 1 :])}
            )

            terms = left * (1 + middle) * right
            bs += coeff * terms

            current += delta

        return bs


class BoseSentence(dict):
    r"""Dictionary used to represent a Bose sentence, a linear combination of Bose words,
    with the keys as BoseWord instances and the values correspond to coefficients.

    >>> w1 = qml.BoseWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = qml.BoseWord({(0, 1) : '+', (1, 2) : '-'})
    >>> s = qml.BoseSentence({w1 : 1.2, w2: 3.1})
    >>> print(s)
    1.2 * b⁺(0) b(1)
    + 3.1 * b⁺(1) b(2)
    """

    # override the arithmetic dunder methods for numpy arrays so that the
    # methods defined on this class are used instead
    # (i.e. ensure `np.array + BoseSentence` uses `BoseSentence.__radd__`
    # instead of `np.array.__add__`)
    __numpy_ufunc__ = None
    __array_ufunc__ = None

    def __init__(self, operator):
        super().__init__(operator)

    def adjoint(self):
        r"""Return the adjoint of BoseSentence."""
        adjoint_dict = {}
        for key, value in self.items():
            word = key.adjoint()
            scalar = math.conj(value)
            adjoint_dict[word] = scalar

        return BoseSentence(adjoint_dict)

    @property
    def wires(self):
        r"""Return wires of the BoseSentence."""
        return set().union(*(bw.wires for bw in self.keys()))

    def __str__(self):
        r"""String representation of a BoseSentence."""
        if len(self) == 0:
            return "0 * I"
        return "\n+ ".join(f"{coeff} * {bw.to_string()}" for bw, coeff in self.items())

    def __repr__(self):
        r"""Terminal representation for BoseSentence."""
        return f"BoseSentence({dict(self)})"

    def __missing__(self, key):
        r"""If the BoseSentence does not contain a BoseWord then the associated value will be 0."""
        return 0.0

    def __add__(self, other):
        r"""Add a BoseSentence, BoseWord or constant to a BoseSentence by iterating over the
        smaller one and adding its terms to the larger one."""

        if not isinstance(other, (TensorLike, BoseWord, BoseSentence)):
            raise TypeError(f"Cannot add {type(other)} to a BoseSentence.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Bose operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        if isinstance(other, BoseWord):
            other = BoseSentence({other: 1})
        if isinstance(other, TensorLike):
            other = BoseSentence({BoseWord({}): other})

        smaller_bs, larger_bs = (
            (self, copy(other)) if len(self) < len(other) else (other, copy(self))
        )
        for key in smaller_bs:
            larger_bs[key] += smaller_bs[key]

        return larger_bs

    def __radd__(self, other):
        """Add a BoseSentence to a constant, i.e. `2 + BoseSentence({...})`"""
        return self.__add__(other)

    def __sub__(self, other):
        r"""Subtract a BoseSentence, BoseWord or constant from a BoseSentence"""
        if isinstance(other, BoseWord):
            other = BoseSentence({other: -1})
            return self.__add__(other)

        if isinstance(other, BoseSentence):
            other = BoseSentence(dict(zip(other.keys(), [-1 * v for v in other.values()])))
            return self.__add__(other)

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot subtract {type(other)} from a BoseSentence.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Bose operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        other = BoseSentence({BoseWord({}): -1 * other})  # -constant * I
        return self.__add__(other)

    def __rsub__(self, other):
        """Subtract a BoseSentence to a constant, i.e. 2 - BoseSentence({...})"""

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot subtract a BoseSentence from {type(other)}.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Bose operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        self_bs = BoseSentence(dict(zip(self.keys(), [-1 * v for v in self.values()])))
        other_bs = BoseSentence({BoseWord({}): other})  # constant * I
        return self_bs + other_bs

    def __mul__(self, other):
        r"""Multiply two Bose sentences by iterating over each sentence and multiplying the Bose
        words pair-wise"""

        if isinstance(other, BoseWord):
            other = BoseSentence({other: 1})

        if isinstance(other, BoseSentence):
            if (len(self) == 0) or (len(other) == 0):
                return BoseSentence({BoseWord({}): 0})

            product = BoseSentence({})

            for bw1, coeff1 in self.items():
                for bw2, coeff2 in other.items():
                    product[bw1 * bw2] += coeff1 * coeff2

            return product

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot multiply BoseSentence by {type(other)}.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Bose operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )
        vals = [i * other for i in self.values()]
        return BoseSentence(dict(zip(self.keys(), vals)))

    def __rmul__(self, other):
        r"""Reverse multiply a BoseSentence

        Multiplies a BoseSentence "from the left" with an object that can't be modified
        to support __mul__ for BoseSentence. Will be defaulted in for example when
        multiplying ``2 * bose_sentence``, since the ``__mul__`` operator on an integer
        will fail to multiply with a BoseSentence"""

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot multiply {type(other)} by BoseSentence.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Bose operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        vals = [i * other for i in self.values()]
        return BoseSentence(dict(zip(self.keys(), vals)))

    def __pow__(self, value):
        r"""Exponentiate a Bose sentence to an integer power."""
        if value < 0 or not isinstance(value, int):
            raise ValueError("The exponent must be a positive integer.")

        operator = BoseSentence({BoseWord({}): 1})  # 1 times Identity

        for _ in range(value):
            operator *= self

        return operator

    def simplify(self, tol=1e-8):
        r"""Remove any BoseWords in the BoseSentence with coefficients less than the threshold
        tolerance."""
        items = list(self.items())
        for bw, coeff in items:
            if abs(coeff) <= tol:
                del self[bw]

    def normal_order(self):
        r"""Convert a BoseSentence to its normal-ordered form.

        >>> bw = qml.BoseWord({(0, 0): "-", (1, 0): "-", (2, 0): "+", (3, 0): "+"})
        >>> bs = qml.BoseSentence({bw: 1})
        >>> print(bs.normal_order())
        2.0 * I
        + 4.0 * b⁺(0) b(0)
        + 1.0 * b⁺(0) b⁺(0) b(0) b(0)
        """

        bose_sen_ordered = BoseSentence({})

        for bw, coeff in self.items():
            bose_word_ordered = bw.normal_order()
            for bw_ord, coeff_ord in bose_word_ordered.items():
                bose_sen_ordered += coeff_ord * coeff * bw_ord

        return bose_sen_ordered
