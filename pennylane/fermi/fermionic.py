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
"""The fermionic representation classes and functions."""
import re
from copy import copy

from pennylane import fermi, math
from pennylane.typing import TensorLike


class FermiWord(dict):
    r"""Immutable dictionary used to represent a Fermi word, a product of fermionic creation and
    annihilation operators, that can be constructed from a standard dictionary.

    The keys of the dictionary are tuples of two integers. The first integer represents the
    position of the creation/annihilation operator in the Fermi word and the second integer
    represents the orbital it acts on. The values of the dictionary are one of ``'+'`` or ``'-'``
    symbols that denote creation and annihilation operators, respectively. The operator
    :math:`a^{\dagger}_0 a_1` can then be constructed as

    >>> w = qml.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> print(w)
    a⁺(0) a(1)
    """

    # override the arithmetic dunder methods for numpy arrays so that the
    # methods defined on this class are used instead
    # (i.e. ensure `np.array + FermiWord` uses `FermiWord.__radd__` instead of `np.array.__add__`)
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
        r"""Return the adjoint of FermiWord."""
        n = len(self.items())
        adjoint_dict = {}
        for key, value in reversed(self.items()):
            position = n - key[0] - 1
            orbital = key[1]
            fermi_string = "+" if value == "-" else "-"
            adjoint_dict[(position, orbital)] = fermi_string

        return FermiWord(adjoint_dict)

    def items(self):
        """Returns the dictionary items in sorted order."""
        return self.sorted_dic.items()

    @property
    def wires(self):
        r"""Return wires in a FermiWord."""
        return {i[1] for i in self.sorted_dic.keys()}

    def __missing__(self, key):
        r"""Return empty string for a missing key in FermiWord."""
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
        r"""Return a compact string representation of a FermiWord. Each operator in the word is
        represented by the number of the wire it operates on, and a `+` or `-` to indicate either
        a creation or annihilation operator.

        >>> w = qml.FermiWord({(0, 0) : '+', (1, 1) : '-'})
        >>> w.to_string()
        'a⁺(0) a(1)'
        """
        if len(self) == 0:
            return "I"

        symbol_map = {"+": "\u207a", "-": ""}

        string = " ".join(
            [
                "a" + symbol_map[j] + "(" + i + ")"
                for i, j in zip(
                    [str(i[1]) for i in self.sorted_dic.keys()], self.sorted_dic.values()
                )
            ]
        )
        return string

    def __str__(self):
        r"""String representation of a FermiWord."""
        return f"{self.to_string()}"

    def __repr__(self):
        r"""Terminal representation of a FermiWord"""
        return f"FermiWord({self.sorted_dic})"

    def __add__(self, other):
        """Add a FermiSentence, FermiWord or constant to a FermiWord. Converts both
        elements into FermiSentences, and uses the FermiSentence __add__
        method"""

        self_fs = FermiSentence({self: 1.0})

        if isinstance(other, FermiSentence):
            return self_fs + other

        if isinstance(other, FermiWord):
            return self_fs + FermiSentence({other: 1.0})

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot add {type(other)} to a FermiWord.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Fermi operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        return self_fs + FermiSentence({FermiWord({}): other})

    def __radd__(self, other):
        """Add a FermiWord to a constant, i.e. `2 + FermiWord({...})`"""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract a FermiSentence, FermiWord or constant from a FermiWord. Converts both
        elements into FermiSentences (with negative coefficient for `other`), and
        uses the FermiSentence __add__  method"""

        self_fs = FermiSentence({self: 1.0})

        if isinstance(other, FermiWord):
            return self_fs + FermiSentence({other: -1.0})

        if isinstance(other, FermiSentence):
            other_fs = FermiSentence(dict(zip(other.keys(), [-v for v in other.values()])))
            return self_fs + other_fs

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot subtract {type(other)} from a FermiWord.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Fermi operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        return self_fs + FermiSentence({FermiWord({}): -1 * other})  # -constant * I

    def __rsub__(self, other):
        """Subtract a FermiWord to a constant, i.e. `2 - FermiWord({...})`"""

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot subtract a FermiWord from {type(other)}.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Fermi operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )
        self_fs = FermiSentence({self: -1.0})
        other_fs = FermiSentence({FermiWord({}): other})
        return self_fs + other_fs

    def __mul__(self, other):
        r"""Multiply a FermiWord with another FermiWord, a FermiSentence, or a constant.

        >>> w = qml.FermiWord({(0, 0) : '+', (1, 1) : '-'})
        >>> print(w * w)
        a⁺(0) a(1) a⁺(0) a(1)
        """

        if isinstance(other, FermiWord):
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

            return FermiWord(dict_self)

        if isinstance(other, FermiSentence):
            return FermiSentence({self: 1}) * other

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot multiply FermiWord by {type(other)}.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Fermi operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        return FermiSentence({self: other})

    def __rmul__(self, other):
        r"""Reverse multiply a FermiWord

        Multiplies a FermiWord "from the left" with an object that can't be modified
        to support __mul__ for FermiWord. Will be defaulted in for example
        ``2 * FermiWord({(0, 0): "+"})``, where the ``__mul__`` operator on an integer
        will fail to multiply with a FermiWord"""

        return self.__mul__(other)

    def __pow__(self, value):
        r"""Exponentiate a Fermi word to an integer power.

        >>> w = qml.FermiWord({(0, 0) : '+', (1, 1) : '-'})
        >>> print(w**3)
        a⁺(0) a(1) a⁺(0) a(1) a⁺(0) a(1)
        """

        if value < 0 or not isinstance(value, int):
            raise ValueError("The exponent must be a positive integer.")

        operator = FermiWord({})

        for _ in range(value):
            operator *= self

        return operator

    def to_mat(self, n_orbitals=None, format="dense", buffer_size=None):
        r"""Return the matrix representation.

        Args:
            n_orbitals (int or None): Number of orbitals. If not provided, it will be inferred from
                the largest orbital index in the Fermi operator.
            format (str): The format of the matrix. It is "dense" by default. Use "csr" for sparse.
            buffer_size (int or None)`: The maximum allowed memory in bytes to store intermediate results
                in the calculation of sparse matrices. It defaults to ``2 ** 30`` bytes that make
                1GB of memory. In general, larger buffers allow faster computations.

        Returns:
            NumpyArray: Matrix representation of the :class:`~.FermiWord`.

        **Example**

        >>> w = qml.FermiWord({(0, 0): '+', (1, 1): '-'})
        >>> w.to_mat()
        array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
        """
        largest_orb_id = max(key[1] for key in self.keys()) + 1
        if n_orbitals and n_orbitals < largest_orb_id:
            raise ValueError(
                f"n_orbitals cannot be smaller than {largest_orb_id}, got: {n_orbitals}."
            )

        largest_order = n_orbitals or largest_orb_id

        return fermi.jordan_wigner(self, ps=True).to_mat(
            wire_order=list(range(largest_order)), format=format, buffer_size=buffer_size
        )

    def shift_operator(self, initial_position, final_position):
        r"""Shifts an operator in the FermiWord from ``initial_position`` to ``final_position`` by applying the fermionic anti-commutation relations.

        There are three `anti-commutator relations <https://en.wikipedia.org/wiki/Creation_and_annihilation_operators#Creation_and_annihilation_operators_in_quantum_field_theories>`_:

        .. math::
            \left\{ a_i, a_j \right\} = 0, \quad \left\{ a^{\dagger}_i, a^{\dagger}_j \right\} = 0, \quad \left\{ a_i, a^{\dagger}_j \right\} = \delta_{ij},


        where

        .. math::
            \left\{a_i, a_j \right\} = a_i a_j + a_j a_i,

        and

        .. math::
            \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}.

        Args:
            initial_position (int): The position of the operator to be shifted.
            final_position (int): The desired position of the operator.

        Returns:
            FermiSentence: The ``FermiSentence`` obtained after applying the anti-commutator relations.

        Raises:
            TypeError: if ``initial_position`` or ``final_position`` is not an integer
            ValueError: if ``initial_position`` or ``final_position`` are outside the range ``[0, len(fermiword) - 1]``
                        where ``len(fermiword)`` is the number of operators in the FermiWord.


        **Example**

        >>> w = qml.FermiWord({(0, 0): '+', (1, 1): '-'})
        >>> print(w.shift_operator(0, 1))
        -1 * a(1) a⁺(0)
        """

        if not isinstance(initial_position, int) or not isinstance(final_position, int):
            raise TypeError("Positions must be integers.")

        if initial_position < 0 or final_position < 0:
            raise ValueError("Positions must be positive integers.")

        if initial_position > len(self.sorted_dic) - 1 or final_position > len(self.sorted_dic) - 1:
            raise ValueError("Positions are out of range.")

        if initial_position == final_position:
            return FermiSentence({self: 1})

        fw = self
        fs = FermiSentence({fw: 1})
        delta = 1 if initial_position < final_position else -1
        current = initial_position

        while current != final_position:
            indices = list(fw.sorted_dic.keys())
            next = current + delta
            curr_idx, curr_val = indices[current], fw[indices[current]]
            next_idx, next_val = indices[next], fw[indices[next]]

            # commuting identical terms
            if curr_idx[1] == next_idx[1] and curr_val == next_val:
                current += delta
                continue

            coeff = fs.pop(fw)

            fw = dict(fw)
            fw[(current, next_idx[1])] = next_val
            fw[(next, curr_idx[1])] = curr_val

            if curr_idx[1] != next_idx[1]:
                del fw[curr_idx], fw[next_idx]

            fw = FermiWord(fw)

            # anti-commutator is 0
            if curr_val == next_val or curr_idx[1] != next_idx[1]:
                current += delta
                fs += -coeff * fw
                continue

            # anti-commutator is 1
            _min = min(current, next)
            _max = max(current, next)
            items = list(fw.sorted_dic.items())

            left = FermiWord({(i, key[1]): value for i, (key, value) in enumerate(items[:_min])})
            middle = FermiWord(
                {(i, key[1]): value for i, (key, value) in enumerate(items[_min : _max + 1])}
            )
            right = FermiWord(
                {(i, key[1]): value for i, (key, value) in enumerate(items[_max + 1 :])}
            )

            terms = left * (1 - middle) * right
            fs += coeff * terms

            current += delta

        return fs


class FermiSentence(dict):
    r"""Immutable dictionary used to represent a Fermi sentence, a linear combination of Fermi words, with the keys
    as FermiWord instances and the values correspond to coefficients.

    >>> w1 = qml.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = qml.FermiWord({(0, 1) : '+', (1, 2) : '-'})
    >>> s = qml.FermiSentence({w1 : 1.2, w2: 3.1})
    >>> print(s)
    1.2 * a⁺(0) a(1)
    + 3.1 * a⁺(1) a(2)
    """

    # override the arithmetic dunder methods for numpy arrays so that the
    # methods defined on this class are used instead
    # (i.e. ensure `np.array + FermiSentence` uses `FermiSentence.__radd__` instead of `np.array.__add__`)
    __numpy_ufunc__ = None
    __array_ufunc__ = None

    def __init__(self, operator):
        super().__init__(operator)

    def adjoint(self):
        r"""Return the adjoint of FermiSentence."""
        adjoint_dict = {}
        for key, value in self.items():
            word = key.adjoint()
            scalar = math.conj(value)
            adjoint_dict[word] = scalar

        return FermiSentence(adjoint_dict)

    @property
    def wires(self):
        r"""Return wires of the FermiSentence."""
        return set().union(*(fw.wires for fw in self.keys()))

    def __str__(self):
        r"""String representation of a FermiSentence."""
        if len(self) == 0:
            return "0 * I"
        return "\n+ ".join(f"{coeff} * {fw.to_string()}" for fw, coeff in self.items())

    def __repr__(self):
        r"""Terminal representation for FermiSentence."""
        return f"FermiSentence({dict(self)})"

    def __missing__(self, key):
        r"""If the FermiSentence does not contain a FermiWord then the associated value will be 0."""
        return 0.0

    def __add__(self, other):
        r"""Add a FermiSentence, FermiWord or constant to a FermiSentence by iterating over the
        smaller one and adding its terms to the larger one."""

        if not isinstance(other, (TensorLike, FermiWord, FermiSentence)):
            raise TypeError(f"Cannot add {type(other)} to a FermiSentence.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Fermi operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        if isinstance(other, FermiWord):
            other = FermiSentence({other: 1})
        if isinstance(other, TensorLike):
            other = FermiSentence({FermiWord({}): other})

        smaller_fs, larger_fs = (
            (self, copy(other)) if len(self) < len(other) else (other, copy(self))
        )
        for key in smaller_fs:
            larger_fs[key] += smaller_fs[key]

        return larger_fs

    def __radd__(self, other):
        """Add a FermiSentence to a constant, i.e. `2 + FermiSentence({...})`"""
        return self.__add__(other)

    def __sub__(self, other):
        r"""Subtract a FermiSentence, FermiWord or constant from a FermiSentence"""
        if isinstance(other, FermiWord):
            other = FermiSentence({other: -1})
            return self.__add__(other)

        if isinstance(other, FermiSentence):
            other = FermiSentence(dict(zip(other.keys(), [-1 * v for v in other.values()])))
            return self.__add__(other)

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot subtract {type(other)} from a FermiSentence.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Fermi operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        other = FermiSentence({FermiWord({}): -1 * other})  # -constant * I
        return self.__add__(other)

    def __rsub__(self, other):
        """Subtract a FermiSentence to a constant, i.e.

        >>> 2 - FermiSentence({...}) # doctest: +SKIP
        """
        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot subtract a FermiSentence from {type(other)}.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Fermi operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        self_fs = FermiSentence(dict(zip(self.keys(), [-1 * v for v in self.values()])))
        other_fs = FermiSentence({FermiWord({}): other})  # constant * I
        return self_fs + other_fs

    def __mul__(self, other):
        r"""Multiply two Fermi sentences by iterating over each sentence and multiplying the Fermi
        words pair-wise"""

        if isinstance(other, FermiWord):
            other = FermiSentence({other: 1})

        if isinstance(other, FermiSentence):
            if (len(self) == 0) or (len(other) == 0):
                return FermiSentence({FermiWord({}): 0})

            product = FermiSentence({})

            for fw1, coeff1 in self.items():
                for fw2, coeff2 in other.items():
                    product[fw1 * fw2] += coeff1 * coeff2

            return product

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot multiply FermiSentence by {type(other)}.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Fermi operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )
        vals = [i * other for i in self.values()]
        return FermiSentence(dict(zip(self.keys(), vals)))

    def __rmul__(self, other):
        r"""Reverse multiply a FermiSentence

        Multiplies a FermiSentence "from the left" with an object that can't be modified
        to support __mul__ for FermiSentence. Will be defaulted in for example when
        multiplying ``2 * fermi_sentence``, since the ``__mul__`` operator on an integer
        will fail to multiply with a FermiSentence"""

        if not isinstance(other, TensorLike):
            raise TypeError(f"Cannot multiply {type(other)} by FermiSentence.")

        if math.size(other) > 1:
            raise ValueError(
                f"Arithmetic Fermi operations can only accept an array of length 1, "
                f"but received {other} of length {len(other)}"
            )

        vals = [i * other for i in self.values()]
        return FermiSentence(dict(zip(self.keys(), vals)))

    def __pow__(self, value):
        r"""Exponentiate a Fermi sentence to an integer power."""
        if value < 0 or not isinstance(value, int):
            raise ValueError("The exponent must be a positive integer.")

        operator = FermiSentence({FermiWord({}): 1})  # 1 times Identity

        for _ in range(value):
            operator *= self

        return operator

    def simplify(self, tol=1e-8):
        r"""Remove any FermiWords in the FermiSentence with coefficients less than the threshold
        tolerance."""
        items = list(self.items())
        for fw, coeff in items:
            if abs(coeff) <= tol:
                del self[fw]

    def to_mat(self, n_orbitals=None, format="dense", buffer_size=None):
        r"""Return the matrix representation.

        Args:
            n_orbitals (int or None): Number of orbitals. If not provided, it will be inferred from
                the largest orbital index in the Fermi operator
            format (str): The format of the matrix. It is "dense" by default. Use "csr" for sparse.
            buffer_size (int or None)`: The maximum allowed memory in bytes to store intermediate results
                in the calculation of sparse matrices. It defaults to ``2 ** 30`` bytes that make
                1GB of memory. In general, larger buffers allow faster computations.

        Returns:
            NumpyArray: Matrix representation of the :class:`~.FermiSentence`.

        **Example**

        >>> fw1 = qml.FermiWord({(0, 0): "+", (1, 1): "-"})
        >>> fw2 = qml.FermiWord({(0, 0): "+", (1, 0): "-"})
        >>> fs = qml.FermiSentence({fw1: 1.2, fw2: 3.1})
        >>> fs.to_mat()
        array([[0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
               [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
               [0. +0.j, 1.2+0.j, 3.1+0.j, 0. +0.j],
               [0. +0.j, 0. +0.j, 0. +0.j, 3.1+0.j]])
        """
        largest_orb_id = max(key[1] for fermi_word in self.keys() for key in fermi_word.keys()) + 1
        if n_orbitals and n_orbitals < largest_orb_id:
            raise ValueError(
                f"n_orbitals cannot be smaller than {largest_orb_id}, got: {n_orbitals}."
            )

        largest_order = n_orbitals or largest_orb_id

        return fermi.jordan_wigner(self, ps=True).to_mat(
            wire_order=list(range(largest_order)), format=format, buffer_size=buffer_size
        )


def from_string(fermi_string):
    r"""Return a fermionic operator object from its string representation.

    The string representation is a compact format that uses the orbital index and ``'+'`` or ``'-'``
    symbols to indicate creation and annihilation operators, respectively. For instance, the string
    representation for the operator :math:`a^{\dagger}_0 a_1 a^{\dagger}_0 a_1` is
    ``'0+ 1- 0+ 1-'``. The ``'-'`` symbols can be optionally dropped such that ``'0+ 1 0+ 1'``
    represents the same operator. The format commonly used in OpenFermion to represent the same
    operator, ``'0^ 1 0^ 1'`` , is also supported.

    Args:
        fermi_string (str): string representation of the fermionic object

    Returns:
        FermiWord: the fermionic operator object

    **Example**

    >>> from pennylane.fermi import from_string
    >>> print(from_string('0+ 1- 0+ 1-'))
    a⁺(0) a(1) a⁺(0) a(1)

    >>> print(from_string('0+ 1 0+ 1'))
    a⁺(0) a(1) a⁺(0) a(1)

    >>> print(from_string('0^ 1 0^ 1'))
    a⁺(0) a(1) a⁺(0) a(1)

    >>> op1 = qml.FermiC(0) * qml.FermiA(1) * qml.FermiC(2) * qml.FermiA(3)
    >>> op2 = from_string('0+ 1- 2+ 3-')
    >>> op1 == op2
    True
    """
    if fermi_string.isspace() or not fermi_string:
        return FermiWord({})

    fermi_string = " ".join(fermi_string.split())

    if not all(s.isdigit() or s in ["+", "-", "^", " "] for s in fermi_string):
        raise ValueError(f"Invalid character encountered in string {fermi_string}.")

    fermi_string = re.sub(r"\^", "+", fermi_string)

    operators = [i + "-" if i[-1] not in "+-" else i for i in re.split(r"\s", fermi_string)]

    return FermiWord({(i, int(s[:-1])): s[-1] for i, s in enumerate(operators)})


def _to_string(fermi_op, of=False):
    r"""Return a string representation of the :class:`~.FermiWord` object.

    Args:
        fermi_op (FermiWord): the fermionic operator
        of (bool): whether to return a string representation in the same style as OpenFermion using
                    the shorthand: 'q^' = a^\dagger_q 'q' = a_q. Each operator in the word is
                    represented by the number of the wire it operates on

    Returns:
        (str): a string representation of the :class:`~.FermiWord` object

    **Example**

    >>> w = qml.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> _to_string(w)
    '0+ 1-'

    >>> w = qml.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> _to_string(w, of=True)
    '0^ 1'
    """
    if not isinstance(fermi_op, FermiWord):
        raise ValueError(f"fermi_op must be a FermiWord, got: {type(fermi_op)}")

    pl_to_of_map = {"+": "^", "-": ""}

    if len(fermi_op) == 0:
        return "I"

    op_list = ["" for _ in range(len(fermi_op))]
    for loc, wire in fermi_op:
        if of:
            op_str = str(wire) + pl_to_of_map[fermi_op[(loc, wire)]]
        else:
            op_str = str(wire) + fermi_op[(loc, wire)]

        op_list[loc] += op_str

    return " ".join(op_list).rstrip()


class FermiC(FermiWord):
    r"""FermiC(orbital)
    The fermionic creation operator :math:`a^{\dagger}`

    For instance, the operator ``qml.FermiC(2)`` denotes :math:`a^{\dagger}_2`. This operator applied
    to :math:`\ket{0000}` gives :math:`\ket{0010}`.

    Args:
        orbital(int): the non-negative integer indicating the orbital the operator acts on.

    .. note:: While the ``FermiC`` class represents a mathematical operator, it is not a PennyLane qubit :class:`~.Operator`.

    .. seealso:: :class:`~pennylane.FermiA`

    **Example**

    To construct the operator :math:`a^{\dagger}_0`:

    >>> w = qml.FermiC(0)
    >>> print(w)
    a⁺(0)

    This can be combined with the annihilation operator :class:`~pennylane.FermiA`. For example,
    :math:`a^{\dagger}_0 a_1 a^{\dagger}_2 a_3` can be constructed as:

    >>> w = qml.FermiC(0) * qml.FermiA(1) * qml.FermiC(2) * qml.FermiA(3)
    >>> print(w)
    a⁺(0) a(1) a⁺(2) a(3)
    """

    def __init__(self, orbital):
        if not isinstance(orbital, int) or orbital < 0:
            raise ValueError(
                f"FermiC: expected a single, positive integer value for orbital, but received {orbital}"
            )
        self.orbital = orbital
        operator = {(0, orbital): "+"}
        super().__init__(operator)

    def adjoint(self):
        """Return the adjoint of FermiC."""
        return FermiA(self.orbital)


class FermiA(FermiWord):
    r"""FermiA(orbital)
    The fermionic annihilation operator :math:`a`

    For instance, the operator ``qml.FermiA(2)`` denotes :math:`a_2`. This operator applied
    to :math:`\ket{0010}` gives :math:`\ket{0000}`.

    Args:
        orbital(int): the non-negative integer indicating the orbital the operator acts on.

    .. note:: While the ``FermiA`` class represents a mathematical operator, it is not a PennyLane qubit :class:`~.Operator`.

    .. seealso:: :class:`~pennylane.FermiC`

    **Example**

    To construct the operator :math:`a_0`:

    >>> w = qml.FermiA(0)
    >>> print(w)
    a(0)

    This can be combined with the creation operator :class:`~pennylane.FermiC`. For example,
    :math:`a^{\dagger}_0 a_1 a^{\dagger}_2 a_3` can be constructed as:

    >>> w = qml.FermiC(0) * qml.FermiA(1) * qml.FermiC(2) * qml.FermiA(3)
    >>> print(w)
    a⁺(0) a(1) a⁺(2) a(3)
    """

    def __init__(self, orbital):
        if not isinstance(orbital, int) or orbital < 0:
            raise ValueError(
                f"FermiA: expected a single, positive integer value for orbital, but received {orbital}"
            )
        self.orbital = orbital
        operator = {(0, orbital): "-"}
        super().__init__(operator)

    def adjoint(self):
        """Return the adjoint of FermiA."""
        return FermiC(self.orbital)
