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
from numbers import Number
from numpy import ndarray
import numpy as np
from pennylane.typing import TensorLike
import pennylane as qml
from pennylane.labs.vibrational_ham.real_space_ham import _find_2d_degs, _find_3d_degs
from functools import singledispatch
from typing import Union

class BoseWord(dict):
    r"""Immutable dictionary used to represent a Bose word, a product of bosonic creation and
    annihilation operators, that can be constructed from a standard dictionary.

    The keys of the dictionary are tuples of two integers. The first integer represents the
    position of the creation/annihilation operator in the Bose word and the second integer
    represents the orbital it acts on. The values of the dictionary are one of ``'+'`` or ``'-'``
    symbols that denote creation and annihilation operators, respectively. The operator
    :math:`a^{\dagger}_0 a_1` can then be constructed as

    >>> w = BoseWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w
    a⁺(0) a(1)
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
        return set(i[1] for i in self.sorted_dic.keys())

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

        >>> w = BoseWord({(0, 0) : '+', (1, 1) : '-'})
        >>> w.to_string()
        a⁺(0) a(1)
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

        self_fs = BoseSentence({self: 1.0})

        if isinstance(other, BoseSentence):
            return self_fs + other

        if isinstance(other, BoseWord):
            return self_fs + BoseSentence({other: 1.0})

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Bose operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            return self_fs + BoseSentence({BoseWord({}): other})

        raise TypeError(f"Cannot add {type(other)} to a BoseWord.")

    def __radd__(self, other):
        """Add a BoseWord to a constant, i.e. `2 + BoseWord({...})`"""

        if isinstance(other, (Number, ndarray)):
            return self.__add__(other)

        raise TypeError(f"Cannot add a BoseWord to {type(other)}.")

    def __sub__(self, other):
        """Subtract a BoseSentence, BoseWord or constant from a BoseWord. Converts both
        elements into BoseSentences (with negative coefficient for `other`), and
        uses the BoseSentence __add__  method"""

        self_fs = BoseSentence({self: 1.0})

        if isinstance(other, BoseWord):
            return self_fs + BoseSentence({other: -1.0})

        if isinstance(other, BoseSentence):
            other_fs = BoseSentence(dict(zip(other.keys(), [-v for v in other.values()])))
            return self_fs + other_fs

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Bose operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            return self_fs + BoseSentence({BoseWord({}): -1 * other})  # -constant * I

        raise TypeError(f"Cannot subtract {type(other)} from a BoseWord.")

    def __rsub__(self, other):
        """Subtract a BoseWord to a constant, i.e. `2 - BoseWord({...})`"""
        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Bose operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            self_fs = BoseSentence({self: -1.0})
            other_fs = BoseSentence({BoseWord({}): other})
            return self_fs + other_fs

        raise TypeError(f"Cannot subtract a BoseWord from {type(other)}.")

    def __mul__(self, other):
        r"""Multiply a BoseWord with another BoseWord, a BoseSentence, or a constant.

        >>> w = BoseWord({(0, 0) : '+', (1, 1) : '-'})
        >>> w * w
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

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Bose operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            return BoseSentence({self: other})

        raise TypeError(f"Cannot multiply BoseWord by {type(other)}.")

    def __rmul__(self, other):
        r"""Reverse multiply a BoseWord

        Multiplies a BoseWord "from the left" with an object that can't be modified
        to support __mul__ for BoseWord. Will be defaulted in for example
        ``2 * BoseWord({(0, 0): "+"})``, where the ``__mul__`` operator on an integer
        will fail to multiply with a BoseWord"""

        if isinstance(other, (Number, TensorLike)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Bose operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            return BoseSentence({self: other})

        raise TypeError(f"Cannot multiply BoseWord by {type(other)}.")

    def __pow__(self, value):
        r"""Exponentiate a Bose word to an integer power.

        >>> w = BoseWord({(0, 0) : '+', (1, 1) : '-'})
        >>> w**3
        b⁺(0) b(1) b⁺(0) b(1) b⁺(0) b(1)
        """

        if value < 0 or not isinstance(value, int):
            raise ValueError("The exponent must be a positive integer.")

        operator = BoseWord({})

        for _ in range(value):
            operator *= self

        return operator

# pylint: disable=useless-super-delegation
class BoseSentence(dict):
    r"""Immutable dictionary used to represent a Bose sentence, a linear combination of Bose words, with the keys
    as BoseWord instances and the values correspond to coefficients.

    >>> w1 = BoseWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = BoseWord({(0, 1) : '+', (1, 2) : '-'})
    >>> s = BoseSentence({w1 : 1.2, w2: 3.1})
    >>> s
    1.2 * a⁺(0) a(1)
    + 3.1 * a⁺(1) a(2)
    """

    # override the arithmetic dunder methods for numpy arrays so that the
    # methods defined on this class are used instead
    # (i.e. ensure `np.array + BoseSentence` uses `BoseSentence.__radd__` instead of `np.array.__add__`)
    __numpy_ufunc__ = None
    __array_ufunc__ = None

    def __init__(self, operator):
        super().__init__(operator)

    def adjoint(self):
        r"""Return the adjoint of BoseSentence."""
        adjoint_dict = {}
        for key, value in self.items():
            word = key.adjoint()
            scalar = qml.math.conj(value)
            adjoint_dict[word] = scalar

        return BoseSentence(adjoint_dict)

    @property
    def wires(self):
        r"""Return wires of the BoseSentence."""
        return set().union(*(fw.wires for fw in self.keys()))

    def __str__(self):
        r"""String representation of a BoseSentence."""
        if len(self) == 0:
            return "0 * I"
        return "\n+ ".join(f"{coeff} * {fw.to_string()}" for fw, coeff in self.items())

    def __repr__(self):
        r"""Terminal representation for BoseSentence."""
        return f"BoseSentence({dict(self)})"

    def __missing__(self, key):
        r"""If the BoseSentence does not contain a BoseWord then the associated value will be 0."""
        return 0.0

    def __add__(self, other):
        r"""Add a BoseSentence, BoseWord or constant to a BoseSentence by iterating over the
        smaller one and adding its terms to the larger one."""

        # ensure other is BoseSentence
        if isinstance(other, BoseWord):
            other = BoseSentence({other: 1})
        if isinstance(other, Number):
            other = BoseSentence({BoseWord({}): other})
        if isinstance(other, ndarray):
            if qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Bose operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            other = BoseSentence({BoseWord({}): other})

        if isinstance(other, BoseSentence):
            smaller_fs, larger_fs = (
                (self, copy(other)) if len(self) < len(other) else (other, copy(self))
            )
            for key in smaller_fs:
                larger_fs[key] += smaller_fs[key]

            return larger_fs

        raise TypeError(f"Cannot add {type(other)} to a BoseSentence.")

    def __radd__(self, other):
        """Add a BoseSentence to a constant, i.e. `2 + BoseSentence({...})`"""

        if isinstance(other, (Number, ndarray)):
            return self.__add__(other)

        raise TypeError(f"Cannot add a BoseSentence to {type(other)}.")

    def __sub__(self, other):
        r"""Subtract a BoseSentence, BoseWord or constant from a BoseSentence"""
        if isinstance(other, BoseWord):
            other = BoseSentence({other: -1})
            return self.__add__(other)

        if isinstance(other, Number):
            other = BoseSentence({BoseWord({}): -1 * other})  # -constant * I
            return self.__add__(other)

        if isinstance(other, ndarray):
            if qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Bose operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            other = BoseSentence({BoseWord({}): -1 * other})  # -constant * I
            return self.__add__(other)

        if isinstance(other, BoseSentence):
            other = BoseSentence(dict(zip(other.keys(), [-1 * v for v in other.values()])))
            return self.__add__(other)

        raise TypeError(f"Cannot subtract {type(other)} from a BoseSentence.")

    def __rsub__(self, other):
        """Subtract a BoseSentence to a constant, i.e.

        >>> 2 - BoseSentence({...})
        """

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Bose operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            self_fs = BoseSentence(dict(zip(self.keys(), [-1 * v for v in self.values()])))
            other_fs = BoseSentence({BoseWord({}): other})  # constant * I
            return self_fs + other_fs

        raise TypeError(f"Cannot subtract a BoseSentence from {type(other)}.")

    def __mul__(self, other):
        r"""Multiply two Bose sentences by iterating over each sentence and multiplying the Bose
        words pair-wise"""

        if isinstance(other, BoseWord):
            other = BoseSentence({other: 1})

        if isinstance(other, BoseSentence):
            if (len(self) == 0) or (len(other) == 0):
                return BoseSentence({BoseWord({}): 0})

            product = BoseSentence({})

            for fw1, coeff1 in self.items():
                for fw2, coeff2 in other.items():
                    product[fw1 * fw2] += coeff1 * coeff2

            return product

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Bose operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            vals = [i * other for i in self.values()]
            return BoseSentence(dict(zip(self.keys(), vals)))

        raise TypeError(f"Cannot multiply BoseSentence by {type(other)}.")

    def __rmul__(self, other):
        r"""Reverse multiply a BoseSentence

        Multiplies a BoseSentence "from the left" with an object that can't be modified
        to support __mul__ for BoseSentence. Will be defaulted in for example when
        multiplying ``2 * bose_sentence``, since the ``__mul__`` operator on an integer
        will fail to multiply with a BoseSentence"""

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Bose operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            vals = [i * other for i in self.values()]
            return BoseSentence(dict(zip(self.keys(), vals)))

        raise TypeError(f"Cannot multiply {type(other)} by BoseSentence.")

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
        for fw, coeff in items:
            if abs(coeff) <= tol:
                del self[fw]


def normal_order(bose_operator: Union[BoseWord, BoseSentence]):
    r"""Convert a bosonic operator to normal-ordered form.
    Args:
      bose_operator(BoseWord, BoseSentence): the bosonic operator

    Returns:
      normal-ordered bosonic operator
    
    """
    return _normal_order_dispatch(bose_operator)

@singledispatch
def _normal_order_dispatch(bose_operator):
    """Dispatches to appropriate function if bose_operator is a BoseWord or BoseSentence."""
    raise ValueError(f"bose_operator must be a BoseWord or BoseSentence, got: {bose_operator}")

@_normal_order_dispatch.register
def _(bose_operator: BoseWord):

    bw_terms = sorted(bose_operator)
    len_op = len(bw_terms)
    bw_comm = BoseSentence({BoseWord({}): 0.0})
    for i in range(1, len_op):
        for j in range(i, 0, -1):
            key_r = bw_terms[j]
            key_l = bw_terms[j-1]

            if bose_operator[key_l] == "-" and bose_operator[key_r] == "+":
                bw_terms[j] = key_l
                bw_terms[j-1] = key_r

                # Add the term for commutator
                if key_r[1] == key_l[1]:
                    term_dict_comm = {key: value for key, value in bose_operator.items()
                                       if key not in [key_r, key_l]}
                    bw_comm += normal_order(BoseWord(term_dict_comm))

    bose_dict = {}
    for i in range(len_op):
        bose_dict[(i, bw_terms[i][1])] = bose_operator[bw_terms[i]]

    ordered_op = BoseWord(bose_dict) + bw_comm
    ordered_op.simplify(tol=1e-8)

    return ordered_op

               
@_normal_order_dispatch.register
def _(bose_operator: BoseSentence):

    bose_sen_ordered = BoseSentence()  # Empty PS as 0 operator to add Pws to

    for bw, coeff in bose_operator.items():
        bose_word_ordered = normal_order(bw)

        bose_sen_ordered[bose_word_ordered] = coeff

    return bose_sen_ordered

def bosonic_hamiltonian(pes_data):
    """
    Implementation pending
    """
    pass

def kinetic_term(freqs):
    nmodes = len(freqs)
    expr = BoseSentence()
    for ii in range(nmodes):
        pi = BoseWord({(0, ii): '+'}) - BoseWord({(1, ii): '-'})
        expr -= 0.25 * freqs[ii] * pi*pi

    return normal_order(expr)

def harmonic_oscillators(freqs):
    """
    Builds the harmonic oscillator term in the Vib Hamiltonian as Bosonic Operators
    """

    nmodes = len(freqs)
    kin = kinetic_term(freqs)

    pot = BoseSentence()
    for ii in range(nmodes):
        pot += position_to_boson([ii,ii]) * freqs[ii] * 0.5

    return kin + pot

def position_to_boson(index):
    """
    Given an index of a product of position operators,
    expand them using HO ladder operators and simplify.
    """

    # creation operator
    factors_c = tuple([(int(entry), 1) for entry in index])
    # annihilation operator
    factors_a = tuple([(int(entry), 0) for entry in index])

    # This section should be re-written using BoseSentences.
    expr = BoseSentence()
    for ii in range(len(index)):
        expr *= (
            BoseWord({(ii, index[ii]) : '+'}) + BoseWord({(ii, index[ii]) : '-'})
        ) / np.sqrt(2)

    return normal_order(expr)


def taylor_to_bosonic(coeffs):
    """
    Convert taylor coefficients to bosonic hamiltonian.
    """
    num_coups = len(coeffs)

    nmodes, deg = np.shape(coeffs[0])
    deg += 2

    degs_2d = _find_2d_degs(deg)  # Missing for now
    degs_3d = _find_3d_degs(deg)

    b_op = BoseSentence()  # Should use BoseSentence.
    for nc in range(num_coups):
        f_eff = coeffs[nc]

        # The following can be refactored.
        if nc == 0:
            for ii in range(nmodes):
                for i_deg, f_val in enumerate(f_eff[ii, :]):
                    idx = (i_deg + 3) * [ii]
                    b_op += f_val * position_to_boson(idx)

        if nc == 1:
            for i1 in range(nmodes):
                for i2 in range(i1):
                    for deg_idx, Qs in enumerate(degs_2d):
                        idx = Qs[0] * [i1] + Qs[1] * [i2]
                        b_op += f_eff[i1, i2, deg_idx] * position_to_boson(idx)

        if nc == 2:
            for i1 in range(nmodes):
                for i2 in range(i1):
                    for i3 in range(i2):
                        for deg_idx, Qs in enumerate(degs_3d):
                            idx = Qs[0] * [i1] + Qs[1] * [i2] + Qs[2] * [i3]
                            b_op += f_eff[i1, i2, i3, deg_idx] * position_to_boson(idx)

        if nc > 2:
            print("Warning, enter   ed array for more than 3-mode couplings, not implemented!")
            print("Returning up to three-mode couplings")

    return normal_order(b_op)  # Should use BoseSentence equivalent
