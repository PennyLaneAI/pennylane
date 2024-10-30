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
from pennylane.labs.vibrational_ham.taylorForm import _twobody_degs, _threebody_degs
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

            print(key_r)

            if bose_operator[key_l] == "-" and bose_operator[key_r] == "+":
                bw_terms[j] = key_l
                bw_terms[j-1] = key_r

                # Add the term for commutator
                if key_r[1] == key_l[1]:
                    term_dict_comm = {key: value for key, value in bose_operator.items()
                                       if key not in [key_r, key_l]}
                    print(term_dict_comm)
                    bw_comm += normal_order(BoseWord(term_dict_comm))

    bose_dict = {}
    for i in range(len_op):
        bose_dict[(i, bw_terms[i][1])] = bose_operator[bw_terms[i]]

    ordered_op = BoseWord(bose_dict) + bw_comm
    ordered_op.simplify(tol=1e-8)

    return ordered_op

               
@_normal_order_dispatch.register
def _(bose_operator: BoseSentence):

    ordered_dict = {}  # Empty PS as 0 operator to add Pws to

    for bw, coeff in bose_operator.items():
        bose_word_ordered = normal_order(bw)
        for bw_ordered in bose_word_ordered:
            ordered_dict[bw_ordered] = coeff

    bose_sen_ordered = BoseSentence(ordered_dict)
    return bose_sen_ordered

def bosonic_hamiltonian(pes_data):
    """
    Implementation pending
    """
    pass

def _q_to_bos(mode):
    bop = 1/np.sqrt(2) * BoseWord({(0, mode): "-"})
    bdag = 1/np.sqrt(2) * BoseWord({(0, mode): "+"})

    return bop + bdag

def _p_to_bos(mode):
    bop = 1j/np.sqrt(2) * BoseWord({(0, mode): '-'})
    bdag = 1j/np.sqrt(2) * BoseWord({(0, mode): '+'})

    return bdag - bop

def taylor_to_bosonic(taylor_arr, start_deg = 2, verbose=True):
    num_coups = len(taylor_arr)

    taylor_1D = taylor_arr[0]
    M, num_1D_coeffs = np.shape(taylor_1D)

    taylor_deg = num_1D_coeffs + start_deg - 1

    op_arr = []
    if verbose:
            print("Printing one-mode expansion coefficients:")
    for m in range(M):
        qm = _q_to_bos(m)
        if verbose:
            print(f"qm as bosons is {qm}")
        for deg_i in range(start_deg, taylor_deg+1):
            if verbose:
                print(f"q{m}^{deg_i} --> {taylor_1D[m,deg_i-start_deg]}")
            qpow = qm ** deg_i
            print(qpow)
            op_arr.append(normal_order(taylor_1D[m,deg_i-start_deg] * qpow))
            if verbose:
                print(f"Added associated operator {op_arr[-1]}")

    if num_coups > 1:
        if verbose:
            print("Printing two-mode expansion coefficients:")
        taylor_2D = taylor_arr[1]
        degs_2d = _twobody_degs(taylor_deg, min_deg = start_deg)
        for m1 in range(M):
            qm1 = _q_to_bos(m1)
            for m2 in range(m1):
                qm2 = _q_to_bos(m2)
                for deg_idx, Qs in enumerate(degs_2d):
                    q1deg = Qs[0]
                    q2deg = Qs[1]
                    if verbose:
                        print(f"q{m1}^{q1deg}*q{m2}^{q2deg} --> {taylor_2D[m1,m2,deg_idx]}")
                    qm1pow = qm1 ** q1deg
                    qm2pow = qm2 ** q2deg
                    op_arr.append(normal_order(taylor_2D[m1,m2,deg_idx] * qm1pow * qm2pow))
                    if verbose:
                        print(f"Added associated operator {op_arr[-1]}")

    if num_coups > 2:
        if verbose:
            print("Printing three-mode expansion coefficients:")
        degs_3d = _threebody_degs(taylor_deg, min_deg=start_deg)
        taylor_3D = taylor_arr[2]
        for m1 in range(M):
            qm1 = _q_to_bos(m1)
            for m2 in range(m1):
                qm2 = _q_to_bos(m2)
                for m3 in range(m2):
                    qm3 = _q_to_bos(m3)
                    for deg_idx, Qs in enumerate(degs_3d):
                        q1deg = Qs[0]
                        q2deg = Qs[1]
                        q3deg = Qs[2]
                        qm1pow = qm1 ** q1deg
                        qm2pow = qm2 ** q2deg
                        qm3pow = qm3 ** q3deg
                        if verbose:
                            print(f"q{m1}^{q1deg}*q{m2}^{q2deg}*q{m3}^{q3deg} --> {taylor_3D[m1,m2,m3,deg_idx]}")
                        op_arr.append(normal_order(taylor_3D[m1,m2,m3,deg_idx] * qm1pow * qm2pow * qm3pow))
                        if verbose:
                            print(f"Added associated operator {op_arr[-1]}")

    if num_coups > 3:
        raise ValueError("Found 4-mode expansion coefficients, not defined!")


    return normal_order(BoseSentence(op_arr))

def taylor_ham_to_bosonic(taylor_arr, freqs, is_loc = True, Uloc = None, verbose=True):
    taylor_1D = taylor_arr[0]
    M, num_1D_coeffs = np.shape(taylor_1D)
    if is_loc:
        start_deg = 2
    else:
        start_deg = 3

    harm_pot = []

    #Add Harmonic component
    for m in range(M):
        qm2 = normal_order(_q_to_bos(m) * _q_to_bos(m))
        harm_pot.append(qm2 * freqs[m] * 0.5)

    ham = taylor_to_bosonic(taylor_arr, start_deg, verbose) + BoseSentence(harm_pot)

    #Create kinetic energy operation
    alphas_arr = np.einsum('ij,ik,j,k->jk', Uloc, Uloc, np.sqrt(freqs), np.sqrt(freqs))
    kin_arr = []
    for m1 in range(M):
        pm1 = _p_to_bos(m1)
        if verbose:
            print(f"p{m1} as bosons is {pm1}")
        for m2 in range(M):
            pm2 = _p_to_bos(m2)
            kin_arr.append((0.5 * alphas_arr[m1,m2]) * normal_order(pm1 * pm2))

    return normal_order(ham), normal_order(BoseSentence(kin_arr))
