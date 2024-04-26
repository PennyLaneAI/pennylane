# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Pauli arithmetic abstract reduced representation classes"""
# pylint:disable=protected-access
from copy import copy
from functools import reduce, lru_cache

import numpy as np
from scipy import sparse

import pennylane as qml
from pennylane import math
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pennylane.operation import Tensor
from pennylane.ops import Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum


I = "I"
X = "X"
Y = "Y"
Z = "Z"

op_map = {
    I: Identity,
    X: PauliX,
    Y: PauliY,
    Z: PauliZ,
}

op_to_str_map = {
    Identity: I,
    PauliX: X,
    PauliY: Y,
    PauliZ: Z,
}

matI = np.eye(2)
matX = np.array([[0, 1], [1, 0]])
matY = np.array([[0, -1j], [1j, 0]])
matZ = np.array([[1, 0], [0, -1]])

mat_map = {
    I: matI,
    X: matX,
    Y: matY,
    Z: matZ,
}

anticom_map = {
    I: {I: 0, X: 0, Y: 0, Z: 0},
    X: {I: 0, X: 0, Y: 1, Z: 1},
    Y: {I: 0, X: 1, Y: 0, Z: 1},
    Z: {I: 0, X: 1, Y: 1, Z: 0},
}


@lru_cache
def _make_operation(op, wire):
    return op_map[op](wire)


@lru_cache
def _cached_sparse_data(op):
    """Returns the sparse data and indices of a Pauli operator."""
    if op == "I":
        data = np.array([1.0, 1.0], dtype=np.complex128)
        indices = np.array([0, 1], dtype=np.int64)
    elif op == "X":
        data = np.array([1.0, 1.0], dtype=np.complex128)
        indices = np.array([1, 0], dtype=np.int64)
    elif op == "Y":
        data = np.array([-1.0j, 1.0j], dtype=np.complex128)
        indices = np.array([1, 0], dtype=np.int64)
    elif op == "Z":
        data = np.array([1.0, -1.0], dtype=np.complex128)
        indices = np.array([0, 1], dtype=np.int64)
    return data, indices


@lru_cache(maxsize=2)
def _cached_arange(n):
    "Caches `np.arange` output to speed up sparse calculations."
    return np.arange(n)


pauli_to_sparse_int = {I: 0, X: 1, Y: 1, Z: 0}  # (I, Z) and (X, Y) have the same sparsity


def _ps_to_sparse_index(pauli_words, wires):
    """Represent the Pauli words sparse structure in a matrix of shape n_words x n_wires."""
    indices = np.zeros((len(pauli_words), len(wires)))
    for i, pw in enumerate(pauli_words):
        if not pw.wires:
            continue
        wire_indices = np.array(wires.indices(pw.wires))
        indices[i, wire_indices] = [pauli_to_sparse_int[pw[w]] for w in pw.wires]
    return indices


_map_I = {
    I: (1, I),
    X: (1, X),
    Y: (1, Y),
    Z: (1, Z),
}
_map_X = {
    I: (1, X),
    X: (1, I),
    Y: (1.0j, Z),
    Z: (-1.0j, Y),
}
_map_Y = {
    I: (1, Y),
    X: (-1.0j, Z),
    Y: (1, I),
    Z: (1j, X),
}
_map_Z = {
    I: (1, Z),
    X: (1j, Y),
    Y: (-1.0j, X),
    Z: (1, I),
}

mul_map = {I: _map_I, X: _map_X, Y: _map_Y, Z: _map_Z}


class PauliWord(dict):
    r"""
    Immutable dictionary used to represent a Pauli Word,
    associating wires with their respective operators.
    Can be constructed from a standard dictionary.

    .. note::

        An empty :class:`~.PauliWord` will be treated as the multiplicative
        identity (i.e identity on all wires). Its matrix is the identity matrix
        (trivially the :math:`1\times 1` one matrix when no ``wire_order`` is passed to
        ``PauliWord({}).to_mat()``).

    **Examples**

    Initializing a Pauli word:

    >>> w = PauliWord({"a": 'X', 2: 'Y', 3: 'Z'})
    >>> w
    X(a) @ Y(2) @ Z(3)

    When multiplying Pauli words together, we obtain a :class:`~PauliSentence` with the resulting ``PauliWord`` as a key and the corresponding coefficient as its value.

    >>> w1 = PauliWord({0:"X", 1:"Y"})
    >>> w2 = PauliWord({1:"X", 2:"Z"})
    >>> w1 @ w2
    -1j * Z(1) @ Z(2) @ X(0)

    We can multiply scalars to Pauli words or add/subtract them, resulting in a :class:`~PauliSentence` instance.

    >>> 0.5 * w1 - 1.5 * w2 + 2
    0.5 * X(0) @ Y(1)
    + -1.5 * X(1) @ Z(2)
    + 2 * I

    """

    # this allows scalar multiplication from left with numpy arrays np.array(0.5) * pw1
    # taken from [stackexchange](https://stackoverflow.com/questions/40694380/forcing-multiplication-to-use-rmul-instead-of-numpy-array-mul-or-byp/44634634#44634634)
    __array_priority__ = 1000

    def __missing__(self, key):
        """If the wire is not in the Pauli word,
        then no operator acts on it, so return the Identity."""
        return I

    def __init__(self, mapping):
        """Strip identities from PauliWord on init!"""
        for wire, op in mapping.copy().items():
            if op == I:
                del mapping[wire]
        super().__init__(mapping)

    @property
    def pauli_rep(self):
        """Trivial pauli_rep"""
        return PauliSentence({self: 1.0})

    def __reduce__(self):
        """Defines how to pickle and unpickle a PauliWord. Otherwise, un-pickling
        would cause __setitem__ to be called, which is forbidden on PauliWord.
        For more information, see: https://docs.python.org/3/library/pickle.html#object.__reduce__
        """
        return (PauliWord, (dict(self),))

    def __copy__(self):
        """Copy the PauliWord instance."""
        return PauliWord(dict(self.items()))

    def __deepcopy__(self, memo):
        res = self.__copy__()
        memo[id(self)] = res
        return res

    def __setitem__(self, key, item):
        """Restrict setting items after instantiation."""
        raise TypeError("PauliWord object does not support assignment")

    def update(self, __m, **kwargs) -> None:
        """Restrict updating PW after instantiation."""
        raise TypeError("PauliWord object does not support assignment")

    def __hash__(self):
        return hash(frozenset(self.items()))

    def _matmul(self, other):
        """Private matrix multiplication that returns (pauli_word, coeff) tuple for more lightweight processing"""
        base, iterator, swapped = (
            (self, other, False) if len(self) >= len(other) else (other, self, True)
        )
        result = copy(dict(base))
        coeff = 1

        for wire, term in iterator.items():
            if wire in base:
                factor, new_op = mul_map[term][base[wire]] if swapped else mul_map[base[wire]][term]
                if new_op == I:
                    del result[wire]
                else:
                    coeff *= factor
                    result[wire] = new_op
            elif term != I:
                result[wire] = term

        return PauliWord(result), coeff

    def __matmul__(self, other):
        """Multiply two Pauli words together using the matrix product if wires overlap
        and the tensor product otherwise.

        Empty Pauli words are treated as the Identity operator on all wires.

        Args:
            other (PauliWord): The Pauli word to multiply with

        Returns:
            PauliSentence: coeff * new_word
        """
        if isinstance(other, PauliSentence):
            return PauliSentence({self: 1.0}) @ other

        new_word, coeff = self._matmul(other)
        return PauliSentence({new_word: coeff})

    def __mul__(self, other):
        """Multiply a PauliWord by a scalar

        Args:
            other (Scalar): The scalar to multiply the PauliWord with

        Returns:
            PauliSentence
        """

        if isinstance(other, TensorLike):
            if not qml.math.ndim(other) == 0:
                raise ValueError(
                    f"Attempting to multiply a PauliWord with an array of dimension {qml.math.ndim(other)}"
                )

            return PauliSentence({self: other})
        raise TypeError(
            f"PauliWord can only be multiplied by numerical data. Attempting to multiply by {other} of type {type(other)}"
        )

    __rmul__ = __mul__

    def __add__(self, other):
        """Add PauliWord instances and scalars to PauliWord.
        Returns a PauliSentence."""
        # Note that the case of PauliWord + PauliSentence is covered in PauliSentence
        if isinstance(other, PauliWord):
            if other == self:
                return PauliSentence({self: 2.0})
            return PauliSentence({self: 1.0, other: 1.0})

        if isinstance(other, TensorLike):
            # Scalars are interepreted as scalar * Identity
            IdWord = PauliWord({})
            if IdWord == self:
                return PauliSentence({self: 1.0 + other})
            return PauliSentence({self: 1.0, IdWord: other})

        return NotImplemented

    __radd__ = __add__

    def __iadd__(self, other):
        """Inplace addition"""
        return self + other

    def __sub__(self, other):
        """Subtract other PauliSentence, PauliWord, or scalar"""
        return self + -1 * other

    def __rsub__(self, other):
        """Subtract other PauliSentence, PauliWord, or scalar"""
        return -1 * self + other

    def __truediv__(self, other):
        """Divide a PauliWord by a scalar"""
        if isinstance(other, TensorLike):
            return self * (1 / other)
        raise TypeError(
            f"PauliWord can only be divided by numerical data. Attempting to divide by {other} of type {type(other)}"
        )

    def commutes_with(self, other):
        """Fast check if two PauliWords commute with each other"""
        wires = set(self) & set(other)
        if not wires:
            return True
        anticom_count = sum(anticom_map[self[wire]][other[wire]] for wire in wires)
        return (anticom_count % 2) == 0

    def _commutator(self, other):
        """comm between two PauliWords, returns tuple (new_word, coeff) for faster arithmetic"""
        # This may be helpful to developers that need a more lightweight comm between pauli words
        # without creating PauliSentence classes

        if self.commutes_with(other):
            return PauliWord({}), 0.0
        new_word, coeff = self._matmul(other)
        return new_word, 2 * coeff

    def commutator(self, other):
        """
        Compute commutator between a ``PauliWord`` :math:`P` and other operator :math:`O`

        .. math:: [P, O] = P O - O P

        When the other operator is a :class:`~PauliWord` or :class:`~PauliSentence`,
        this method is faster than computing ``P @ O - O @ P``. It is what is being used
        in :func:`~commutator` when setting ``pauli=True``.

        Args:
            other (Union[Operator, PauliWord, PauliSentence]): Second operator

        Returns:
            ~PauliSentence: The commutator result in form of a :class:`~PauliSentence` instances.

        **Examples**

        You can compute commutators between :class:`~PauliWord` instances.

        >>> pw = PauliWord({0:"X"})
        >>> pw.commutator(PauliWord({0:"Y"}))
        2j * Z(0)

        You can also compute the commutator with other operator types if they have a Pauli representation.

        >>> pw.commutator(qml.Y(0))
        2j * Z(0)
        """
        if isinstance(other, PauliWord):
            new_word, coeff = self._commutator(other)
            if coeff == 0:
                return PauliSentence({})
            return PauliSentence({new_word: coeff})

        if isinstance(other, qml.operation.Operator):
            op_self = PauliSentence({self: 1.0})
            return op_self.commutator(other)

        if isinstance(other, PauliSentence):
            # for infix method, this would be handled by __ror__
            return -1.0 * other.commutator(self)

        raise NotImplementedError(
            f"Cannot compute natively a commutator between PauliWord and {other} of type {type(other)}"
        )

    def __str__(self):
        """String representation of a PauliWord."""
        if len(self) == 0:
            return "I"
        return " @ ".join(f"{op}({w})" for w, op in self.items())

    def __repr__(self):
        """Terminal representation for PauliWord"""
        return str(self)

    @property
    def wires(self):
        """Track wires in a PauliWord."""
        return Wires(self)

    def to_mat(self, wire_order=None, format="dense", coeff=1.0):
        """Returns the matrix representation.

        Keyword Args:
            wire_order (iterable or None): The order of qubits in the tensor product.
            format (str): The format of the matrix. It is "dense" by default. Use "csr" for sparse.
            coeff (float): Coefficient multiplying the resulting matrix.

        Returns:
            (Union[NumpyArray, ScipySparseArray]): Matrix representation of the Pauli word.

        Raises:
            ValueError: Can't get the matrix of an empty PauliWord.
        """
        wire_order = self.wires if wire_order is None else Wires(wire_order)
        if not wire_order.contains_wires(self.wires):
            raise ValueError(
                "Can't get the matrix for the specified wire order because it "
                f"does not contain all the Pauli word's wires {self.wires}"
            )

        if len(self) == 0:
            n = len(wire_order) if wire_order is not None else 0
            return (
                np.diag([coeff] * 2**n)
                if format == "dense"
                else coeff * sparse.eye(2**n, format=format, dtype="complex128")
            )

        if format == "dense":
            return coeff * reduce(math.kron, (mat_map[self[w]] for w in wire_order))

        return self._to_sparse_mat(wire_order, coeff)

    def _to_sparse_mat(self, wire_order, coeff):
        """Compute the sparse matrix of the Pauli word times a coefficient, given a wire order.
        See pauli_sparse_matrices.md for the technical details of the implementation."""
        matrix_size = 2 ** len(wire_order)
        matrix = sparse.csr_matrix((matrix_size, matrix_size), dtype="complex128")
        # Avoid checks and copies in __init__ by directly setting the attributes of an empty matrix
        matrix.data = self._get_csr_data(wire_order, coeff)
        matrix.indices = self._get_csr_indices(wire_order)
        matrix.indptr = _cached_arange(matrix_size + 1)  # Non-zero entries by row (starting from 0)
        return matrix

    def _get_csr_data(self, wire_order, coeff):
        """Computes the sparse matrix data of the Pauli word times a coefficient, given a wire order."""
        full_word = [self[wire] for wire in wire_order]

        matrix_size = 2 ** len(wire_order)
        if len(self) == 0:
            return np.full(matrix_size, coeff, dtype=np.complex128)
        data = np.empty(matrix_size, dtype=np.complex128)  # Non-zero values
        current_size = 2
        data[:current_size], _ = _cached_sparse_data(full_word[-1])
        data[:current_size] *= coeff  # Multiply initial term better than the full matrix
        for s in full_word[-2::-1]:
            if s == "I":
                data[current_size : 2 * current_size] = data[:current_size]
            elif s == "X":
                data[current_size : 2 * current_size] = data[:current_size]
            elif s == "Y":
                data[current_size : 2 * current_size] = 1j * data[:current_size]
                data[:current_size] *= -1j
            elif s == "Z":
                data[current_size : 2 * current_size] = -data[:current_size]
            current_size *= 2
        return data

    def _get_csr_data_2(self, wire_order, coeff):
        """Computes the sparse matrix data of the Pauli word times a coefficient, given a wire order."""
        full_word = [self[wire] for wire in wire_order]
        nwords = len(full_word)
        if nwords < 2:
            return np.array([1.0]), self._get_csr_data(wire_order, coeff)
        outer = self._get_csr_data(wire_order[: nwords // 2], 1.0)
        inner = self._get_csr_data(wire_order[nwords // 2 :], coeff)
        return outer, inner

    def _get_csr_indices(self, wire_order):
        """Computes the sparse matrix indices of the Pauli word times a coefficient, given a wire order."""
        full_word = [self[wire] for wire in wire_order]
        matrix_size = 2 ** len(wire_order)
        if len(self) == 0:
            return _cached_arange(matrix_size)
        indices = np.empty(matrix_size, dtype=np.int64)  # Column index of non-zero values
        current_size = 2
        _, indices[:current_size] = _cached_sparse_data(full_word[-1])
        for s in full_word[-2::-1]:
            if s == "I":
                indices[current_size : 2 * current_size] = indices[:current_size] + current_size
            elif s == "X":
                indices[current_size : 2 * current_size] = indices[:current_size]
                indices[:current_size] += current_size
            elif s == "Y":
                indices[current_size : 2 * current_size] = indices[:current_size]
                indices[:current_size] += current_size
            elif s == "Z":
                indices[current_size : 2 * current_size] = indices[:current_size] + current_size
            current_size *= 2
        return indices

    def operation(self, wire_order=None, get_as_tensor=False):
        """Returns a native PennyLane :class:`~pennylane.operation.Operation` representing the PauliWord."""
        if len(self) == 0:
            return Identity(wires=wire_order)

        factors = [_make_operation(op, wire) for wire, op in self.items()]

        if get_as_tensor:
            return factors[0] if len(factors) == 1 else Tensor(*factors)
        pauli_rep = PauliSentence({self: 1})
        return factors[0] if len(factors) == 1 else Prod(*factors, _pauli_rep=pauli_rep)

    def hamiltonian(self, wire_order=None):
        """Return :class:`~pennylane.Hamiltonian` representing the PauliWord."""
        if len(self) == 0:
            if wire_order in (None, [], Wires([])):
                raise ValueError("Can't get the Hamiltonian for an empty PauliWord.")
            return qml.Hamiltonian([1], [Identity(wires=wire_order)])

        obs = [_make_operation(op, wire) for wire, op in self.items()]
        return qml.Hamiltonian([1], [obs[0] if len(obs) == 1 else Tensor(*obs)])

    def map_wires(self, wire_map: dict) -> "PauliWord":
        """Return a new PauliWord with the wires mapped."""
        return self.__class__({wire_map.get(w, w): op for w, op in self.items()})


pw_id = PauliWord({})  # empty pauli word to be re-used


class PauliSentence(dict):
    r"""Dictionary representing a linear combination of Pauli words, with the keys
    as :class:`~pennylane.pauli.PauliWord` instances and the values correspond to coefficients.

    .. note::

        An empty :class:`~.PauliSentence` will be treated as the additive
        identity (i.e ``0 * Identity()``). Its matrix is the all-zero matrix
        (trivially the :math:`1\times 1` zero matrix when no ``wire_order`` is passed to
        ``PauliSentence({}).to_mat()``).

    **Examples**

    >>> ps = PauliSentence({
            PauliWord({0:'X', 1:'Y'}): 1.23,
            PauliWord({2:'Z', 0:'Y'}): -0.45j
        })
    >>> ps
    1.23 * X(0) @ Y(1)
    + (-0-0.45j) * Z(2) @ Y(0)

    Combining Pauli words automatically results in Pauli sentences that can be used to construct more complicated operators.

    >>> w1 = PauliWord({0:"X", 1:"Y"})
    >>> w2 = PauliWord({1:"X", 2:"Z"})
    >>> ps = 0.5 * w1 - 1.5 * w2 + 2
    >>> ps + PauliWord({3:"Z"}) - 1
    0.5 * X(0) @ Y(1)
    + -1.5 * X(1) @ Z(2)
    + 1 * I
    + 1.0 * Z(3)

    Note that while the empty :class:`~PauliWord` ``PauliWord({})`` respresents the identity, the empty ``PauliSentence`` represents 0

    >>> PauliSentence({})
    0 * I

    We can compute commutators using the ``PauliSentence.commutator()`` method

    >>> op1 = PauliWord({0:"X", 1:"X"})
    >>> op2 = PauliWord({0:"Y"}) + PauliWord({1:"Y"})
    >>> op1.commutator(op2)
    2j * Z(0) @ X(1)
    + 2j * X(0) @ Z(1)

    Or, alternatively, use :func:`~commutator`.

    >>> qml.commutator(op1, op2, pauli=True)

    Note that we need to specify ``pauli=True`` as :func:`~.commutator` returns PennyLane operators by default.

    """

    # this allows scalar multiplication from left with numpy arrays np.array(0.5) * ps1
    # taken from [stackexchange](https://stackoverflow.com/questions/40694380/forcing-multiplication-to-use-rmul-instead-of-numpy-array-mul-or-byp/44634634#44634634)
    __array_priority__ = 1000

    @property
    def pauli_rep(self):
        """Trivial pauli_rep"""
        return self

    def __missing__(self, key):
        """If the PauliWord is not in the sentence then the coefficient
        associated with it should be 0."""
        return 0.0

    def trace(self):
        r"""Return the normalized trace of the ``PauliSentence`` instance

        .. math:: \frac{1}{2^n} \text{tr}\left( P \right).

        The normalized trace does not scale with the number of qubits :math:`n`.

        >>> PauliSentence({PauliWord({0:"I", 1:"I"}): 0.5}).trace()
        0.5
        >>> PauliSentence({PauliWord({}): 0.5}).trace()
        0.5

        """
        return self.get(pw_id, 0.0)

    def __add__(self, other):
        """Add a PauliWord, scalar or other PauliSentence to a PauliSentence.

        Empty Pauli sentences are treated as the additive identity
        (i.e 0 * Identity on all wires). The non-empty Pauli sentence is returned.
        """
        if isinstance(other, PauliSentence):
            smaller_ps, larger_ps = (
                (self, copy(other)) if len(self) < len(other) else (other, copy(self))
            )
            for key in smaller_ps:
                larger_ps[key] += smaller_ps[key]

            return larger_ps

        if isinstance(other, PauliWord):
            res = copy(self)
            if other in res:
                res[other] += 1.0
            else:
                res[other] = 1.0
            return res

        if isinstance(other, TensorLike):
            # Scalars are interepreted as scalar * Identity
            res = copy(self)
            IdWord = PauliWord({})
            if IdWord in res:
                res[IdWord] += other
            else:
                res[IdWord] = other
            return res

        raise TypeError(f"Cannot add {other} of type {type(other)} to PauliSentence")

    __radd__ = __add__

    def __iadd__(self, other):
        """Inplace addition of two Pauli sentence together by adding terms of other to self"""
        if isinstance(other, PauliSentence):
            for key in other:
                if key in self:
                    self[key] += other[key]
                else:
                    self[key] = other[key]
            return self

        if isinstance(other, PauliWord):
            if other in self:
                self[other] += 1.0
            else:
                self[other] = 1.0
            return self

        if isinstance(other, TensorLike):
            IdWord = PauliWord({})
            if IdWord in self:
                self[IdWord] += other
            else:
                self[IdWord] = other
            return self

        raise TypeError(f"Cannot add {other} of type {type(other)} to PauliSentence")

    def __sub__(self, other):
        """Subtract other PauliSentence, PauliWord, or scalar"""
        return self + -1 * other

    def __rsub__(self, other):
        """Subtract other PauliSentence, PauliWord, or scalar"""
        return -1 * self + other

    def __copy__(self):
        """Copy the PauliSentence instance."""
        copied_ps = {}
        for pw, coeff in self.items():
            copied_ps[copy(pw)] = coeff
        return PauliSentence(copied_ps)

    def __deepcopy__(self, memo):
        res = self.__copy__()
        memo[id(self)] = res
        return res

    def __matmul__(self, other):
        """Matrix / tensor product between two PauliSentences by iterating over each sentence and multiplying
        the Pauli words pair-wise"""
        if isinstance(other, PauliWord):
            other = PauliSentence({other: 1.0})

        final_ps = PauliSentence()

        if len(self) == 0 or len(other) == 0:
            return final_ps

        for pw1 in self:
            for pw2 in other:
                prod_pw, coeff = pw1._matmul(pw2)
                final_ps[prod_pw] = final_ps[prod_pw] + coeff * self[pw1] * other[pw2]

        return final_ps

    def __mul__(self, other):
        """Multiply a PauliWord by a scalar

        Args:
            other (Scalar): The scalar to multiply the PauliWord with

        Returns:
            PauliSentence
        """

        if isinstance(other, TensorLike):
            if not qml.math.ndim(other) == 0:
                raise ValueError(
                    f"Attempting to multiply a PauliSentence with an array of dimension {qml.math.ndim(other)}"
                )

            return PauliSentence({key: other * value for key, value in self.items()})

        raise TypeError(
            f"PauliSentence can only be multiplied by numerical data. Attempting to multiply by {other} of type {type(other)}"
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Divide a PauliSentence by a scalar"""
        if isinstance(other, TensorLike):
            return self * (1 / other)
        raise TypeError(
            f"PauliSentence can only be divided by numerical data. Attempting to divide by {other} of type {type(other)}"
        )

    def commutator(self, other):
        """
        Compute commutator between a ``PauliSentence`` :math:`P` and other operator :math:`O`

        .. math:: [P, O] = P O - O P

        When the other operator is a :class:`~PauliWord` or :class:`~PauliSentence`,
        this method is faster than computing ``P @ O - O @ P``. It is what is being used
        in :func:`~commutator` when setting ``pauli=True``.

        Args:
            other (Union[Operator, PauliWord, PauliSentence]): Second operator

        Returns:
            ~PauliSentence: The commutator result in form of a :class:`~PauliSentence` instances.

        **Examples**

        You can compute commutators between :class:`~PauliSentence` instances.

        >>> pw1 = PauliWord({0:"X"})
        >>> pw2 = PauliWord({1:"X"})
        >>> ps1 = PauliSentence({pw1: 1., pw2: 2.})
        >>> ps2 = PauliSentence({pw1: 0.5j, pw2: 1j})
        >>> ps1.commutator(ps2)
        0 * I

        You can also compute the commutator with other operator types if they have a Pauli representation.

        >>> ps1.commutator(qml.Y(0))
        2j * Z(0)"""
        final_ps = PauliSentence()

        if isinstance(other, PauliWord):
            for pw1 in self:
                comm_pw, coeff = pw1._commutator(other)
                if len(comm_pw) != 0:
                    final_ps[comm_pw] += coeff * self[pw1]

            return final_ps

        if not isinstance(other, PauliSentence):
            if other.pauli_rep is None:
                raise NotImplementedError(
                    f"Cannot compute a native commutator of a Pauli word or sentence with the operator {other} of type {type(other)}."
                    f"You can try to use qml.commutator(op1, op2, pauli=False) instead."
                )
            other = qml.pauli.pauli_sentence(other)

        for pw1 in self:
            for pw2 in other:
                comm_pw, coeff = pw1._commutator(pw2)
                if len(comm_pw) != 0:
                    final_ps[comm_pw] += coeff * self[pw1] * other[pw2]

        return final_ps

    def __str__(self):
        """String representation of the PauliSentence."""
        if len(self) == 0:
            return "0 * I"
        return "\n+ ".join(f"{coeff} * {str(pw)}" for pw, coeff in self.items())

    def __repr__(self):
        """Terminal representation for PauliSentence"""
        return str(self)

    @property
    def wires(self):
        """Track wires of the PauliSentence."""
        return Wires.all_wires((pw.wires for pw in self.keys()))

    def to_mat(self, wire_order=None, format="dense", buffer_size=None):
        """Returns the matrix representation.

        Keyword Args:
            wire_order (iterable or None): The order of qubits in the tensor product.
            format (str): The format of the matrix. It is "dense" by default. Use "csr" for sparse.
            buffer_size (int or None): The maximum allowed memory in bytes to store intermediate results
                in the calculation of sparse matrices. It defaults to ``2 ** 30`` bytes that make
                1GB of memory. In general, larger buffers allow faster computations.

        Returns:
            (Union[NumpyArray, ScipySparseArray]): Matrix representation of the Pauli sentence.

        Raises:
            ValueError: Can't get the matrix of an empty PauliSentence.
        """
        wire_order = self.wires if wire_order is None else Wires(wire_order)
        if len(self) == 0:
            n = len(wire_order) if wire_order is not None else 0
            if format == "dense":
                return np.zeros((2**n, 2**n))
            return sparse.csr_matrix((2**n, 2**n), dtype="complex128")

        if format == "dense":
            return self._to_dense_mat(wire_order)
        return self._to_sparse_mat(wire_order, buffer_size=buffer_size)

    def _to_sparse_mat(self, wire_order, buffer_size=None):
        """Compute the sparse matrix of the Pauli sentence by efficiently adding the Pauli words
        that it is composed of. See pauli_sparse_matrices.md for the technical details."""
        pauli_words = list(self)  # Ensure consistent ordering
        n_wires = len(wire_order)
        matrix_size = 2**n_wires
        matrix = sparse.csr_matrix((matrix_size, matrix_size), dtype="complex128")
        op_sparse_idx = _ps_to_sparse_index(pauli_words, wire_order)
        _, unique_sparse_structures, unique_invs = np.unique(
            op_sparse_idx, axis=0, return_index=True, return_inverse=True
        )
        pw_sparse_structures = unique_sparse_structures[unique_invs]

        buffer_size = buffer_size or 2**30  # Default to 1GB of memory
        # Convert bytes to number of matrices:
        # complex128 (16) for each data entry and int64 (8) for each indices entry
        buffer_size = max(1, buffer_size // ((16 + 8) * matrix_size))
        mat_data = np.empty((matrix_size, buffer_size), dtype=np.complex128)
        mat_indices = np.empty((matrix_size, buffer_size), dtype=np.int64)
        n_matrices_in_buffer = 0
        for sparse_structure in unique_sparse_structures:
            indices, *_ = np.nonzero(pw_sparse_structures == sparse_structure)
            mat = self._sum_same_structure_pws([pauli_words[i] for i in indices], wire_order)
            mat_data[:, n_matrices_in_buffer] = mat.data
            mat_indices[:, n_matrices_in_buffer] = mat.indices

            n_matrices_in_buffer += 1
            if n_matrices_in_buffer == buffer_size:
                # Add partial results in batches to control the memory usage
                matrix += self._sum_different_structure_pws(mat_indices, mat_data)
                n_matrices_in_buffer = 0

        matrix += self._sum_different_structure_pws(
            mat_indices[:, :n_matrices_in_buffer], mat_data[:, :n_matrices_in_buffer]
        )
        matrix.eliminate_zeros()
        return matrix

    def _to_dense_mat(self, wire_order):
        """Compute the dense matrix of the Pauli sentence by efficiently adding the Pauli words
        that it is composed of. See pauli_sparse_matrices.md for the technical details."""
        pauli_words = list(self)  # Ensure consistent ordering

        try:
            op_sparse_idx = _ps_to_sparse_index(pauli_words, wire_order)
        except qml.wires.WireError as e:
            raise ValueError(
                "Can't get the matrix for the specified wire order because it "
                f"does not contain all the Pauli sentence's wires {self.wires}"
            ) from e
        _, unique_sparse_structures, unique_invs = np.unique(
            op_sparse_idx, axis=0, return_index=True, return_inverse=True
        )
        pw_sparse_structures = unique_sparse_structures[unique_invs]

        full_matrix = None
        for sparse_structure in unique_sparse_structures:
            indices, *_ = np.nonzero(pw_sparse_structures == sparse_structure)
            mat = self._sum_same_structure_pws_dense([pauli_words[i] for i in indices], wire_order)

            full_matrix = mat if full_matrix is None else qml.math.add(full_matrix, mat)
        return full_matrix

    def dot(self, vector, wire_order=None):
        """Computes the matrix-vector product of the Pauli sentence with a state vector.
        See pauli_sparse_matrices.md for the technical details."""
        wire_order = self.wires if wire_order is None else Wires(wire_order)
        if not wire_order.contains_wires(self.wires):
            raise ValueError(
                "Can't get the matrix for the specified wire order because it "
                f"does not contain all the Pauli sentence's wires {self.wires}"
            )
        pauli_words = list(self)  # Ensure consistent ordering
        op_sparse_idx = _ps_to_sparse_index(pauli_words, wire_order)
        _, unique_sparse_structures, unique_invs = np.unique(
            op_sparse_idx, axis=0, return_index=True, return_inverse=True
        )
        pw_sparse_structures = unique_sparse_structures[unique_invs]

        dtype = np.complex64 if vector.dtype in (np.float32, np.complex64) else np.complex128
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        mv = np.zeros_like(vector, dtype=dtype)
        for sparse_structure in unique_sparse_structures:
            indices, *_ = np.nonzero(pw_sparse_structures == sparse_structure)
            entries, data = self._get_same_structure_csr(
                [pauli_words[i] for i in indices], wire_order
            )
            mv += vector[:, entries] * data.reshape(1, -1)
        return mv.reshape(vector.shape)

    def _get_same_structure_csr(self, pauli_words, wire_order):
        """Returns the CSR indices and data for Pauli words with the same sparse structure."""
        indices = pauli_words[0]._get_csr_indices(wire_order)
        nwires = len(wire_order)
        nwords = len(pauli_words)
        inner = np.empty((nwords, 2 ** (nwires - nwires // 2)), dtype=np.complex128)
        outer = np.empty((nwords, 2 ** (nwires // 2)), dtype=np.complex128)
        for i, word in enumerate(pauli_words):
            outer[i, :], inner[i, :] = word._get_csr_data_2(
                wire_order, coeff=qml.math.to_numpy(self[word])
            )
        data = outer.T @ inner
        return indices, data.ravel()

    def _sum_same_structure_pws_dense(self, pauli_words, wire_order):
        matrix_size = 2 ** (len(wire_order))
        base_matrix = sparse.csr_matrix((matrix_size, matrix_size), dtype="complex128")

        data0 = pauli_words[0]._get_csr_data(wire_order, 1)
        base_matrix.data = np.ones_like(data0)
        base_matrix.indices = pauli_words[0]._get_csr_indices(wire_order)
        base_matrix.indptr = _cached_arange(
            matrix_size + 1
        )  # Non-zero entries by row (starting from 0)
        base_matrix = base_matrix.toarray()
        coeff = self[pauli_words[0]]
        ml_interface = qml.math.get_interface(coeff)
        if ml_interface == "torch":
            data0 = qml.math.convert_like(data0, coeff)
        data = coeff * data0
        for pw in pauli_words[1:]:
            coeff = self[pw]
            csr_data = pw._get_csr_data(wire_order, 1)
            ml_interface = qml.math.get_interface(coeff)
            if ml_interface == "torch":
                csr_data = qml.math.convert_like(csr_data, coeff)
            data += self[pw] * csr_data

        return qml.math.einsum("ij,i->ij", base_matrix, data)

    def _sum_same_structure_pws(self, pauli_words, wire_order):
        """Sums Pauli words with the same sparse structure."""
        mat = pauli_words[0].to_mat(
            wire_order, coeff=qml.math.to_numpy(self[pauli_words[0]]), format="csr"
        )
        for word in pauli_words[1:]:
            mat.data += word.to_mat(
                wire_order, coeff=qml.math.to_numpy(self[word]), format="csr"
            ).data
        return mat

    @staticmethod
    def _sum_different_structure_pws(indices, data):
        """Sums Pauli words with different parse structures."""
        size = indices.shape[0]
        idx = np.argsort(indices, axis=1)
        matrix = sparse.csr_matrix((size, size), dtype="complex128")
        matrix.indices = np.take_along_axis(indices, idx, axis=1).ravel()
        matrix.data = np.take_along_axis(data, idx, axis=1).ravel()
        num_entries_per_row = indices.shape[1]
        matrix.indptr = _cached_arange(size + 1) * num_entries_per_row

        # remove zeros and things sufficiently close to zero
        matrix.data[np.abs(matrix.data) < 1e-16] = 0  # Faster than np.isclose(matrix.data, 0)
        matrix.eliminate_zeros()
        return matrix

    def operation(self, wire_order=None):
        """Returns a native PennyLane :class:`~pennylane.operation.Operation` representing the PauliSentence."""
        if len(self) == 0:
            return qml.s_prod(0, Identity(wires=wire_order))

        summands = []
        wire_order = wire_order or self.wires
        for pw, coeff in self.items():
            pw_op = pw.operation(wire_order=list(wire_order))
            rep = PauliSentence({pw: coeff})
            summands.append(pw_op if coeff == 1 else SProd(coeff, pw_op, _pauli_rep=rep))
        return summands[0] if len(summands) == 1 else Sum(*summands, _pauli_rep=self)

    def hamiltonian(self, wire_order=None):
        """Returns a native PennyLane :class:`~pennylane.Hamiltonian` representing the PauliSentence."""
        if len(self) == 0:
            if wire_order in (None, [], Wires([])):
                raise ValueError("Can't get the Hamiltonian for an empty PauliSentence.")
            return qml.Hamiltonian([], [])

        wire_order = wire_order or self.wires
        wire_order = list(wire_order)

        return qml.Hamiltonian(
            list(self.values()),
            [pw.operation(wire_order=wire_order, get_as_tensor=True) for pw in self],
        )

    def simplify(self, tol=1e-8):
        """Remove any PauliWords in the PauliSentence with coefficients less than the threshold tolerance."""
        items = list(self.items())
        for pw, coeff in items:
            if abs(coeff) <= tol:
                del self[pw]
        if len(self) == 0:
            self = PauliSentence({})  # pylint: disable=self-cls-assignment

    def map_wires(self, wire_map: dict) -> "PauliSentence":
        """Return a new PauliSentence with the wires mapped."""
        return self.__class__({pw.map_wires(wire_map): coeff for pw, coeff in self.items()})
