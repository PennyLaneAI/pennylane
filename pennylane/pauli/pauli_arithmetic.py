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
from copy import copy
from functools import reduce, lru_cache
from typing import Iterable

import numpy as np
from scipy import sparse

import pennylane as qml
from pennylane import math, wires
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, prod, s_prod

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
    """Immutable dictionary used to represent a Pauli Word,
    associating wires with their respective operators.
    Can be constructed from a standard dictionary.

    >>> w = PauliWord({"a": 'X', 2: 'Y', 3: 'Z'})
    >>> w
    X(a) @ Y(2) @ Z(3)
    """

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

    def __mul__(self, other):
        """Multiply two Pauli words together using the matrix product if wires overlap
        and the tensor product otherwise.

        Args:
            other (PauliWord): The Pauli word to multiply with

        Returns:
            result(PauliWord): The resulting operator of the multiplication
            coeff(complex): The complex phase factor
        """
        base, iterator, swapped = (
            (self, other, False) if len(self) > len(other) else (other, self, True)
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
        return set(self)

    def to_mat(self, wire_order, format="dense", coeff=1.0):
        """Returns the matrix representation.

        Keyword Args:
            wire_order (iterable or None): The order of qubits in the tensor product.
            format (str): The format of the matrix ("dense" by default), if not a dense
                matrix, then the format for the sparse representation of the matrix.
            coeff (float): Coefficient multiplying the resulting sparse matrix.

        Returns:
            (Union[NumpyArray, ScipySparseArray]): Matrix representation of the Pauliword.

        Raises:
            ValueError: Can't get the matrix of an empty PauliWord.
        """
        if len(self) == 0:
            if wire_order is None or wire_order == wires.Wires([]):
                raise ValueError("Can't get the matrix of an empty PauliWord.")
            return (
                np.diag([coeff] * 2 ** len(wire_order))
                if format == "dense"
                else coeff * sparse.eye(2 ** len(wire_order), format=format)
            )

        if format == "dense":
            return coeff * reduce(math.kron, (mat_map[self[w]] for w in wire_order))

        return self._to_sparse_mat(wire_order, coeff)

    def _to_sparse_mat(self, wire_order, coeff):
        """Compute the sparse matrix of the Pauli word times a coefficient, given a wire order.
        See pauli_word_sparse_matrix.md for the technical details of the implementation."""
        full_word = [self[wire] for wire in wire_order]
        matrix_size = 2 ** len(wire_order)
        data = np.empty(matrix_size, dtype=np.complex128)  # Non-zero values
        indices = np.empty(matrix_size, dtype=np.int64)  # Column index of non-zero values
        indptr = _cached_arange(matrix_size + 1)  # Non-zero entries by row (starting from 0)

        current_size = 2
        data[:current_size], indices[:current_size] = _cached_sparse_data(full_word[-1])
        data[:current_size] *= coeff  # Multiply initial term better than the full matrix
        for s in full_word[-2::-1]:
            if s == "I":
                data[current_size : 2 * current_size] = data[:current_size]
                indices[current_size : 2 * current_size] = indices[:current_size] + current_size
            elif s == "X":
                data[current_size : 2 * current_size] = data[:current_size]
                indices[current_size : 2 * current_size] = indices[:current_size]
                indices[:current_size] += current_size
            elif s == "Y":
                data[current_size : 2 * current_size] = 1j * data[:current_size]
                data[:current_size] *= -1j
                indices[current_size : 2 * current_size] = indices[:current_size]
                indices[:current_size] += current_size
            elif s == "Z":
                data[current_size : 2 * current_size] = -data[:current_size]
                indices[current_size : 2 * current_size] = indices[:current_size] + current_size
            current_size *= 2
        # Avoid checks and copies in __init__ by directly setting the attributes of an empty matrix
        matrix = sparse.csr_matrix((matrix_size, matrix_size), dtype=np.complex128)
        matrix.data, matrix.indices, matrix.indptr = data, indices, indptr
        return matrix

    def operation(self, wire_order=None, get_as_tensor=False):
        """Returns a native PennyLane :class:`~pennylane.operation.Operation` representing the PauliWord."""
        if len(self) == 0:
            if wire_order in (None, [], wires.Wires([])):
                raise ValueError("Can't get the operation for an empty PauliWord.")
            return Identity(wires=wire_order)

        factors = [op_map[op](wire) for wire, op in self.items()]

        if get_as_tensor:
            return factors[0] if len(factors) == 1 else Tensor(*factors)
        return factors[0] if len(factors) == 1 else prod(*factors)

    def hamiltonian(self, wire_order=None):
        """Return :class:`~pennylane.Hamiltonian` representing the PauliWord."""
        if len(self) == 0:
            if wire_order in (None, [], wires.Wires([])):
                raise ValueError("Can't get the Hamiltonian for an empty PauliWord.")
            return Hamiltonian([1], [Identity(wires=wire_order)])

        obs = [op_map[op](wire) for wire, op in self.items()]
        return Hamiltonian([1], [obs[0] if len(obs) == 1 else Tensor(*obs)])

    def map_wires(self, wire_map: dict) -> "PauliWord":
        """Return a new PauliWord with the wires mapped."""
        return self.__class__({wire_map.get(w, w): op for w, op in self.items()})


class PauliSentence(dict):
    """Dictionary representing a linear combination of Pauli words, with the keys
    as PauliWord instances and the values correspond to coefficients.

    >>> ps = qml.pauli.PauliSentence({
            qml.pauli.PauliWord({0:'X', 1:'Y'}): 1.23,
            qml.pauli.PauliWord({2:'Z', 0:'Y'}): -0.45j
        })
    >>> ps
    1.23 * X(0) @ Y(1)
    + (-0-0.45j) * Z(2) @ Y(0)
    """

    def __missing__(self, key):
        """If the PauliWord is not in the sentence then the coefficient
        associated with it should be 0."""
        return 0.0

    def __add__(self, other):
        """Add two Pauli sentence together by iterating over the smaller
        one and adding its terms to the larger one."""
        smaller_ps, larger_ps = (
            (self, copy(other)) if len(self) < len(other) else (other, copy(self))
        )
        for key in smaller_ps:
            larger_ps[key] += smaller_ps[key]

        return larger_ps

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

    def __mul__(self, other):
        """Multiply two Pauli sentences by iterating over each sentence and multiplying
        the Pauli words pair-wise"""
        final_ps = PauliSentence()

        if len(self) == 0:
            return copy(other)

        if len(other) == 0:
            return copy(self)

        for pw1 in self:
            for pw2 in other:
                prod_pw, coeff = pw1 * pw2
                final_ps[prod_pw] = final_ps[prod_pw] + coeff * self[pw1] * other[pw2]

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
        return set().union(*(pw.wires for pw in self.keys()))

    def to_mat(self, wire_order, format="dense"):
        """Returns the matrix representation.

        Keyword Args:
            wire_order (iterable or None): The order of qubits in the tensor product.
            format (str): The format of the matrix ("dense" by default), if not a dense
                matrix, then the format for the sparse representation of the matrix.

        Returns:
            (Union[NumpyArray, ScipySparseArray]): Matrix representation of the PauliSentence.

        Rasies:
            ValueError: Can't get the matrix of an empty PauliSentence.
        """

        def _pw_wires(w: Iterable) -> wires.Wires:
            """Return the native Wires instance for a list of wire labels.
            w represents the wires of the PauliWord being processed. In case
            the PauliWord is empty ({}), choose any arbitrary wire from the
            PauliSentence it is composed in.
            """
            if w:
                return wires.Wires(w)

            return wires.Wires(list(self.wires)[0]) if len(self.wires) > 0 else wires.Wires([])

        if len(self) == 0:
            if wire_order is None or wire_order == wires.Wires([]):
                raise ValueError("Can't get the matrix of an empty PauliSentence.")
            if format == "dense":
                return np.eye(2 ** len(wire_order))
            return sparse.eye(2 ** len(wire_order), format=format)

        mats_and_wires_gen = (
            (
                coeff * pw.to_mat(wire_order=_pw_wires(pw.wires), format=format),
                _pw_wires(pw.wires),
            )
            for pw, coeff in self.items()
        )

        reduced_mat, result_wire_order = math.reduce_matrices(
            mats_and_wires_gen=mats_and_wires_gen, reduce_func=math.add
        )

        return math.expand_matrix(reduced_mat, result_wire_order, wire_order=wire_order)

    def operation(self, wire_order=None):
        """Returns a native PennyLane :class:`~pennylane.operation.Operation` representing the PauliSentence."""
        if len(self) == 0:
            if wire_order in (None, [], wires.Wires([])):
                raise ValueError("Can't get the operation for an empty PauliSentence.")
            return qml.s_prod(0, Identity(wires=wire_order))

        summands = []
        wire_order = wire_order or self.wires
        for pw, coeff in self.items():
            pw_op = pw.operation(wire_order=list(wire_order))
            summands.append(pw_op if coeff == 1 else s_prod(coeff, pw_op))
        return summands[0] if len(summands) == 1 else qml.sum(*summands)

    def hamiltonian(self, wire_order=None):
        """Returns a native PennyLane :class:`~pennylane.Hamiltonian` representing the PauliSentence."""
        if len(self) == 0:
            if wire_order in (None, [], wires.Wires([])):
                raise ValueError("Can't get the Hamiltonian for an empty PauliSentence.")
            return Hamiltonian([], [])

        wire_order = wire_order or self.wires
        wire_order = list(wire_order)

        return Hamiltonian(
            list(self.values()),
            [pw.operation(wire_order=wire_order, get_as_tensor=True) for pw in self],
        )

    def simplify(self, tol=1e-8):
        """Remove any PauliWords in the PauliSentence with coefficients less than the threshold tolerance."""
        items = list(self.items())
        for pw, coeff in items:
            if abs(coeff) <= tol:
                del self[pw]

    def map_wires(self, wire_map: dict) -> "PauliSentence":
        """Return a new PauliSentence with the wires mapped."""
        return self.__class__({pw.map_wires(wire_map): coeff for pw, coeff in self.items()})
