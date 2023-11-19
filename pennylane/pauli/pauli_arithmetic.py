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
from warnings import warn

import numpy as np
from scipy import sparse
from scipy.linalg import svd

import pennylane as qml
from pennylane import math
from pennylane.wires import Wires
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

anticom_map = {
    I: {I: 0, X: 0, Y: 0, Z: 0},
    X: {I: 0, X: 0, Y: 1, Z: 1},
    Y: {I: 0, X: 1, Y: 0, Z: 1},
    Z: {I: 0, X: 1, Y: 1, Z: 0},
}
anticom_set = {"XY", "YX", "YZ", "ZY","XZ", "ZX"}


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
        self._hash = hash(frozenset(self.items()))

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
        return self._hash
        return hash(frozenset(self.items()))

    @lru_cache
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
                factor, new_op = mul_map[base[wire]][term]
                if new_op == I:
                    del result[wire]
                else:
                    coeff *= -factor if swapped else factor
                    result[wire] = new_op
            else:
                result[wire] = term

        return PauliWord(result), coeff

    def commutes_with(self, other):
        wires = set(self) & set(other)
        if not wires:
            return True
        anticom_count = sum([anticom_map[self[wire]][other[wire]] for wire in wires])
        #anticom_count = sum((self[wire] + other[wire] in anticom_set) for wire in wires)
        return 1 - (anticom_count % 2)

    def __or__(self, other):
        """Commutator between two PauliWords"""
        if self.commutes_with(other):
            return None, 0.0
        new_word, coeff = self.__mul__(other)
        return new_word, 2 * coeff

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    def __gt__(self, other):
        return self.__str__() > other.__str__()

    def __str__(self):
        """String representation of a PauliWord."""
        if len(self) == 0:
            return "I"
        return " @ ".join(f"{op}({w})" for w, op in sorted(self.items()))

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
            if not wire_order:
                raise ValueError("Can't get the matrix of an empty PauliWord.")
            return (
                np.diag([coeff] * 2 ** len(wire_order))
                if format == "dense"
                else coeff * sparse.eye(2 ** len(wire_order), format=format, dtype="complex128")
            )

        if format == "dense":
            return coeff * reduce(math.kron, (mat_map[self[w]] for w in wire_order))

        return self._to_sparse_mat(wire_order, coeff)

    def _to_sparse_mat(self, wire_order, coeff):
        """Compute the sparse matrix of the Pauli word times a coefficient, given a wire order.
        See pauli_sparse_matrices.md for the technical details of the implementation."""
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
        matrix = sparse.csr_matrix((matrix_size, matrix_size), dtype="complex128")
        matrix.data, matrix.indices, matrix.indptr = data, indices, indptr
        return matrix

    def operation(self, wire_order=None, get_as_tensor=False):
        """Returns a native PennyLane :class:`~pennylane.operation.Operation` representing the PauliWord."""
        if len(self) == 0:
            if wire_order in (None, [], Wires([])):
                raise ValueError("Can't get the operation for an empty PauliWord.")
            return Identity(wires=wire_order)

        factors = [op_map[op](wire) for wire, op in self.items()]

        if get_as_tensor:
            return factors[0] if len(factors) == 1 else Tensor(*factors)
        return factors[0] if len(factors) == 1 else prod(*factors)

    def hamiltonian(self, wire_order=None):
        """Return :class:`~pennylane.Hamiltonian` representing the PauliWord."""
        if len(self) == 0:
            if wire_order in (None, [], Wires([])):
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

    def __iadd__(self, other):
        """Inplace addition of two Pauli sentence together by adding terms of other to self"""
        for key in other:
            if key in self:
                self[key] += other[key]
            else:
                self[key] = other[key]
        return self

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

    def __or__(self, other):
        """commute two Pauli sentences by iterating over each sentence and commuting
        the Pauli words pair-wise"""
        final_ps = PauliSentence()

        if len(self) == 0:
            return copy(other)

        if len(other) == 0:
            return copy(self)

        for pw1 in self:
            for pw2 in other:
                comm_pw, coeff = pw1 | pw2
                if comm_pw is not None:
                    final_ps[comm_pw] = final_ps[comm_pw] + coeff * self[pw1] * other[pw2]

        return final_ps

    # def __hash__(self):
    #     return hash(str())

    def __eq__(self, other):
        """Loose definition of equality: proportionality"""
        # Step 1: Check if keys are the same
        if set(self.keys()) != set(other.keys()):
            return False

        # Step 2: Check if values are proportional
        # Find the proportionality factor by comparing the first pair of values
        key = next(iter(self.keys()))
        factor = other[key] / self[key]

        # Check if all values are proportional by the same factor
        return all(np.isclose(v1 * factor, other[k]) for k, v1 in self.items())

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
        return Wires(set().union(*(pw.wires for pw in self.keys())))

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

        Rasies:
            ValueError: Can't get the matrix of an empty PauliSentence.
        """
        wire_order = self.wires if wire_order is None else Wires(wire_order)
        if not wire_order.contains_wires(self.wires):
            raise ValueError(
                "Can't get the matrix for the specified wire order because it "
                f"does not contain all the Pauli sentence's wires {self.wires}"
            )

        def _pw_wires(w: Iterable) -> Wires:
            """Return the native Wires instance for a list of wire labels.
            w represents the wires of the PauliWord being processed. In case
            the PauliWord is empty ({}), choose any arbitrary wire from the
            PauliSentence it is composed in.
            """
            return w or Wires(self.wires[0]) if self.wires else self.wires

        if len(self) == 0:
            if not wire_order:
                raise ValueError("Can't get the matrix of an empty PauliSentence.")
            if format == "dense":
                return np.eye(2 ** len(wire_order))
            return sparse.eye(2 ** len(wire_order), format=format, dtype="complex128")

        if format != "dense":
            return self._to_sparse_mat(wire_order, buffer_size=buffer_size)

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
            if wire_order in (None, [], Wires([])):
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
            if wire_order in (None, [], Wires([])):
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


def all_commutators(oplist):
    """
    Compute the commutator of all elements with all elements and keep only the unique ones (up to proportionality constants)

    Args:
        oplist: A list of PauliSentence operators

    Returns:
        List: A list of all unique operators and their commutators.
    """
    # TODO: implement for pure PauliWords
    out = copy(oplist)
    empty_keys = PauliSentence().keys()  # hacky

    for i, pauli1 in enumerate(oplist[:-1]):
        for pauli2 in oplist[i + 1 :]:
            q = pauli1 | pauli2
            if len(q) == 0 or q in out:
                continue
            out.append(q)

    return out


def lie_closure(oplist, size_limit=1e5, print_progress=False):
    """
    Repeatedly compute all commutators of the operator set until no new elements are added
    and the set is algebraically closed.

    Args:
        oplist: list of ``PauliSentence``
        size_limit (int):
        max_batch_size (bool | int): Batch over generators to reduce memory requirements. This is
            only relevant for large DLAs. A good value to start testing is 200, but for very large
            sytems smaller values might be necessary.

    Kwargs:
        print_progress(bool): Print size of dla after each iteration.
    """

    numpaulis = len(oplist)
    counter = 0
    while True:
        oplist = all_commutators(oplist)

        numpaulis_new = len(oplist)
        if numpaulis_new == numpaulis:
            break
        if numpaulis_new > size_limit:
            warn("Size limit was reached. Aborting.", UserWarning)
            break
        numpaulis = numpaulis_new
        counter += 1

        if print_progress:
            print(f"Iteration {counter} done. {numpaulis = }")

    return oplist


def _add_if_independent(M, pauli_sentence, pw_to_idx, rank, num_pw):
    """Utility function for ``lie_closure_alt``.
    Adds a column, corresponding to a new PauliSentence in the linearly independent set (LIS),
    to the matrix ``M`` if it is linearly independent from the existing columns.

    Args:
        M (ndarray): coefficient matrix for current LIS
        pauli_sentence (`~.PauliSentence`): Pauli sentence for which to add a column if independent
        pw_to_idx (dict): map from ``PauliWord`` to row index in ``M``
        rank (int): current rank of ``M``, equal to its number of columns
        num_pw (int): current number of ``PauliWord``\ s, equal to the number of rows in ``M``

    Returns:
        ndarray: updated coefficient matrix for the LIS
        dict: updated map from ``PauliWord`` to row index in ``M``. Includes new ``PauliWord`` keys
            from the input ``pauli_sentence`` if it was linearly independent
        int: updated rank/number of columns of ``M``
        int: updated number of ``PauliWord``\ s/number of rows of ``M``
        bool: whether ``pauli_sentence`` was linearly independent and its column was added to ``M``
    """
    new_pws = [pw for pw in pauli_sentence.keys() if pw not in pw_to_idx]
    new_num_pw = num_pw + len(new_pws)

    if new_num_pw < rank + 1:
        # Can't span rank+1 independent vectors in fewer than rank+1 dimensions
        # The input PauliSentence must have been linearly dependent
        return M, pw_to_idx, rank, num_pw, False

    # Sparse alternative
    # M.resize((new_num_pw, rank + 1))

    M = np.pad(M, ((0, new_num_pw - num_pw), (0, 1)))
    # M = np.concatenate([np.concatenate([M, np.zeros((num_pw, 1))], axis=1), np.zeros((new_num_pw-num_pw, rank+1))], axis=0)


    # If we actually had to add PauliWords (rows) to our basis, the new PauliSentence must have been
    # linearly independent from all others
    if new_num_pw > num_pw:
        new_pw_to_idx = copy(pw_to_idx)
        for i, pw in enumerate(new_pws, start=num_pw):
            new_pw_to_idx[pw] = i
        # Add new PauliSentence entries to matrix
        for pw, value in pauli_sentence.items():
            M[new_pw_to_idx[pw], rank] = value

        return M, new_pw_to_idx, rank + 1, new_num_pw, True

    # Add new PauliSentence entries to matrix
    for pw, value in pauli_sentence.items():
        M[pw_to_idx[pw], rank] = value

    # Sparse alternative
    # sing_value = svds(M, k=1, which="SM", return_singular_vectors=False)
    # if np.allclose(sing_value, 0.):

    new_rank = np.linalg.matrix_rank(M)
    # Manual singular value alternative, probably slower than ``matrix_rank``
    # sing_value = np.min(np.abs(svd(M, compute_uv=False, lapack_driver="gesdd", check_finite=False)))
    if new_rank == rank:
        # Sparse alternative
        # M.resize((num_pw, rank))
        M = M[:num_pw, :rank]
        return M, pw_to_idx, rank, num_pw, False

    return M, pw_to_idx, rank + 1, new_num_pw, True


def _simplify_and_remove_empty(ps_list):
    """Clean a list of PauliSentences by removing empty Sentences. Used in ``lie_closure_alt``."""
    new_ps_list = []
    for ps in ps_list:
        ps.simplify()
        if len(ps) > 0:
            new_ps_list.append(ps)
    return new_ps_list


def lie_closure_alt(generating_set, clean_input=True, orthonormalize=False):
    r"""Compute the Lie closure of a set of ``PauliSentence``\ s, using a representation in a
    Pauli sub-basis.
    Sparsified version of Algorithm 1 in `arXiv:1012.5256<https://arxiv.org/abs/1012.5256>`__.

    Args:
        generating_set (list[`~.PauliSentence`]): generating set for which to compute the
            Lie closure
        clean_input (bool): whether to remove empty instances from ``generating_set``
        orthonormalize (bool): whether to form an orthonormal basis from the Lie closure.
            If ``True``, the Pauli sentences in the output will be orthonormal with respect to
            the inner product defined below.

    Returns:
        list[`~.PauliSentence`]: a basis of ``PauliSentence`` instances that is closed under
        commutators (Lie closure). If ``orthonormalize=True``, it is an orthonormal basis (ONB).

    """
    if clean_input:
        # Simplify PauliSentences and remove the empty ones if desired
        generating_set = _simplify_and_remove_empty(generating_set)

    # Get all Pauli words that are present in at least one Pauli sentence
    all_pws = list(reduce(set.__or__, [set(ps.keys()) for ps in generating_set]))
    num_pw = len(all_pws)
    # Create a dictionary mapping from PauliWord to row index
    pw_to_idx = {pw: i for i, pw in enumerate(all_pws)}

    # List all linearly independent ``PauliSentence`` objects. The first element
    # always is independent, and the initial rank will be 1, correspondingly.
    basis = [generating_set[0]]
    rank = 1

    # Sparse alternative
    # M = dok_array((num_pw, rank), dtype=float)
    M = np.zeros((num_pw, rank), dtype=float)
    # Add the values of the first PauliSentence to the sparse array.
    for pw, value in generating_set[0].items():
        M[pw_to_idx[pw], 0] = value

    # Add the values of all other PauliSentence objects from the input to the sparse array,
    # but only if they are linearly independent from the previous objects.
    for ps_idx, ps in enumerate(generating_set[1:]):
        M, pw_to_idx, rank, num_pw, added = _add_if_independent(M, ps, pw_to_idx, rank, num_pw)
        if added:
            # After adding a linearly independent PauliSentence to the matrix, remember it.
            basis.append(ps)

    # Dummy initial value for some comparison length
    old_length = 0
    # Current number of linearly independent PauliSentences
    new_length = len(basis)
    while new_length > old_length:  # Should add a maximal length criterion here.
        for idx1 in range(new_length - 1):
            ps1 = basis[idx1]
            for idx2 in range(max([idx1 + 1, old_length]), new_length):
                ps2 = basis[idx2]
                com = ps1 | ps2
                if len(com) == 0:
                    continue
                # The commutator is taken per pair of PauliWords, which collect purely imaginary
                # coefficients. We will only consider the real-value imaginary part here.
                for pw, val in com.items():
                    com[pw] = np.imag(val)
                # Add commutator to sparse array if it is linearly independent
                M, pw_to_idx, rank, num_pw, added = _add_if_independent(
                    M, com, pw_to_idx, rank, num_pw
                )
                # After adding a linearly independent commutator to the matrix, remember it.
                if added:
                    basis.append(com)

        # Updated number of linearly independent PauliSentences from previous and current step
        old_length = new_length
        new_length = len(basis)

    if orthonormalize:
        # Use the matrix M and the fact that the Pauli basis is orthonormal to orthonormalize
        # the basis
        raise NotImplementedError("to do")

    return basis


def compact_str(ps_list, decimals=1):
    """Write a compact string representation of a list of ``PauliSentence``\ s."""
    max_wire = max([max(ps.wires) for ps in ps_list])
    wire_map = {i: i for i in range(max_wire + 1)}
    out = "[\n"
    for ps in ps_list:
        terms = [
            f"{val: 2.{decimals}f} {qml.pauli.pauli_word_to_string(pw.operation(wire_map), wire_map).replace('I', '_')}"
            for pw, val in sorted(ps.items())
        ]
        out += " " * 4 + " + ".join(terms) + "\n"
    out += "]"
    return out
