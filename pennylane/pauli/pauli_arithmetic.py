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
import numpy as np
from scipy import sparse
from pennylane import math

I = "I"
X = "X"
Y = "Y"
Z = "Z"

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

sparse_matI = sparse.eye(2, format="csr")
sparse_matX = sparse.csr_matrix([[0, 1], [1, 0]])
sparse_matY = sparse.csr_matrix([[0, -1j], [1j, 0]])
sparse_matZ = sparse.csr_matrix([[1, 0], [0, -1]])

sparse_mat_map = {
    I: sparse_matI,
    X: sparse_matX,
    Y: sparse_matY,
    Z: sparse_matZ,
}

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
    """Immutable dictionary used to represent a Pauli Word.
    Can be constructed from a standard dictionary.

    >>> w = PauliWord({"a": X, 2: Y, 3: Z})
    """

    def __missing__(self, key):
        """If the wire is not in the Pauli word,
        then no operator acts on it, so return the Identity."""
        return I

    def __setitem__(self, key, item):
        raise NotImplementedError

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
        result, iterator = (dict(self), other) if len(self) > len(other) else (dict(other), self)
        coeff = 1

        for wire, term in iterator.items():
            if wire in result:
                factor, new_op = mul_map[result[wire]][term]
                if new_op == I:
                    del result[wire]
                else:
                    coeff *= factor
                    result[wire] = new_op
            elif term != I:
                result[wire] = term

        return PauliWord(result), coeff

    @property
    def wires(self):
        """Track wires in a PauliWord."""
        return set(k for k in self.keys())

    def to_mat(self, wire_order=None, format="dense"):
        """Given a Pauli word, get the matrix representation.

        KeywordArgs:
            wire_order (iterable or None): The order of qubits in the tensor product.
            format (str): The format of the matrix ("dense" by default), if not a dense
                matrix, then the format for the sparse representation of the matrix.

        Returns:
            (Union[NumpyArray, ScipySparseArray]): Matrix representation of the Pauliword
        """
        matrix_map = sparse_mat_map if format != "dense" else mat_map
        if wire_order is None:
            wire_order = self.wires

        initial_wire = wire_order[0]
        result_matrix = matrix_map[self[initial_wire]]

        for wire in wire_order[1:]:
            result_matrix = math.kron(result_matrix, matrix_map[self[wire]])

        return result_matrix


class PauliSentence(dict):
    """Dict representing a Pauli Sentence. The keys are
    PauliWord instances and the values correspond to coefficients.

    >>> ps = PauliSentence({
            PauliWord({0:X, 1:Y}): 1.23
            PauliWord({2:Z, 0:Y}): -0.45j
        })
    """

    def __missing__(self, key):
        """If the PauliWord is not in the sentence then the coefficient
        associated with it should be 0."""
        return 0.0

    def __add__(self, other):
        """Add two Pauli sentence together by iterating over the smaller
        one and adding its terms to the larger one."""
        c_self, c_other = (self, other)
        smaller_ps, larger_ps = (c_self, c_other) if len(self) < len(other) else (c_other, c_self)
        for key in smaller_ps:
            larger_ps[key] += smaller_ps[key]

        return larger_ps

    def __mul__(self, other):
        """Multiply two Pauli sentences by iterating over each sentence and multiplying
        the Pauli words pair-wise"""
        final_ps = PauliSentence()

        for pw1 in self:
            for pw2 in other:
                prod_pw, coeff = pw1 * pw2
                final_ps[prod_pw] += coeff * self[pw1] * other[pw2]

        return final_ps

    def __str__(self):
        """String representation of the PauliSentence."""
        rep_str = ""
        for index, (pw, coeff) in enumerate(self.items()):
            if index == 0:
                rep_str += "= "
            else:
                rep_str += "+ "
            rep_str += f"({round(coeff, 2)}) * "
            for w, op in pw.items():
                rep_str += f"[{op}({w.labels[0]})]"
            rep_str += "\n"

        return rep_str

    @property
    def wires(self):
        """Track wires of the PauliSentence."""
        return set().union(*(pw.wires for pw in self.keys()))

    def to_mat(self, wire_order=None, format="dense"):
        """Given a Pauli word, get the matrix representation.

        KeywordArgs:
            wire_order (iterable or None): The order of qubits in the tensor product.
            format (str): The format of the matrix ("dense" by default), if not a dense
                matrix, then the format for the sparse representation of the matrix.

        Returns:
            (Union[NumpyArray, ScipySparseArray]): Matrix representation of the PauliSentence.
        """
        if wire_order is None:
            wire_order = self.wires

        mats_and_wires_gen = (
            (coeff * pw.to_mat(format=format), pw.wires) for pw, coeff in self.items()
        )

        reduced_mat, result_wire_order = math.reduce_matrices(
            mats_and_wires_gen=mats_and_wires_gen, reduce_func=math.add
        )

        return math.expand_matrix(reduced_mat, result_wire_order, wire_order=wire_order)

    def simplify(self, tol=1e-8):
        """Remove any PauliWords in the PauliSentence with coefficients less than the threshold tolerance."""
        for pw in self:
            if abs(self[pw]) <= tol:
                del self[pw]
