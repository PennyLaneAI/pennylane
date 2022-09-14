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
from enum import Enum
from functools import reduce

# import pennylane as qml
# import pennylane.ops
from pennylane import math
import numpy as np
from scipy import sparse


class Pauli(Enum):
    I = 0
    X = 1
    Y = 2
    Z = 3

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return self.name


I = Pauli.I
X = Pauli.X
Y = Pauli.Y
Z = Pauli.Z

matX = np.array([[0, 1], [1, 0]])
matY = np.array([[0, -1j], [1j, 0]])
matZ = np.array([[1, 0], [0, -1]])
matI = np.eye(2)

mat_map = {
    I: matI,
    X: matX,
    Y: matY,
    Z: matZ,
}

sparse_matX = sparse.csr_matrix([[0, 1], [1, 0]])
sparse_matY = sparse.csr_matrix([[0, -1j], [1j, 0]])
sparse_matZ = sparse.csr_matrix([[1, 0], [0, -1]])
sparse_matI = sparse.eye(2, format="csr")

sparse_mat_map = {
    I: sparse_matI,
    X: sparse_matX,
    Y: sparse_matY,
    Z: sparse_matZ,
}

# op_map = {
#     I: pennylane.ops.Identity,
#     X: qml.PauliX,
#     Y: qml.PauliY,
#     Z: qml.PauliZ,
# }

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

mul_map = {
    I: _map_I,
    X: _map_X,
    Y: _map_Y,
    Z: _map_Z
}


class PauliWord(dict):
    """Immutable dictionary used to represent a Pauli Word.
    Can be constructed from a standard dictionary.

    >>> w = PauliWord({"a": X, 2: Y, 3: z})
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
        d = dict(self)
        coeff = 1

        for wire, term in other.items():

            if wire in d:
                factor, new_op = mul_map[d[wire]][term]
                if new_op == I:
                    del d[wire]
                else:
                    coeff *= factor
                    d[wire] = new_op
            elif term != I:
                d[wire] = term

        return PauliWord(d), coeff


class PauliSentence(dict):
    """Dict representing a Pauli Sentence."""
    def __missing__(self, key):
        """If the pauliword is not in the sentence then the coefficient
        associated with it should be 0."""
        return 0.0

    def __add__(self, other):
        """Add two Pauli sentence together by iterating over the smaller
        one and adding its terms to the larger one."""
        smaller_ps, larger_ps = (self, other) if len(self) < len(other) else (other, self)
        for key in smaller_ps:
            larger_ps[key] += smaller_ps[key]

        return larger_ps

    def __mul__(self, other):
        """Multiply two Pauli sentences by iterating over each sentence and multiplying
        the Pauli words pair-wise"""
        final_ps = PauliSentence({})
        for pw1 in self:
            for pw2 in other:
                prod_pw, coeff = pw1 * pw2
                final_ps[prod_pw] += coeff * self[pw1] * other[pw2]

        return final_ps

    def __str__(self):
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

    def _to_mat(self, wire_order, is_sparse=False):
        """Get the matrix by iterating over each term and getting its matrix
        representation for each wire listed in the wire order."""
        matrix_map = sparse_mat_map if is_sparse else mat_map
        final_mat = sparse.eye(2, format="csr") if is_sparse else np.eye(2)

        for i, (pw, coeff) in enumerate(self.items()):
            mat = sparse.eye(2, format="csr") if is_sparse else np.eye(2)
            for j, wire in enumerate(wire_order):
                mat = math.dot(mat, matrix_map[pw[wire]]) if j == 0 else math.kron(mat, matrix_map[pw[wire]])

            mat = coeff * mat
            final_mat = mat if i == 0 else final_mat + mat

        return final_mat

    def to_mat(self, wire_order):
        return self._to_mat(wire_order=wire_order)

    def to_sparse_mat(self, wire_order):
        return self._to_mat(wire_order=wire_order, is_sparse=True)

    def simplify(self, tol=1e-8):
        for pw in self:
            if abs(self[pw]) <= tol:
                del self[pw]

    # def to_op(self):
    #     """Generate a native PL operator from the reduced representation"""
    #     pl_op = qml.op_sum(
    #         *(qml.s_prod(coeff, qml.prod(
    #             *(op_map[op](w) for w, op in pw.items())
    #         )) for pw, coeff in self.items())
    #     )
    #     return pl_op
    #
    # def to_hamiltonian(self):
    #     """Generate a native PL hamiltonian from the reduced representation"""
    #     coeffs, terms = ([], [])
    #
    #     for pw, coeff in self.items():
    #         coeffs.append(coeff)
    #         terms.append(qml.Tensor(*(op_map[op](w) for w, op in pw.items)))
    #     return qml.Hamiltonian(coeffs, terms)
