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


class PauliWord(dict):
    """Immutable dictionary used to represent a Pauli Word.
    Can be constructed from a standard dictionary.

    >>> w = PauliWord({"a": X, 2: Y, 3: z})
    """

    _map_X = {
        X: (1, I),
        Y: (1.0j, Z),
        Z: (-1.0j, Y),
    }
    _map_Y = {
        X: (-1.0j, Z),
        Y: (1, I),
        Z: (1j, X),
    }
    _map_Z = {
        X: (1j, Y),
        Y: (-1.0j, X),
        Z: (1, I),
    }

    mul_map = {
        X: _map_X,
        Y: _map_Y,
        Z: _map_Z
    }

    def __missing__(self, key):
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
                factor, new_op = self.mul_map[d[wire]][term]
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
        return 0.0

    def __add__(self, other):
        smaller_ps, larger_ps = (self, other) if len(self) < len(other) else (other, self)
        for key in smaller_ps:
            larger_ps[key] += smaller_ps[key]

        return larger_ps

    def __mul__(self, other):
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

    def _to_mat(self, wire_order, sparse=False):
        matrix_map = sparse_mat_map if sparse else mat_map
        final_mat = sparse.eye(2, format="csr") if sparse else np.eye(2)

        for i, (pw, coeff) in enumerate(self.items()):
            mat = sparse.eye(2, format="csr") if sparse else np.eye(2)
            for j, wire in enumerate(wire_order):
                mat = math.dot(mat, matrix_map[pw[wire]]) if j == 0 else math.kron(mat, matrix_map[pw[wire]])

            mat = coeff * mat
            final_mat = mat if i == 0 else final_mat + mat

        return final_mat

    def to_mat(self, wire_order):
        return self._to_mat(wire_order=wire_order)

    def to_sparse_mat(self, wire_order):
        return self._to_mat(wire_order=wire_order, sparse=True)
