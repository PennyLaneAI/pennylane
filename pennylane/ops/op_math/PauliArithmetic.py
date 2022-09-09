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
            else:
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
                rep_str += f"[{op}({w})]"
            rep_str += "\n"


# def make_mul_map():
#     map_X = {
#         X: (1, I),
#         Y: (1.0j, Z),
#         Z: (-1.0j, Y),
#     }
#     map_Y = {
#         X: (-1.0j, Z),
#         Y: (1, I),
#         Z: (1j, X),
#     }
#     map_Z = {
#         X: (1j, Y),
#         Y: (-1.0j, X),
#         Z: (1, I),
#     }
#
#     mul_map = {
#         X: map_X,
#         Y: map_Y,
#         Z: map_Z
#     }
#     return mul_map
#
#
# mul_map = make_mul_map()
#
# #### Conversion of observables and Hamiltonians ###########
#
#
# name_map = {'PauliX': X, 'PauliY': Y, 'PauliZ': Z, "Identity": I}
#
#
# def convert(obs):
#     if type(obs).__name__ != "Tensor":
#         return PauliWord({obs.wires[0]: name_map[obs.name]}), 1
#
#     d = {}
#     coeff = 1.0
#     for o in obs.obs:
#         w = o.wires[0]
#         if w in d:
#             factor, new_op = mul_map[d[w]][name_map[o.name]]
#             if new_op == I:
#                 del d[w]
#             else:
#                 coeff *= factor
#                 d[w] = new_op
#         elif o.name != "Identity":
#             d[w] = name_map[o.name]
#     return PauliWord(d), coeff
#
#
# def convert_H(H, tol=1e-4):
#     H_dict = defaultdict(float)
#
#     for coeff, ops in zip(H.coeffs, H.ops):
#         term, term_coeff = convert(ops)
#         H_dict[term] += coeff * term_coeff
#
#     delete_terms = tuple(term for term, coeff in H_dict.items() if abs(coeff) < tol)
#     for term in delete_terms:
#         del H_dict[term]
#
#     return H_dict
#
#
# #### Multiplication ############
#
#
# def mul_pw(pw1, pw2):
#     d = dict(pw1)
#     coeff = 1
#
#     for wire, term in pw2.items():
#         if wire in d:
#             factor, new_op = mul_map[d[wire]][term]
#             if new_op == I:
#                 del d[wire]
#             else:
#                 coeff *= factor
#                 d[wire] = new_op
#         else:
#             d[wire] = term
#
#     return PauliWord(d), coeff
#
#
# def mul_ps(ps1, ps2):
#     final_ps = PauliSentence({})
#     for pw1 in ps1:
#         for pw2 in ps2:
#             prod_pw, coeff = mul_pw(pw1, pw2)
#             final_ps[prod_pw] += coeff * ps1[pw1] * ps2[pw2]
#
#     return final_ps
#
#
# def add_ps(ps1, ps2):
#     smaller_ps, larger_ps = (ps1, ps2) if len(ps1) < len(ps2) else (ps2, ps1)
#     for key in smaller_ps:
#         larger_ps[key] += smaller_ps[key]
#
#     return larger_ps
#
# def add_multi_ps(*all_ps):
#     final_ps = all_ps[0]
#     for ps in all_ps[1:]:
#         for key in ps:
#             final_ps[key] += ps[key]
#
#     return final_ps



# def mul_H(h1, h2, tol=1e-4):
#     new_h = defaultdict(float)
#
#     for (term1, coeff1), (term2, coeff2) in product(h1.items(), h2.items()):
#         new_pw, coeff = mul_pw(term1, term2)
#         new_h[new_pw] += coeff1 * coeff2 * coeff
#
#     delete_terms = tuple(term for term, coeff in new_h.items() if abs(coeff) < tol)
#     for term in delete_terms:
#         del new_h[term]
#
#     return new_h