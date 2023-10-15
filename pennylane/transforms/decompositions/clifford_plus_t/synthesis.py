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
"""Top-level tool to decompose a single RZ rotation to Clifford+T gates."""
# pylint:disable=missing-function-docstring

from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from typing import List, Tuple
import numpy as np

from .newsynth import gridsynth
from .conversion import denomexp
from .rings import DOmega, Matrix, ZRootTwo, ZOmega, Z2, omega_power


ZERO = ZOmega(0, 0, 0, 0)
ONE = ZOmega(0, 0, 0, 1)


def theta_to_gates(theta, epsilon=1e-4) -> str:
    """
    Convert an RZ rotation by theta to a sequence of Clifford+T gates.

    Args:
        theta (float): the RZ rotation angle to approximate
        epsilon (float): the error bound

    Returns:
        str: The list of operators that approximate the RZ rotation.
            Note that they are in the mathematical ordering, and should be
            applied from right-to-left
    """
    mat, _, _ = gridsynth(epsilon, theta)
    synth = synthesis_nqubit(mat)
    return to_gates(synth)


def synthesis_nqubit(m: Matrix) -> List["TwoLevel"]:
    def aux(n_by_m: Matrix, idx: int):
        c = np.array([n_by_m[:, 0]])
        gates = reduce_column(c, idx)
        if n_by_m.size == 2:
            return gates
        gates_matrix = matrix_of_twolevels(invert_twolevels(gates))
        m_ = gates_matrix @ n_by_m[:, 1].reshape((2, 1))
        return gates + aux(m_, idx + 1)

    return aux(m, 0)


def reduce_column(vectors, i):
    vlist = vectors[0]
    w, k = denomexp_decompose(vlist)

    def aux(w, k):
        if k == 0:
            j = [i for i, v in enumerate(w) if v != ZERO]
            if len(j) != 1:
                raise ValueError("not a unit vector:", w)
            j = j[0]
            wj = w[j]
            l = wj.log()
            if l is None:
                raise ValueError("not a unit vector:", wj)
            m1 = [] if i == j else [TL_X(i, j)]
            m2 = [TL_omega(l, i)]
            return m1 + m2
        res = residue(w)
        res1010 = [
            (i, a, x)
            for i, (a, x) in enumerate(zip(res, w))
            if residue_type(a) == ResidueType.RT_1010
        ]
        if len(res1010) % 2 != 0:
            raise ValueError("not a unit vector (should have even number of 1010 residues):", w)
        res0001 = [
            (i, a, x)
            for i, (a, x) in enumerate(zip(res, w))
            if residue_type(a) == ResidueType.RT_0001
        ]
        if len(res0001) % 2 != 0:
            raise ValueError("not a unit vector (should have even number of 0001 residues):", w)
        # split into pairs. eg. list(zip(*[iter([a, b, c, d])] * 2)) -> [(a, b), (c, d)]
        res1010_pairs = list(zip(*[iter(res1010)] * 2))
        res0001_pairs = list(zip(*[iter(res0001)] * 2))
        m1010 = [two_level for res_pair in res1010_pairs for two_level in row_step(res_pair)]
        m0001 = [two_level for res_pair in res0001_pairs for two_level in row_step(res_pair)]
        gates = m1010 + m0001
        return gates + aux(
            [reduce_ZOmega(z) for z in apply_twolevels_zomega(invert_twolevels(gates), w)], k - 1
        )

    return aux(w, k)


def denomexp_decompose(omegas: List[DOmega]):
    k = max(denomexp(d) for d in omegas)
    root_two_k = ZOmega.from_root_two(ZRootTwo(0, 1) ** k)
    b = [a * root_two_k for a in omegas]
    if not all(isinstance(a, ZOmega) for a in b):
        raise TypeError("All values should be converted to ZOmega during decomposition.")
    return b, k


def residue(x):
    if isinstance(x, list):
        return [residue(i) for i in x]
    if isinstance(x, ZOmega):
        return Z2(x.a, x.b, x.c, x.d)
    if isinstance(x, ZRootTwo):
        return ZRootTwo(x.a % 2, x.b % 2)
    if isinstance(x, int):
        return x % 2
    raise TypeError(f"residue not defined for {x} of type {type(x).__name__}")


def residue_type(z2: Z2) -> "ResidueType":
    return Z2_TO_TYPE_AND_SHIFT[z2][0]


def residue_shift(z2: Z2) -> int:
    return Z2_TO_TYPE_AND_SHIFT[z2][1]


def residue_offset(a: Z2, b: Z2) -> int:
    return (residue_shift(a) - residue_shift(b)) % 4


def opH_zomega(x: ZOmega, y: ZOmega):
    return reduce_ZOmega(x + y), reduce_ZOmega(x - y)


def reduce_ZOmega(z: ZOmega):
    a, b, c, d = z.a, z.b, z.c, z.d
    if (a - c) % 2 != 0 or (b - d) % 2 != 0:
        raise ValueError("ZOmega element not reducible:", z)
    return ZOmega((b - d) // 2, (c + a) // 2, (b + d) // 2, (c - a) // 2)


class TwoLevel(ABC):
    """Two-Level operator."""

    @abstractmethod
    def apply_zomega(self, omegas: List[ZOmega]) -> List[ZOmega]:
        """Apply this operator to a ZOmega-vector."""

    @abstractmethod
    def invert(self):
        """Invert this operator."""

    @abstractmethod
    def matrix(self):
        """Return the one- or two-level matrix for this operator."""

    @abstractmethod
    def gate(self) -> str:
        """Return the sequence of gate names this TwoLevel represents."""

    @staticmethod
    def transform_at2(op, i: int, j: int, omegas: List[ZOmega]) -> List[ZOmega]:
        res = omegas.copy()
        res[i], res[j] = op(omegas[i], omegas[j])
        return res

    @staticmethod
    def transform_at(i: int, k: int, omegas: List[ZOmega]) -> List[ZOmega]:
        res = omegas.copy()
        res[i] = omegas[i] * omega_power(k)
        return res

    @staticmethod
    def twolevel_matrix(xs, ys, i, j):
        a, b = xs
        c, d = ys

        def f(x, y):
            if x == i:
                if y == i:
                    return a
                if y == j:
                    return b
            if x == j:
                if y == i:
                    return c
                if y == j:
                    return d
            return ONE if x == y else ZERO

        return matrix_of_function(f)

    @staticmethod
    def onelevel_matrix(val, i):
        f = lambda x, y: val if x == y == i else ZOmega(0, 0, 0, int(x == y))
        return matrix_of_function(f)


class TL_X(TwoLevel):
    """The TwoLevel PauliX gate."""

    def __init__(self, i, j):
        self.i = i
        self.j = j

    def apply_zomega(self, omegas):
        return self.transform_at2(lambda a, b: (b, a), self.i, self.j, omegas)

    def invert(self):
        return self

    def matrix(self):
        x = (ZERO, ONE)
        y = (ONE, ZERO)
        return self.twolevel_matrix(x, y, self.i, self.j)

    def gate(self) -> str:
        if (self.i, self.j) in {(0, 1), (1, 0)}:
            return "X"
        raise ValueError("invalid gate")


class TL_H(TwoLevel):
    """The TwoLevel Hadamard gate."""

    def __init__(self, i, j):
        self.i = i
        self.j = j

    def apply_zomega(self, omegas):
        return self.transform_at2(opH_zomega, self.i, self.j, omegas)

    def invert(self):
        return self

    def matrix(self):
        roothalf = DOmega.from_root_two(1 / ZRootTwo(0, 1))
        return self.twolevel_matrix((roothalf, roothalf), (roothalf, -roothalf), self.i, self.j)

    def gate(self) -> str:
        ij = (self.i, self.j)
        if ij == (0, 1):
            return "H"
        if ij == (1, 0):
            return "XHX"
        raise ValueError("invalid gate")


class TL_T(TwoLevel):
    """The TwoLevel T gate."""

    def __init__(self, k, i, j):
        self.k = k
        self.i = i
        self.j = j

    def apply_zomega(self, omegas):
        return self.transform_at(self.j, self.k, omegas)

    def invert(self):
        return TL_T(-self.k, self.i, self.j)

    def matrix(self):
        x = (ZOmega(0, 0, 0, 1), ZERO)
        y = (ZERO, omega_power(self.k))
        return self.twolevel_matrix(x, y, self.i, self.j)

    def gate(self) -> str:
        ij = (self.i, self.j)
        if ij == (0, 1):
            if self.k % 2 == 1:
                return "T" + TL_T(self.k - 1, 0, 1).gate()
            if self.k % 4 == 2:
                return "S" + TL_T(self.k - 2, 0, 1).gate()
            if self.k % 8 == 4:
                return "Z"
            return ""
        if ij == (1, 0):
            return "X" + TL_T(self.k, 0, 1).gate() + "X"
        raise ValueError("invalid gate")


class TL_omega(TwoLevel):
    """The TwoLevel Omega (π/4) gate."""

    def __init__(self, k, i):
        self.k = k
        self.i = i

    def apply_zomega(self, omegas):
        return self.transform_at(self.i, self.k, omegas)

    def invert(self):
        return TL_omega(-self.k, self.i)

    def matrix(self):
        return self.onelevel_matrix(omega_power(self.k), self.i)

    def gate(self) -> str:
        if self.i == 0:
            return TL_T(self.k, 1, 0).gate()
        if self.i == 1:
            return TL_T(self.k, 0, 1).gate()
        raise ValueError("invalid gate")


def row_step(residue_pair: List[Tuple[int, Z2, ZOmega]]) -> List[TwoLevel]:
    (i, a, x), (j, b, y) = residue_pair
    if a.reducible() and b.reducible():
        return []

    offs = residue_offset(a=b, b=a)
    y_ = y * omega_power(-offs)
    b_ = residue(y_)
    if offs != 0:
        return [TL_T(offs, i, j)] + row_step(((i, a, x), (j, b_, y_)))

    x1, y1 = opH_zomega(x, y)
    a1, b1 = residue([x1, y1])
    return [TL_H(i, j)] + row_step(((i, a1, x1), (j, b1, y1)))


def apply_twolevels_zomega(tl_ops: List[TwoLevel], zomegas: List[ZOmega]):
    for tl_op in reversed(tl_ops):
        zomegas = tl_op.apply_zomega(zomegas)
    return zomegas


def invert_twolevels(tl_ops: List[TwoLevel]):
    return [tl_op.invert() for tl_op in reversed(tl_ops)]


def matrix_of_twolevels(tl_ops: List[TwoLevel]) -> Matrix:
    return reduce(np.matmul, [tl_op.matrix() for tl_op in tl_ops])


def matrix_of_function(func):
    a, b, c, d = [func(x, y) for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    return Matrix.array([[a, b], [c, d]])


def to_gates(tl_ops: List[TwoLevel]) -> str:
    return "".join(tl_op.gate() for tl_op in tl_ops)


class ResidueType(Enum):
    """The reside type of a ZOmega value. Used to compute shifts."""

    RT_0000 = 0
    RT_0001 = 1
    RT_1010 = 2


Z2_TO_TYPE_AND_SHIFT = {
    Z2(0, 0, 0, 0): (ResidueType.RT_0000, 0),
    Z2(0, 0, 0, 1): (ResidueType.RT_0001, 0),
    Z2(0, 0, 1, 0): (ResidueType.RT_0001, 1),
    Z2(0, 0, 1, 1): (ResidueType.RT_1010, 0),
    Z2(0, 1, 0, 0): (ResidueType.RT_0001, 2),
    Z2(0, 1, 0, 1): (ResidueType.RT_0000, 0),
    Z2(0, 1, 1, 0): (ResidueType.RT_1010, 1),
    Z2(0, 1, 1, 1): (ResidueType.RT_0001, 3),
    Z2(1, 0, 0, 0): (ResidueType.RT_0001, 3),
    Z2(1, 0, 0, 1): (ResidueType.RT_1010, 3),
    Z2(1, 0, 1, 0): (ResidueType.RT_0000, 0),
    Z2(1, 0, 1, 1): (ResidueType.RT_0001, 2),
    Z2(1, 1, 0, 0): (ResidueType.RT_1010, 2),
    Z2(1, 1, 0, 1): (ResidueType.RT_0001, 1),
    Z2(1, 1, 1, 0): (ResidueType.RT_0001, 0),
    Z2(1, 1, 1, 1): (ResidueType.RT_0000, 0),
}
