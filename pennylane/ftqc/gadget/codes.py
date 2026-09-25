# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the CSS code definitions used by gadgets, and distance claims.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import _gf2

Regime = Literal["static", "phenomenological", "circuit"]


class CodeError(ValueError):
    """Raised when a code definition is inconsistent."""


@dataclass(frozen=True)
class DistanceClaim:
    """A code or fault distance, together with the noise model it refers to and how it was
    obtained.

    A code distance says nothing on its own about the fault distance of a gadget built from
    that code, so every distance in this module carries its regime. Claims made by an author
    are uncertified; :func:`~pennylane.ftqc.gadget.verify` adds certified claims when it
    establishes a distance by simulation.

    Args:
        value (int): the distance
        regime (str): ``"static"`` for the distance of the code itself,
            ``"phenomenological"`` for data and measurement errors with ideal gates, or
            ``"circuit"`` for circuit-level faults
        certified (bool): whether ``method`` established the value for this instance
        method (str): how the value was obtained

    **Example**

    >>> from pennylane.ftqc import gadget
    >>> print(gadget.DistanceClaim(3, "phenomenological"))
    d=3 (phenomenological, UNCERTIFIED: asserted by author, not checked)
    """

    value: int
    regime: Regime = "static"
    certified: bool = False
    method: str = "asserted by author, not checked"

    def __str__(self) -> str:
        mark = "certified" if self.certified else "UNCERTIFIED"
        return f"d={self.value} ({self.regime}, {mark}: {self.method})"


@dataclass(frozen=True)
class CSSCode:
    """A CSS stabilizer code, checked for consistency when it is created.

    Logical qubit ``i`` of the code is the pair of rows ``lx[i]`` and ``lz[i]``; gadgets
    refer to logical qubits by this index.

    Args:
        name (str): name, used in diagnostics and in emitted IR
        hx (array[int]): X checks, shape ``(mx, n)``
        hz (array[int]): Z checks, shape ``(mz, n)``
        lx (array[int]): X logical operators, shape ``(k, n)``
        lz (array[int]): Z logical operators, shape ``(k, n)``
        distance (~.DistanceClaim or None): distance of the code, if known

    Raises:
        CodeError: If the matrices have inconsistent widths, the X and Z checks do not
            commute, the number of logical operators disagrees with the check ranks, a
            logical operator does not commute with the checks, or ``lx @ lz.T`` is not the
            identity.

    .. seealso:: :meth:`~.CSSCode.from_matrices`

    **Example**

    >>> from pennylane.ftqc import gadget
    >>> code = gadget.CSSCode.from_matrices(
    ...     "rep3", hx=[], hz=[[1, 1, 0], [0, 1, 1]], lx=[1, 1, 1], lz=[1, 0, 0]
    ... )
    >>> code.n, code.k
    (3, 1)
    >>> print(code)
    rep3 [[3,1]]

    Inconsistent definitions are rejected:

    >>> gadget.CSSCode.from_matrices("bad", hx=[], hz=[[1, 1, 0]], lx=[1, 1, 1], lz=[1, 0, 0])
    Traceback (most recent call last):
    ...
    pennylane.ftqc.gadget.codes.CodeError: bad: rank accounting gives k=2, but got 1 X logicals and 1 Z logicals
    """

    name: str
    hx: np.ndarray
    hz: np.ndarray
    lx: np.ndarray
    lz: np.ndarray
    distance: DistanceClaim | None = None

    def __post_init__(self) -> None:
        n = self.n
        for label, mat in (("hx", self.hx), ("hz", self.hz), ("lx", self.lx), ("lz", self.lz)):
            if mat.ndim != 2 or mat.shape[1] != n:
                raise CodeError(f"{self.name}: {label} has shape {mat.shape}, expected (*, {n})")

        violations = _gf2.commutes(self.hx, self.hz)
        if violations.any():
            bad = int(violations.sum())
            raise CodeError(
                f"{self.name}: CSS condition violated, Hx Hz^T != 0 mod 2 "
                f"({bad} anticommuting check pairs)"
            )

        k_from_rank = n - _gf2.rank(self.hx) - _gf2.rank(self.hz)
        if self.lx.shape[0] != k_from_rank or self.lz.shape[0] != k_from_rank:
            raise CodeError(
                f"{self.name}: rank accounting gives k={k_from_rank}, but got "
                f"{self.lx.shape[0]} X logicals and {self.lz.shape[0]} Z logicals"
            )

        if _gf2.commutes(self.hz, self.lx).any():
            raise CodeError(f"{self.name}: some X logical does not commute with a Z check")
        if _gf2.commutes(self.hx, self.lz).any():
            raise CodeError(f"{self.name}: some Z logical does not commute with an X check")

        if k_from_rank:
            pairing = _gf2.commutes(self.lx, self.lz)
            if not np.array_equal(pairing, np.eye(k_from_rank, dtype=np.uint8)):
                raise CodeError(
                    f"{self.name}: X and Z logicals are not symplectically paired "
                    "(Lx Lz^T must be the identity)"
                )

    @staticmethod
    def from_matrices(
        name: str,
        hx,
        hz,
        lx,
        lz,
        distance: DistanceClaim | None = None,
        n: int | None = None,
    ) -> CSSCode:
        """Create a code from array-like matrices.

        Entries are converted to ``uint8`` and one-dimensional inputs are treated as a
        single row. Empty matrices are allowed as long as the number of qubits can be
        inferred from another matrix or is given as ``n``.

        Args:
            name (str): name of the code
            hx (array_like): X checks
            hz (array_like): Z checks
            lx (array_like): X logical operators
            lz (array_like): Z logical operators
            distance (~.DistanceClaim or None): distance of the code, if known
            n (int or None): number of qubits, needed only if every matrix is empty

        Returns:
            ~.CSSCode: the code

        Raises:
            CodeError: if the number of qubits cannot be inferred, or the code is invalid
        """
        n_cols = n
        for candidate in (hx, hz, lx, lz):
            arr = np.asarray(candidate)
            if arr.size:
                n_cols = arr.reshape(1, -1).shape[1] if arr.ndim == 1 else arr.shape[1]
                break
        if n_cols is None:
            raise CodeError(f"{name}: cannot infer the number of qubits")
        return CSSCode(
            name=name,
            hx=_gf2.as_bits(hx, n_cols),
            hz=_gf2.as_bits(hz, n_cols),
            lx=_gf2.as_bits(lx, n_cols),
            lz=_gf2.as_bits(lz, n_cols),
            distance=distance,
        )

    @property
    def n(self) -> int:
        """Number of physical qubits."""
        return int(self.hx.shape[1]) if self.hx.size else int(self.hz.shape[1])

    @property
    def k(self) -> int:
        """Number of logical qubits."""
        return int(self.lx.shape[0])

    @property
    def max_check_weight(self) -> int:
        """Largest check weight, over both X and Z checks."""
        wx = _gf2.row_weights(self.hx)
        wz = _gf2.row_weights(self.hz)
        return int(max(wx.max(initial=0), wz.max(initial=0)))

    @property
    def max_qubit_degree(self) -> int:
        """Largest number of checks acting on any single qubit."""
        return int((_gf2.col_weights(self.hx) + _gf2.col_weights(self.hz)).max(initial=0))

    def fingerprint(self) -> str:
        """A content hash of the check and logical matrices.

        Returns:
            str: 16 hexadecimal characters
        """
        h = hashlib.sha256()
        for mat in (self.hx, self.hz, self.lx, self.lz):
            h.update(np.ascontiguousarray(mat).tobytes())
            h.update(str(mat.shape).encode())
        return h.hexdigest()[:16]

    def __str__(self) -> str:
        dist = f", {self.distance}" if self.distance else ""
        return f"{self.name} [[{self.n},{self.k}]]{dist}"
