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

"""Shared gadgets for the ``pennylane.ftqc.gadget`` tests."""

import numpy as np
import pytest

from pennylane.ftqc import gadget
from pennylane.ftqc.gadget.library import rep_chain, rep_code_zz_merge, steane_memory


@pytest.fixture
def rep_zz():
    """``(code, (base, merged), measure_zz)`` for two distance-3 repetition codes."""
    return rep_code_zz_merge(d=3)


@pytest.fixture
def steane_mem():
    """``(code, (steane,), steane_memory)`` holding a Steane block for three rounds."""
    return steane_memory(rounds=3)


def _aux_pair():
    n = 7
    hz_a, _, _ = rep_chain(3, offset=0, n_frame=n)
    hz_b, _, _ = rep_chain(3, offset=3, n_frame=n)
    hz_base = np.vstack([hz_a, hz_b])
    no_x = np.zeros((0, n), dtype=np.uint8)

    active = np.ones(n, dtype=bool)
    active[6] = False
    base = gadget.Phase(name="base", hx=no_x, hz=hz_base, active=active)

    via = np.zeros((2, n), dtype=np.uint8)
    via[0, [2, 6]] = 1
    via[1, [3, 6]] = 1
    merged = gadget.Phase(
        name="merged", hx=no_x, hz=np.vstack([hz_base, via]), active=np.ones(n, dtype=bool)
    )

    lx = np.zeros((2, 6), dtype=np.uint8)
    lx[0, 0:3] = 1
    lx[1, 3:6] = 1
    lz = np.zeros((2, 6), dtype=np.uint8)
    lz[0, 0] = 1
    lz[1, 3] = 1
    code = gadget.CSSCode(
        name="rep3x2", hx=np.zeros((0, 6), dtype=np.uint8), hz=hz_base[:, :6], lx=lx, lz=lz
    )
    return code, base, merged


@pytest.fixture
def aux_pair():
    """Two distance-3 repetition chains on qubits 0-5, joined through auxiliary qubit 6.

    Returns ``(code, base, merged)``. Qubit 6 is inactive in ``base``; ``merged``
    activates it and adds the checks ``Z2 Z6`` and ``Z6 Z3``, whose product is the join
    ``Z2 Z3``.
    """
    return _aux_pair()


@pytest.fixture
def aux_merge():
    """A ZZ merge through an auxiliary qubit that is initialized, merged, then detached."""
    code, base, merged = _aux_pair()

    @gadget.define(
        action=gadget.Action.measure(("z", (0, 1))),
        code=code,
        phases=(base, merged),
        n_data=6,
        claims=(gadget.DistanceClaim(value=3, regime="phenomenological"),),
    )
    def via_aux(handle):
        """Merge through an aux qubit, then measure it out."""
        handle, _ = gadget.rounds(handle, 1, record="pre")
        handle = gadget.deform(handle, to="merged", init={6: "z"})
        handle, checks = gadget.rounds(handle, 3, record="merged")
        outcome = gadget.observe(checks.product((4, 5)), index=0)
        handle, _ = gadget.detach(handle, to="base", measure_out={6: "z"}, record="out")
        handle, _ = gadget.rounds(handle, 1, record="post")
        return handle, outcome

    return via_aux


def _surface_checks(rows, cols, index, n):
    """Checks of a rotated surface code with X-type boundaries on the left and right, over a
    frame of ``n`` qubits."""
    hx, hz = [], []
    for r in range(-1, rows):
        for c in range(-1, cols):
            cells = [
                (a, b) for a in (r, r + 1) for b in (c, c + 1) if 0 <= a < rows and 0 <= b < cols
            ]
            kind = "x" if (r + c) % 2 == 0 else "z"
            if len(cells) == 2:
                side = c in (-1, cols - 1)
                if (side and kind != "x") or (not side and kind != "z"):
                    continue
            elif len(cells) != 4:
                continue
            row = np.zeros(n, dtype=np.uint8)
            row[[index(a, b) for a, b in cells]] = 1
            (hx if kind == "x" else hz).append(row)
    return np.array(hx), np.array(hz)


def _surface_merge(prep):
    """A Z⊗Z lattice-surgery merge of two distance-3 rotated surface codes through a column of
    three auxiliary qubits, which are prepared in ``prep`` and measured out in X."""
    d, n_data = 3, 18
    n = n_data + d

    def index(r, c):
        if c < d:
            return r * d + c
        if c > d:
            return d * d + r * d + (c - d - 1)
        return n_data + r

    ax, az = _surface_checks(d, d, index, n)
    bx, bz = _surface_checks(d, d, lambda r, c: index(r, c + d + 1), n)
    mx, mz = _surface_checks(d, 2 * d + 1, index, n)
    base_hx, base_hz = np.vstack([ax, bx]), np.vstack([az, bz])
    lx = np.zeros((2, n_data), np.uint8)
    lz = np.zeros((2, n_data), np.uint8)
    lx[0, [index(0, c) for c in range(d)]] = 1
    lx[1, [index(0, c + d + 1) for c in range(d)]] = 1
    lz[0, [index(r, 0) for r in range(d)]] = 1
    lz[1, [index(r, d + 1) for r in range(d)]] = 1
    code = gadget.CSSCode("surface3x2", base_hx[:, :n_data], base_hz[:, :n_data], lx, lz)
    base = gadget.Phase("base", base_hx, base_hz, np.array([True] * n_data + [False] * d))
    merged = gadget.Phase("merged", mx, mz, np.ones(n, dtype=bool))
    aux = list(range(n_data, n))
    seam = tuple(len(mx) + i for i, row in enumerate(mz) if row[aux].any())

    @gadget.define(
        action=gadget.Action.measure(("z", (0, 1))),
        code=code,
        phases=(base, merged),
        n_data=n_data,
        claims=(gadget.DistanceClaim(3, "phenomenological"),),
    )
    def surface_zz(handle):
        handle, _ = gadget.rounds(handle, 1, record="pre")
        handle = gadget.deform(handle, to="merged", init={q: prep for q in aux})
        handle, checks = gadget.rounds(handle, d, record="merged")
        outcome = gadget.observe(checks.product(seam), index=0)
        handle, _ = gadget.detach(
            handle, to="base", measure_out={q: "x" for q in aux}, record="split"
        )
        handle, _ = gadget.rounds(handle, 1, record="post")
        return handle, outcome

    return surface_zz


@pytest.fixture
def surface_merge():
    """The surface-code merge with the auxiliary column prepared in X, as lattice surgery
    requires."""
    return _surface_merge("x")


@pytest.fixture
def surface_merge_factory():
    """A function building the surface-code merge with a given auxiliary preparation basis."""
    return _surface_merge
