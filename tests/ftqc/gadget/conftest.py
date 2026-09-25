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
