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
This module contains example codes and gadgets that are small enough to check by hand.
"""

from __future__ import annotations

import numpy as np

from . import authoring as _auth
from .codes import CSSCode, DistanceClaim
from .ir import Action, Phase

# --------------------------------------------------------------------------------------
# Codes
# --------------------------------------------------------------------------------------


def rep_chain(
    d: int, offset: int = 0, n_frame: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Z checks and logical operators of one repetition-code chain inside a larger frame.

    The matrices are returned rather than a :class:`~.CSSCode`, because a chain padded into
    a wider frame is not a code on its own: the padding qubits are unconstrained.

    Args:
        d (int): length of the chain
        offset (int): index of the first qubit of the chain in the frame
        n_frame (int or None): Size of the frame. Defaults to ``offset + d``.

    Returns:
        tuple[array[int], array[int], array[int]]: the ``(d - 1, n_frame)`` Z checks, and
        the X and Z logical operators as single rows

    **Example**

    >>> from pennylane.ftqc.gadget.library import rep_chain
    >>> hz, lx, lz = rep_chain(3, offset=1, n_frame=5)
    >>> hz
    array([[0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0]], dtype=uint8)
    """
    n = offset + d if n_frame is None else n_frame
    hz = np.zeros((d - 1, n), dtype=np.uint8)
    for i in range(d - 1):
        hz[i, offset + i] = 1
        hz[i, offset + i + 1] = 1
    lx = np.zeros((1, n), dtype=np.uint8)
    lx[0, offset : offset + d] = 1
    lz = np.zeros((1, n), dtype=np.uint8)
    lz[0, offset] = 1
    return hz, lx, lz


def repetition_code(d: int) -> CSSCode:
    """The distance-``d`` repetition code, with Z checks between neighbouring qubits.

    Args:
        d (int): number of qubits

    Returns:
        ~.CSSCode: the code

    **Example**

    >>> from pennylane.ftqc.gadget.library import repetition_code
    >>> code = repetition_code(3)
    >>> code.n, code.k, code.distance.value
    (3, 1, 3)
    """
    hz, lx, lz = rep_chain(d)
    n = d
    return CSSCode(
        name=f"rep{d}",
        hx=np.zeros((0, n), dtype=np.uint8),
        hz=hz,
        lx=lx,
        lz=lz,
        distance=DistanceClaim(
            value=d,
            regime="static",
            certified=True,
            method="repetition code, distance is the chain length by construction",
        ),
    )


def steane_code() -> CSSCode:
    """The seven-qubit Steane code.

    The qubits are ordered as in Catalyst's QEC code library, so gadgets on this code can be
    compiled with the ``"Steane"`` code of Catalyst's QEC pipeline.

    Returns:
        ~.CSSCode: the ``[[7, 1, 3]]`` code, with the same checks for X and Z

    **Example**

    >>> from pennylane.ftqc.gadget.library import steane_code
    >>> code = steane_code()
    >>> code.n, code.k, code.distance.value
    (7, 1, 3)
    """
    h = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    ones = np.ones((1, 7), dtype=np.uint8)
    return CSSCode(
        name="Steane",
        hx=h,
        hz=h,
        lx=ones,
        lz=ones,
        distance=DistanceClaim(
            value=3,
            regime="static",
            certified=True,
            method="known code parameters for the [[7,1,3]] Steane code",
        ),
    )


# --------------------------------------------------------------------------------------
# Gadgets
# --------------------------------------------------------------------------------------


def steane_memory(rounds: int = 3):
    """A memory gadget that measures the Steane code's checks for a number of rounds.

    The gadget has one phase and no outcome, which is the form the current Catalyst
    pipeline can compile (see :func:`~.lowering.lower_gadget_to_qecl`).

    Args:
        rounds (int): number of measurement rounds

    Returns:
        tuple[~.CSSCode, tuple[~.Phase], ~.TracedGadget]: the code, its single phase, and
        the gadget

    **Example**

    >>> from pennylane.ftqc.gadget.library import steane_memory
    >>> code, (phase,), memory = steane_memory(rounds=3)
    >>> str(memory.program.action), memory.program.total_rounds
    ('idle', 3)
    """
    code = steane_code()
    base = Phase.from_code("steane", code)

    @_auth.define(
        action=Action.idle(),
        code=code,
        phases=(base,),
        claims=(
            DistanceClaim(
                value=3,
                regime="phenomenological",
                certified=False,
                method="asserted from the code distance; needs the simulation check",
            ),
        ),
    )
    def steane_memory(handle):
        """Hold a Steane codeblock, measuring its checks every round."""
        handle, _ = _auth.rounds(handle, rounds, record="mem")
        return handle

    return code, (base,), steane_memory


def rep_code_zz_merge(d: int = 3, merged_rounds: int | None = None, pre_rounds: int = 1):
    """A gadget that measures logical Z on both of two repetition codes by merging them.

    The frame has ``2 d`` qubits: block 0 on ``[0, d)`` and block 1 on ``[d, 2 d)``. The
    ``"base"`` phase measures each block's own checks. The ``"merged"`` phase adds the join
    check ``Z_{d-1} Z_d``; its outcome, combined with the base checks of block 0, is the
    logical ZZ of the pair.

    The join check is measured for the first time in the merged phase, so its first outcome
    is random and has no detector. Consequently the phenomenological fault distance of the
    outcome is ``min(d, merged_rounds)`` rather than ``d``.

    Args:
        d (int): distance of each block
        merged_rounds (int or None): Number of rounds in the merged phase. Defaults to
            ``d``.
        pre_rounds (int): number of rounds in the base phase before merging

    Returns:
        tuple[~.CSSCode, tuple[~.Phase, ~.Phase], ~.TracedGadget]: the two-block code, the
        ``(base, merged)`` phases, and the gadget

    **Example**

    >>> from pennylane.ftqc.gadget.library import rep_code_zz_merge
    >>> code, (base, merged), measure_zz = rep_code_zz_merge(d=3)
    >>> code.k, base.k, merged.k
    (2, 2, 1)
    >>> print(measure_zz.program.action)
    measure(Z_0_1)
    """
    merged_rounds = d if merged_rounds is None else merged_rounds
    n = 2 * d
    hz_a, lx_a, lz_a = rep_chain(d, offset=0, n_frame=n)
    hz_b, lx_b, lz_b = rep_chain(d, offset=d, n_frame=n)
    hz_base = np.vstack([hz_a, hz_b])
    lx = np.vstack([lx_a, lx_b])
    lz = np.vstack([lz_a, lz_b])
    pair = CSSCode(
        name=f"rep{d}x2",
        hx=np.zeros((0, n), dtype=np.uint8),
        hz=hz_base,
        lx=lx,
        lz=lz,
        distance=DistanceClaim(
            value=d,
            regime="static",
            certified=True,
            method="two independent repetition codes, distance is the chain length",
        ),
    )

    join = np.zeros((1, n), dtype=np.uint8)
    join[0, d - 1] = 1
    join[0, d] = 1
    base = Phase(
        name="base",
        hx=np.zeros((0, n), dtype=np.uint8),
        hz=hz_base,
        active=np.ones(n, dtype=bool),
    )
    merged = Phase(
        name="merged",
        hx=np.zeros((0, n), dtype=np.uint8),
        hz=np.vstack([hz_base, join]),
        active=np.ones(n, dtype=bool),
    )

    @_auth.define(
        action=Action.measure(("z", (0, 1))),
        code=pair,
        phases=(base, merged),
        claims=(
            DistanceClaim(
                value=min(d, merged_rounds),
                regime="phenomenological",
                certified=False,
                method="asserted: min(code distance, merged rounds); checked by simulation",
            ),
        ),
    )
    def measure_zz(handle):
        """Measure logical Z tensor Z by merging two repetition-code blocks."""
        handle, _ = _auth.rounds(handle, pre_rounds, record="pre")
        handle = _auth.deform(handle, to="merged")
        handle, checks = _auth.rounds(handle, merged_rounds, record="merged")
        outcome = _auth.observe(checks.product((hz_base.shape[0],)), index=0)
        handle = _auth.deform(handle, to="base")
        handle, _ = _auth.rounds(handle, 1, record="post")
        handle = _auth.frame(handle, outcome)
        return handle, outcome

    return pair, (base, merged), measure_zz


__all__ = [
    "rep_chain",
    "repetition_code",
    "steane_code",
    "steane_memory",
    "rep_code_zz_merge",
]
