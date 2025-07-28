# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base class for Vibronic Hamiltonian"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pennylane.labs.trotter_error.realspace import (
    RealspaceCoeffs,
    RealspaceMatrix,
    RealspaceOperator,
    RealspaceSum,
)
from pennylane.labs.trotter_error.realspace.realspace_matrix import _next_pow_2


def vibronic_fragments(
    states: int, modes: int, freqs: np.ndarray, taylor_coeffs: Sequence[np.ndarray]
) -> list[RealspaceMatrix]:
    """Returns a list of fragments summing to a vibronic Hamiltonian.

    Args:
        states (int): the number of electronic states
        modes (int): the number of vibrational modes
        freqs (ndarray): the harmonic frequences
        taylor_coeffs (Sequence[ndarray]): a sequence containing the tensors of coefficients in the Taylor expansion

    Returns:
        List[RealspaceMatrix]: a list of ``RealspaceMatrix`` objects representing the fragments of the vibronic Hamiltonian

    **Example**

    >>> from pennylane.labs.trotter_error import vibronic_fragments
    >>> import numpy as np
    >>> n_modes = 4
    >>> n_states = 2
    >>> r_state = np.random.RandomState(42)
    >>> freqs = r_state.random(4)
    >>> taylor_coeffs = [r_state.random(size=(n_states, n_states, )), r_state.random(size=(n_states, n_states, n_modes))]
    >>> fragments = vibronic_fragments(n_states, n_modes, freqs, taylor_coeffs)
    >>> for fragment in fragments:
    >>>     fragment
    RealspaceMatrix({(0, 0): RealspaceSum((RealspaceOperator(4, (), 0.15601864044243652), RealspaceOperator(4, ('Q',), phi[1][0, 0][idx0]), RealspaceOperator(4, ('Q', 'Q'), omega[idx0,idx1]))), (1, 1): RealspaceSum((RealspaceOperator(4, (), 0.8661761457749352), RealspaceOperator(4, ('Q',), phi[1][1, 1][idx0]), RealspaceOperator(4, ('Q', 'Q'), omega[idx0,idx1])))})
    RealspaceMatrix({(0, 1): RealspaceSum((RealspaceOperator(4, (), 0.15599452033620265), RealspaceOperator(4, ('Q',), phi[1][0, 1][idx0]))), (1, 0): RealspaceSum((RealspaceOperator(4, (), 0.05808361216819946), RealspaceOperator(4, ('Q',), phi[1][1, 0][idx0])))})
    RealspaceMatrix({(0, 0): RealspaceSum((RealspaceOperator(4, ('P', 'P'), omega[idx0,idx1]),)), (1, 1): RealspaceSum((RealspaceOperator(4, ('P', 'P'), omega[idx0,idx1]),))})
    """
    _validate_input(states, modes, freqs, taylor_coeffs)

    frags = [
        _position_fragment(i, states, modes, freqs, taylor_coeffs)
        for i in range(_next_pow_2(states))
    ]
    frags.append(_momentum_fragment(states, modes, freqs))

    return frags


def _position_fragment(
    i: int, states: int, modes: int, freqs: np.ndarray, taylor_coeffs: Sequence[np.ndarray]
) -> RealspaceMatrix:
    """Return the ``i``th position fragment"""
    blocks = {
        (j, i ^ j): _realspace_sum(j, i ^ j, states, modes, freqs, taylor_coeffs)
        for j in range(_next_pow_2(states))
    }
    return RealspaceMatrix(states, modes, blocks)


def _momentum_fragment(states: int, modes: int, freqs: np.ndarray) -> RealspaceMatrix:
    """Return the fragment consisting only of momentum operators."""
    term = RealspaceOperator(
        modes,
        ("P", "P"),
        RealspaceCoeffs(np.diag(freqs) / 2, label="omega"),
    )
    word = RealspaceSum(modes, (term,))
    blocks = {(i, i): word for i in range(states)}

    return RealspaceMatrix(states, modes, blocks)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _realspace_sum(
    i: int, j: int, states: int, modes: int, freqs: np.ndarray, taylor_coeffs: Sequence[np.ndarray]
) -> RealspaceSum:
    """Return a RealspaceSum representation of the ``(i, j)`` block in the RealspaceMatrix"""
    if i > states - 1 or j > states - 1:
        return RealspaceSum.zero(modes)

    realspace_ops = []
    for k, phi in enumerate(taylor_coeffs):
        op = ("Q",) * k
        realspace_op = RealspaceOperator(
            modes,
            op,
            RealspaceCoeffs(phi[i, j], label=f"phi[{k}][{i}, {j}]"),
        )
        realspace_ops.append(realspace_op)

    if i == j:
        op = ("Q", "Q")
        coeffs = RealspaceCoeffs(np.diag(freqs) / 2, label="omega")
        assert coeffs is not None
        realspace_ops.append(RealspaceOperator(modes, op, coeffs))

    ret_val = RealspaceSum(modes, realspace_ops)
    for op in ret_val.ops:
        assert op.coeffs is not None

    return RealspaceSum(modes, realspace_ops)


def _validate_input(
    states: int, modes: int, freqs: np.ndarray, taylor_coeffs: Sequence[np.ndarray]
) -> None:
    """Validate that the shapes of the harmonic frequencies and the Taylor coefficients are correct."""
    for i, phi in enumerate(taylor_coeffs):
        shape = (states, states) + (modes,) * i

        if phi.shape != shape:
            raise ValueError(
                f"{i}th order coefficient tensor must have shape {shape}, got shape {phi.shape}"
            )

    if freqs.shape != (modes,):
        raise TypeError(f"Frequencies must have shape {(modes,)}, got shape {freqs.shape}.")
