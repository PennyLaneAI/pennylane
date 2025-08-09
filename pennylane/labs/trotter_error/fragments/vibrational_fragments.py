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
"""The realspace vibrational Hamiltonian"""

from collections.abc import Sequence

import numpy as np

from pennylane.labs.trotter_error.realspace import RealspaceCoeffs, RealspaceOperator, RealspaceSum


def vibrational_fragments(
    modes: int, freqs: np.ndarray, taylor_coeffs: Sequence[np.ndarray], frag_method="harmonic"
) -> list[RealspaceSum]:
    """Returns a list of fragments summing to a vibrational Hamiltonian.

    Args:
        modes (int): the number of vibrational modes
        freqs (ndarray): the harmonic frequences
        taylor_coeffs (Sequence[ndarray]): a sequence containing the tensors of coefficients in the Taylor expansion
        frag_method (string): the fragmentation method, valid options are ``harmonic``, ``kinetic``, and ``position``

    Returns:
        List[RealspaceSum]: a list of ``RealspaceSum`` objects representing the fragments of the vibrational Hamiltonian

    **Example**

    >>> from pennylane.labs.trotter_error import vibrational_fragments
    >>> import numpy as np
    >>> n_modes = 4
    >>> r_state = np.random.RandomState(42)
    >>> freqs = r_state.random(4)
    >>> taylor_coeffs = [np.array(0), r_state.random(size=(n_modes, )), r_state.random(size=(n_modes, n_modes))]
    >>> fragments = vibrational_fragments(n_modes, freqs, taylor_coeffs)
    >>> for fragment in fragments:
    >>>     print(fragment)
    RealspaceSum((RealspaceOperator(4, ('PP',), omega[idx0]), RealspaceOperator(4, ('QQ',), omega[idx0])))
    RealspaceSum((RealspaceOperator(4, ('Q',), phi[1][idx0]), RealspaceOperator(4, ('Q', 'Q'), phi[2][idx0,idx1])))
    """

    if frag_method == "harmonic":
        return [_harmonic_fragment(modes, freqs), _anharmonic_fragment(modes, taylor_coeffs)]

    if frag_method == "kinetic":
        return [_kinetic_fragment(modes, freqs), _potential_fragment(modes, freqs, taylor_coeffs)]

    if frag_method == "position":
        return [_position_fragment(modes, freqs, taylor_coeffs), _momentum_fragment(modes, freqs)]

    raise ValueError(f"{frag_method} is not a valid fragmentation scheme.")


def _harmonic_fragment(modes: int, freqs: np.ndarray) -> RealspaceSum:
    """Returns the fragment of the Hamiltonian corresponding to the harmonic part."""
    _validate_freqs(modes, freqs)

    coeffs = RealspaceCoeffs(freqs / 2, label="omega")
    momentum = RealspaceOperator(modes, ("PP",), coeffs)
    position = RealspaceOperator(modes, ("QQ",), coeffs)

    return RealspaceSum(modes, [momentum, position])


def _anharmonic_fragment(modes: int, taylor_coeffs: Sequence[np.ndarray]) -> RealspaceSum:
    """Returns the fragment of the Hamiltonian corresponding to the anharmonic part."""
    _validate_taylor_coeffs(modes, taylor_coeffs)

    ops = []
    for i, phi in enumerate(taylor_coeffs):
        op = ("Q",) * i
        coeffs = RealspaceCoeffs(phi, label=f"phi[{i}]")
        realspace_op = RealspaceOperator(modes, op, coeffs)
        ops.append(realspace_op)

    return RealspaceSum(modes, ops)


def _kinetic_fragment(modes: int, freqs: np.ndarray) -> RealspaceSum:
    """Returns the fragment of the Hamiltonian corresponding to the kinetic part"""
    _validate_freqs(modes, freqs)

    coeffs = RealspaceCoeffs(freqs / 2, label="omega")
    kinetic = RealspaceOperator(modes, ("PP",), coeffs)

    return RealspaceSum(modes, [kinetic])


def _potential_fragment(
    modes: int, freqs: np.ndarray, taylor_coeffs: Sequence[np.ndarray]
) -> RealspaceSum:
    """Returns the fragment of the Hamiltonian corresponding to the potential part"""
    _validate_input(modes, freqs, taylor_coeffs)

    ops = []
    for i, phi in enumerate(taylor_coeffs):
        op = ("Q",) * i
        coeffs = RealspaceCoeffs(phi, label=f"phi[{i}]")
        realspace_op = RealspaceOperator(modes, op, coeffs)
        ops.append(realspace_op)

    op = ("Q", "Q")
    diag = np.diag(freqs / 2)
    coeffs = RealspaceCoeffs(diag, label="omega")
    realspace_op = RealspaceOperator(modes, op, coeffs)
    ops.append(realspace_op)

    return RealspaceSum(modes, ops)


def _position_fragment(
    modes: int, freqs: np.ndarray, taylor_coeffs: Sequence[np.ndarray]
) -> RealspaceSum:
    """Return the position term of the Hamiltonian"""
    coeffs = RealspaceCoeffs(freqs / 2, label="omega")
    position = RealspaceOperator(modes, ("QQ",), coeffs)

    return RealspaceSum(modes, [position]) + _anharmonic_fragment(modes, taylor_coeffs)


def _momentum_fragment(modes: int, freqs: np.ndarray) -> RealspaceSum:
    """Return the momentum term of the Hamiltonian"""
    _validate_freqs(modes, freqs)

    coeffs = RealspaceCoeffs(freqs / 2, label="omega")
    momentum = RealspaceOperator(modes, ("PP",), coeffs)

    return RealspaceSum(modes, [momentum])


def _validate_taylor_coeffs(modes: int, taylor_coeffs: Sequence[np.ndarray]) -> None:
    """Validate that the Taylor coefficients have the correct shape."""
    for i, phi in enumerate(taylor_coeffs):
        shape = (modes,) * i

        if not phi.shape == shape:
            raise ValueError(f"Expected order {i} to be of shape {shape}, got {phi.shape}.")


def _validate_freqs(modes: int, freqs: np.ndarray) -> None:
    """Validate that the harmonic frequencies have the correct shape."""
    if not len(freqs) == modes:
        raise ValueError(f"Expected freqs to be of length {modes}, got {len(freqs)}.")


def _validate_input(modes: int, freqs: np.ndarray, taylor_coeffs: Sequence[np.ndarray]) -> None:
    """Validate that the Taylor coefficients and the harmonic frequencies have the correct shape."""
    _validate_taylor_coeffs(modes, taylor_coeffs)
    _validate_freqs(modes, freqs)
