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
"""
Contains classes used to compactly store the metadata of various Hamiltonians which are relevant for resource estimation.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CDFHamiltonian:
    """For a compressed double-factorized (CDF) Hamiltonian, stores the minimum necessary information pertaining to resource estimation.

    Args:
        num_orbitals (int): number of spatial orbitals
        num_fragments (int): number of fragments in the compressed double-factorized (CDF) representation

    Returns:
        CDFHamiltonian: An instance of CDFHamiltonian

    .. seealso::
        :class:`~.estimator.templates.TrotterCDF`
    """

    num_orbitals: int
    num_fragments: int


@dataclass(frozen=True)
class THCHamiltonian:
    """For a tensor hypercontracted (THC) Hamiltonian, stores the minimum necessary information pertaining to resource estimation.

    Args:
        num_orbitals (int): number of spatial orbitals
        tensor_rank (int):  tensor rank of two-body integrals in the tensor hypercontracted (THC) representation

    Returns:
        THCHamiltonian: An instance of THCHamiltonian

    .. seealso::
        :class:`~.estimator.templates.TrotterTHC`
    """

    num_orbitals: int
    tensor_rank: int


@dataclass(frozen=True)
class VibrationalHamiltonian:
    """For a vibrational Hamiltonian, stores the minimum necessary information pertaining to resource estimation.

    Args:
        num_modes (int): number of vibrational modes
        grid_size (int): number of grid points used to discretize each mode
        taylor_degree (int): degree of the Taylor expansion used in the vibrational representation

    Returns:
        VibrationalHamiltonian: An instance of VibrationalHamiltonian

    .. seealso::
        :class:`~.estimator.templates.TrotterVibrational`
    """

    num_modes: int
    grid_size: int
    taylor_degree: int


@dataclass(frozen=True)
class VibronicHamiltonian:
    """For a vibronic Hamiltonian, stores the minimum necessary information pertaining to resource estimation.

    Args:
        num_modes (int): number of vibronic modes
        num_states (int): number of vibronic states
        grid_size (int): number of grid points used to discretize each mode
        taylor_degree (int): degree of the Taylor expansion used in the vibronic representation

    Returns:
        VibronicHamiltonian: An instance of VibronicHamiltonian

    .. seealso::
        :class:`~.estimator.templates.TrotterVibronic`
    """

    num_modes: int
    num_states: int
    grid_size: int
    taylor_degree: int


class PauliHamiltonian:
    """For a pauli Hamiltonian, stores the minimum necessary information pretaining to resource estimation.

    Args:
        num_qubits (int): total number of qubits the hamiltonian acts on
        num_pauli_words (int or None): the number of terms (Pauli words) in the hamiltonian
        max_factors (int or None): the maximum number of factors over all terms in the hamiltonian
        one_norm (float or None): the one-norm of the hamiltonian
        pauli_dist (dict or None): A dictionary representing the various Pauli words and how
            frequently they appear in the hamiltonian.
        commuting_groups (tuple(dict) or None): A tuple of dictionaries where each entry is a group
            of terms from the hamiltonian such that all terms in the group commute. Here each
            dictionary contains the various Pauli words and how frequently they appear in the group.

    Returns:
        PauliHamiltonian: An instance of PauliHamiltonian
    """

    def __init__(
        self,
        num_qubits: int,
        num_pauli_words: int | None = None,
        max_factors: int | None = None,
        one_norm: int | None = None,
        pauli_dist: dict | None = None,
        commuting_groups: tuple[dict] | None = None,
    ):
        self.num_qubits = num_qubits
        self.one_norm = one_norm
        self.max_factors = max_factors
        self.commuting_groups = commuting_groups

        if pauli_dist is not None:
            _validate_pauli_dist(pauli_dist)

        self.pauli_dist = pauli_dist

        if num_pauli_words is None:
            if self.pauli_dist is None and self.commuting_groups is None:
                raise ValueError(
                    "Must specifiy atleast one of `num_pauli_words` or `pauli_dist`. Got None for both."
                )
            if self.pauli_dist is not None:
                num_pauli_words = sum(pauli_dist.values())
            else:
                num_pauli_words = sum(sum(group.values()) for group in self.commuting_groups)
        else:
            if (self.pauli_dist is not None) and (
                sum_freqs := sum(pauli_dist.values())
            ) != num_pauli_words:
                raise ValueError(
                    f"The sum of the frequencies of `pauli_dist` ({sum_freqs}) should match `num_pauli_words`({num_pauli_words})"
                )

        self.num_pauli_words = num_pauli_words


def _validate_pauli_dist(pauli_dist: dict) -> bool:
    """Validate that the pauli_dist is formatted as expected"""
    for pauli_word, freq in pauli_dist.items():
        if (not isinstance(pauli_word, str)) or (
            not all(char in {"X", "Y", "Z"} for char in pauli_word)
        ):
            raise ValueError(
                f"The keys of `pauli_dist` represent Pauli words and should be strings containing either 'X','Y' or 'Z' characters only. Got {pauli_word} : {freq}"
            )

        if not isinstance(freq, int):
            raise ValueError(
                f"The values of `pauli_dist` represent frequencies and should be integers, got {pauli_word} : {freq}"
            )
