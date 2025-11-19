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
from collections import defaultdict
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
    """For a Pauli Hamiltonian, stores the minimum necessary information required for resource estimation.

    Args:
        num_qubits (int): total number of qubits the hamiltonian acts on
        num_pauli_words (int or None): the number of terms (Pauli words) in the hamiltonian
        max_weight (int or None): the maximum number of factors over all terms in the hamiltonian
        pauli_dist (dict or None): A dictionary representing the various Pauli words and how
            frequently they appear in the hamiltonian.
        commuting_groups (tuple(dict) or None): A tuple of dictionaries where each entry is a group
            of terms from the hamiltonian such that all terms in the group commute. Here each
            dictionary contains the various Pauli words and how frequently they appear in the group.

    Returns:
        PauliHamiltonian: An instance of PauliHamiltonian

    .. seealso::
        :class:`~.estimator.templates.TrotterPauli`

    **Example**

    A ``PauliHamiltonian`` is a compact representation which can be used with compatible templates
    to obtain resource estimates:

    >>> import pennylane.estimator as qre
    >>> num_steps, order = (10, 2)
    >>> pauli_ham = qre.PauliHamiltonian(
    ...     num_qubits = 10,
    ...     num_pauli_words = 100,
    ...     max_weight = 3,
    ... )
    >>> pauli_ham
    PauliHamiltonian(num_qubits=10, num_pauli_words=100, max_weight=3)
    >>> res = qre.estimate(qre.TrotterPauli(pauli_ham, num_steps, order))
    >>> print(res)
    --- Resources: ---
     Total wires: 10
       algorithmic wires: 10
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 1.099E+5
       'T': 8.800E+4,
       'CNOT': 8.000E+3,
       'Z': 1.980E+3,
       'S': 3.960E+3,
       'Hadamard': 7.920E+3

    .. details::
        :title: Usage Details

        There are three different ways to instantiate the ``PauliHamiltonian`` class depending on how
        much information is known about the hamiltonian we wish to capture (Note that providing more
        information will often lead to more accurate resource estimates).

        Firstly, when we know fairly little about the explicit form of the hamiltonian, we can express
        it by specifing the number of qubits it acts upon, the total number of terms in the hamiltonian
        and the maximum weight of a term overall terms in the hamiltonian.

        >>> import pennylane.estimator as qre
        >>> pauli_ham = qre.PauliHamiltonian(
        ...     num_qubits = 10,
        ...     num_pauli_words = 100,
        ...     max_weight = 3,
        ... )
        >>> pauli_ham
        PauliHamiltonian(num_qubits=10, num_pauli_words=100, max_weight=3)

        If we know approximately how the Pauli words are distributed in the hamiltonian, than we can
        construct the hamiltonian from this information. Note, if both the ``pauli_dist`` and the
        ``(num_pauli_words, max_weight)`` are provided, than ``pauli_dist`` will take precedent.
        This means that the ``(num_pauli_words, max_weight)`` will be computed from the ``pauli_dist``
        directly.

        >>> import pennylane.estimator as qre
        >>> pauli_ham = qre.PauliHamiltonian(
        ...     num_qubits = 10,
        ...     pauli_dist = {"X":10, "XX":30, "YY":10, "ZZ":45, "ZZZ": 5},
        ... )
        >>> pauli_ham
        PauliHamiltonian(num_qubits=10, num_pauli_words=100, max_weight=3)
        >>> pauli_ham.pauli_dist
        {'X': 10, 'XX': 30, 'YY': 10, 'ZZ': 45, 'ZZZ': 5}

        Finally, if we also know how to group the terms in the commuting groups of operators, we can
        construct the hamiltonian by specifying these groups of terms. This input will take precedent
        over both of the other two methods. Meaning that the attributes
        ``(num_pauli_words, max_weight, pauli_dist)`` will all be computed from the ``commuting_groups``
        directly.

        >>> import pennylane.estimator as qre
        >>> commuting_groups = (
        ...     {"X": 10, "XX": 30},
        ...     {"YY": 10, "ZZ": 5},
        ...     {"ZZ": 40, "ZZZ": 5},
        ... )
        >>> pauli_ham = qre.PauliHamiltonian(
        ...     num_qubits = 10,
        ...     commuting_groups = commuting_groups,
        ... )
        >>> pauli_ham
        PauliHamiltonian(num_qubits=10, num_pauli_words=100, max_weight=3)
        >>> pauli_ham.pauli_dist
        defaultdict(<class 'int'>, {'X': 10, 'XX': 30, 'YY': 10, 'ZZ': 45, 'ZZZ': 5})
        >>> pauli_ham.commuting_groups
        ({'X': 10, 'XX': 30}, {'YY': 10, 'ZZ': 5}, {'ZZ': 40, 'ZZZ': 5})

    """

    def __init__(
        self,
        num_qubits: int,
        num_pauli_words: int | None = None,
        max_weight: int | None = None,
        pauli_dist: dict | None = None,
        commuting_groups: tuple[dict] | None = None,
    ):
        self._num_qubits = num_qubits

        if commuting_groups is not None:
            for group in commuting_groups:
                _validate_pauli_dist(group)  #  ensure the groups are formatted correctly

            self._commuting_groups = commuting_groups
            self._pauli_dist = _pauli_dist_from_commuting_groups(commuting_groups)
            self._max_weight = max(len(pw) for pw in self.pauli_dist.keys())
            self._num_pauli_words = sum(self.pauli_dist.values())
            return

        if pauli_dist is not None:
            _validate_pauli_dist(pauli_dist)

            self._pauli_dist = pauli_dist
            self._commuting_groups = (pauli_dist,)
            self._max_weight = max(len(pw) for pw in pauli_dist.keys())
            self._num_pauli_words = sum(pauli_dist.values())
            return

        if (num_pauli_words is None) or (max_weight is None):
            raise ValueError(
                "One of the following sets of inputs must be provided (not None) in order to"
                f" instantiatea valid PauliHamiltonian:\n - `commuting_groups`\n - `pauli_dist`\n"
                " - `num_pauli_words` and `max_weight`."
            )

        self._max_weight = max_weight
        self._num_pauli_words = num_pauli_words
        self._pauli_dist = pauli_dist
        self._commuting_groups = commuting_groups

    def __repr__(self):
        """The repr dundar method for the PauliHamiltonian class."""
        return f"PauliHamiltonian(num_qubits={self.num_qubits}, num_pauli_words={self.num_pauli_words}, max_weight={self.max_weight})"

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def num_pauli_words(self):
        return self._num_pauli_words

    @property
    def max_weight(self):
        return self._max_weight

    @property
    def pauli_dist(self):
        return self._pauli_dist

    @property
    def commuting_groups(self):
        return self._commuting_groups


def _validate_pauli_dist(pauli_dist: dict) -> bool:
    """Validate that the pauli_dist is formatted as expected"""
    for pauli_word, freq in pauli_dist.items():
        if (not isinstance(pauli_word, str)) or (
            not all(char in {"X", "Y", "Z"} for char in pauli_word)
        ):
            raise ValueError(
                f"The keys represent Pauli words and should be strings containing either 'X','Y' or 'Z' characters only. Got {pauli_word} : {freq}"
            )

        if not isinstance(freq, int):
            raise ValueError(
                f"The values represent frequencies and should be integers, got {pauli_word} : {freq}"
            )


def _pauli_dist_from_commuting_groups(commuting_groups: tuple[dict]):
    """Construct the total Pauliword distribution from the commuting groups."""
    total_pauli_dist = defaultdict(int)

    for group in commuting_groups:
        for pauli_word, frequency in group.items():
            total_pauli_dist[pauli_word] += frequency

    return total_pauli_dist
