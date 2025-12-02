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
import copy
from collections import Counter
from dataclasses import dataclass


def _validate_positive_int(name, value):
    """Helper to validate positive integers."""
    if not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer, got {value}")


@dataclass(frozen=True)
class CDFHamiltonian:
    """For a compressed double-factorized (CDF) Hamiltonian, stores the minimum necessary information pertaining to resource estimation.

    Args:
        num_orbitals (int): number of spatial orbitals
        num_fragments (int): number of fragments in the compressed double-factorized (CDF) representation
        one_norm (float | None): the one-norm of the Hamiltonian

    Returns:
        CDFHamiltonian: An instance of CDFHamiltonian

    .. seealso::
        :class:`~.estimator.templates.TrotterCDF`
    """

    num_orbitals: int
    num_fragments: int
    one_norm: float | None = None

    def __post_init__(self):
        """Checks the types of the inputs."""

        _validate_positive_int("num_orbitals", self.num_orbitals)
        _validate_positive_int("num_fragments", self.num_fragments)
        if self.one_norm is not None and not (
            isinstance(self.one_norm, (float, int)) and self.one_norm >= 0
        ):
            raise TypeError(
                f"one_norm, if provided, must be a positive float or integer. Instead received {self.one_norm}"
            )
        if isinstance(self.one_norm, int):
            object.__setattr__(self, "one_norm", float(self.one_norm))


@dataclass(frozen=True)
class THCHamiltonian:
    """For a tensor hypercontracted (THC) Hamiltonian, stores the minimum necessary information pertaining to resource estimation.

    Args:
        num_orbitals (int): number of spatial orbitals
        tensor_rank (int):  tensor rank of two-body integrals in the tensor hypercontracted (THC) representation
        one_norm (float | None): the one-norm of the Hamiltonian

    Returns:
        THCHamiltonian: An instance of THCHamiltonian

    .. seealso::
        :class:`~.estimator.templates.TrotterTHC`
    """

    num_orbitals: int
    tensor_rank: int
    one_norm: float | None = None

    def __post_init__(self):
        """Checks the types of the inputs."""

        _validate_positive_int("num_orbitals", self.num_orbitals)
        _validate_positive_int("tensor_rank", self.tensor_rank)
        if self.one_norm is not None and not (
            isinstance(self.one_norm, (float, int)) and self.one_norm >= 0
        ):
            raise TypeError(
                f"one_norm, if provided, must be a positive float or integer."
                f" Instead received {self.one_norm}"
            )

        if isinstance(self.one_norm, int):
            object.__setattr__(self, "one_norm", float(self.one_norm))


@dataclass(frozen=True)
class VibrationalHamiltonian:
    """For a vibrational Hamiltonian, stores the minimum necessary information pertaining to resource estimation.

    Args:
        num_modes (int): number of vibrational modes
        grid_size (int): number of grid points used to discretize each mode
        taylor_degree (int): degree of the Taylor expansion used in the vibrational representation
        one_norm (float | None): the one-norm of the Hamiltonian

    Returns:
        VibrationalHamiltonian: An instance of VibrationalHamiltonian

    .. seealso::
        :class:`~.estimator.templates.TrotterVibrational`
    """

    num_modes: int
    grid_size: int
    taylor_degree: int
    one_norm: float | None = None

    def __post_init__(self):
        """Checks the types of the inputs."""

        _validate_positive_int("num_modes", self.num_modes)
        _validate_positive_int("grid_size", self.grid_size)
        _validate_positive_int("taylor_degree", self.taylor_degree)

        if self.one_norm is not None and not (
            isinstance(self.one_norm, (float, int)) and self.one_norm >= 0
        ):
            raise TypeError(
                f"one_norm, if provided, must be a positive float or integer."
                f" Instead received {self.one_norm}"
            )

        if isinstance(self.one_norm, int):
            object.__setattr__(self, "one_norm", float(self.one_norm))


@dataclass(frozen=True)
class VibronicHamiltonian:
    """For a vibronic Hamiltonian, stores the minimum necessary information pertaining to resource estimation.

    Args:
        num_modes (int): number of vibronic modes
        num_states (int): number of vibronic states
        grid_size (int): number of grid points used to discretize each mode
        taylor_degree (int): degree of the Taylor expansion used in the vibronic representation
        one_norm (float | None): the one-norm of the Hamiltonian

    Returns:
        VibronicHamiltonian: An instance of VibronicHamiltonian

    .. seealso::
        :class:`~.estimator.templates.TrotterVibronic`
    """

    num_modes: int
    num_states: int
    grid_size: int
    taylor_degree: int
    one_norm: float | None = None

    def __post_init__(self):
        """Checks the types of the inputs."""

        _validate_positive_int("num_modes", self.num_modes)
        _validate_positive_int("num_states", self.num_states)
        _validate_positive_int("grid_size", self.grid_size)
        _validate_positive_int("taylor_degree", self.taylor_degree)

        if self.one_norm is not None and not (
            isinstance(self.one_norm, (float, int)) and self.one_norm >= 0
        ):
            raise TypeError(
                f"one_norm, if provided, must be a positive float or integer."
                f" Instead received {self.one_norm}"
            )

        if isinstance(self.one_norm, int):
            object.__setattr__(self, "one_norm", float(self.one_norm))


class PauliHamiltonian:
    r"""Stores the minimum necessary information required for resource estimation of a
    Hamiltonian expressed as a linear combination of tensor products of Pauli operators.

    Args:
        num_qubits (int): total number of qubits the Hamiltonian acts on
        num_pauli_words (int | None): the number of terms (Pauli words) in the Hamiltonian
        max_weight (int | None): The maximum number of qubits a term acts upon, over all
            terms in the linear combination.
        one_norm (float | int | None): the one-norm of the Hamiltonian
        pauli_dist (dict | None): A dictionary representing the various Pauli words and how
            frequently they appear in the Hamiltonian.
        commuting_groups (tuple(dict) | None): A tuple of dictionaries where each entry represents
            a group of Pauli words that mutually commute. Each entry is formatted similarly to the
            ``pauli_dist`` argument (see the Usage Details section for more information).
            of terms from the Hamiltonian such that all terms in the group commute. Here each
            dictionary contains the various Pauli words and how frequently they appear in the group.

    Returns:
        PauliHamiltonian: An instance of PauliHamiltonian

    .. seealso::
        :class:`~.estimator.templates.TrotterPauli`, :class:`~.estimator.templates.SelectPauli`

    **Example**

    A ``PauliHamiltonian`` is a compact representation which can be used with compatible templates
    to obtain resource estimates. Consider for example the Hamiltonian:

    .. math::

        \hat{H} = 0.1 \cdot \Sigma^{30}_{j=1} \hat{X}_{j} \hat{X}_{j+1}
        - 0.05 \cdot \Sigma^{30}_{k=1} \hat{Y}_{k} \hat{Y}_{k+1} + 0.25 \cdot \Sigma^{40}_{l=1} \hat{X}_{l}


    >>> import pennylane.estimator as qre
    >>> pauli_ham = qre.PauliHamiltonian(
    ...     num_qubits = 40,
    ...     num_pauli_words = 100,
    ...     max_weight = 2,
    ...     one_norm = 14.5,  # (0.1 * 30) + (0.05 * 30) + (0.25 * 40)
    ... )
    >>> pauli_ham
    PauliHamiltonian(num_qubits=40, num_pauli_words=100, max_weight=2, one_norm=14.5)

    The Hamiltonian can be used as input for other subroutines, like
    :class:`~.estimator.templates.trotter.TrotterPauli`:

    >>> num_steps, order = (10, 2)
    >>> res = qre.estimate(qre.TrotterPauli(pauli_ham, num_steps, order))
    >>> print(res)
    --- Resources: ---
     Total wires: 40
       algorithmic wires: 40
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 1.012E+5
       'T': 8.800E+4,
       'CNOT': 4.000E+3,
       'Z': 1.320E+3,
       'S': 2.640E+3,
       'Hadamard': 5.280E+3

    .. details::
        :title: Usage Details

        There are three different ways to instantiate the ``PauliHamiltonian`` class depending on how
        much information is known about the Hamiltonian we wish to capture. Note that providing more
        information will generally lead to more accurate resource estimates.

        Firstly, when we know fairly little about the explicit form of the Hamiltonian, we can express
        it by specifying the number of qubits it acts upon, the total number of terms in the Hamiltonian
        and the maximum weight of a term in the Hamiltonian.

        >>> import pennylane.estimator as qre
        >>> pauli_ham = qre.PauliHamiltonian(
        ...     num_qubits = 40,
        ...     num_pauli_words = 100,
        ...     max_weight = 2,
        ...     one_norm = 14.5,  # (0.1 * 30) + (0.05 * 30) + (0.25 * 40)
        ... )
        >>> pauli_ham
        PauliHamiltonian(num_qubits=40, num_pauli_words=100, max_weight=2, one_norm=14.5)

        If we know approximately how the Pauli words are distributed in the Hamiltonian, then we can
        construct the Hamiltonian from this information. Note, if both the ``pauli_dist`` and the
        ``(num_pauli_words, max_weight)`` are provided, then ``pauli_dist`` will take precedence.
        This means that the ``(num_pauli_words, max_weight)`` will be computed from the ``pauli_dist``
        directly.

        >>> import pennylane.estimator as qre
        >>> pauli_ham = qre.PauliHamiltonian(
        ...     num_qubits = 40,
        ...     pauli_dist = {"X":40, "XX":30, "YY":30},
        ...     one_norm = 14.5,  # (0.1 * 30) + (0.05 * 30) + (0.25 * 40)
        ... )
        >>> pauli_ham
        PauliHamiltonian(num_qubits=40, num_pauli_words=100, max_weight=2, one_norm=14.5)
        >>> pauli_ham.pauli_dist
        {'X': 40, 'XX': 30, 'YY': 30}

        Finally, if we also know how to separate the terms into commuting groups of operators, we can
        construct the Hamiltonian by specifying these groups of terms. This input will take precedence
        over the other two methods. Meaning that the attributes
        ``(num_pauli_words, max_weight, pauli_dist)`` will all be computed from the ``commuting_groups``
        directly.

        >>> import pennylane.estimator as qre
        >>> commuting_groups = (
        ...     {"X": 40, "XX": 30},
        ...     {"YY": 30},
        ... )
        >>> pauli_ham = qre.PauliHamiltonian(
        ...     num_qubits = 40,
        ...     commuting_groups = commuting_groups,
        ...     one_norm = 14.5,
        ... )
        >>> pauli_ham
        PauliHamiltonian(num_qubits=40, num_pauli_words=100, max_weight=2, one_norm=14.5)
        >>> pauli_ham.pauli_dist
        {'X': 40, 'XX': 30, 'YY': 30}
        >>> pauli_ham.commuting_groups
        ({'X': 40, 'XX': 30}, {'YY': 30})

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_qubits: int,
        num_pauli_words: int | None = None,
        max_weight: int | None = None,
        one_norm: float | None = None,
        pauli_dist: dict | None = None,
        commuting_groups: tuple[dict] | None = None,
    ):
        self._num_qubits = num_qubits

        if one_norm is not None and not (isinstance(one_norm, (float, int)) and one_norm >= 0):
            raise ValueError(
                f"one_norm, if provided, must be a positive float or integer. Instead received {one_norm}"
            )

        (max_weight, num_pauli_words, pauli_dist, commuting_groups) = _preprocess_inputs(
            num_qubits, num_pauli_words, max_weight, pauli_dist, commuting_groups
        )

        self._one_norm = one_norm
        self._max_weight = max_weight
        self._num_pauli_words = num_pauli_words
        self._pauli_dist = pauli_dist
        self._commuting_groups = commuting_groups

    def __repr__(self):
        """The repr dunder method for the PauliHamiltonian class."""
        return f"PauliHamiltonian(num_qubits={self.num_qubits}, num_pauli_words={self.num_pauli_words}, max_weight={self.max_weight}, one_norm={self.one_norm})"

    def __eq__(self, other: "PauliHamiltonian"):
        """Check if two PauliHamiltonians are identical"""
        return all(
            (
                self._num_qubits == other._num_qubits,
                self._num_pauli_words == other._num_pauli_words,
                self._max_weight == other._max_weight,
                self._pauli_dist == other._pauli_dist,
                self._commuting_groups == other._commuting_groups,
                self._one_norm == other._one_norm,
            )
        )

    def __hash__(self):
        """Hash function for the compact Hamiltonian representation"""
        hashable_param = None
        if self._commuting_groups is not None:
            hashable_param = tuple(_sort_and_freeze(group) for group in self._commuting_groups)
        elif self._pauli_dist is not None:
            hashable_param = _sort_and_freeze(self._pauli_dist)

        hashable_params = (
            self._num_qubits,
            self._num_pauli_words,
            self._max_weight,
            hashable_param,
            self._one_norm,
        )
        return hash(hashable_params)

    @property
    def num_qubits(self):
        """The number of qubits the Hamiltonian acts on"""
        return self._num_qubits

    @property
    def num_pauli_words(self):
        """The number of Pauli words (or terms) in the sum."""
        return self._num_pauli_words

    @property
    def max_weight(self):
        """The maximum number of Pauli operators (tensored) in any given term in the sum."""
        return self._max_weight

    @property
    def one_norm(self):
        """The one-norm of the Hamiltonian."""
        return self._one_norm

    @property
    def pauli_dist(self):
        """A dictionary representing the distribution of Pauli words in the Hamiltonian"""
        return copy.deepcopy(self._pauli_dist)

    @property
    def commuting_groups(self):
        """A list of groups where each group is a distribution of pauli words such that each
        term in the group commutes with every other term in the group."""
        return copy.deepcopy(self._commuting_groups)


def _sort_and_freeze(pauli_dist: dict) -> tuple[tuple]:
    """Map a dictionary into a sorted and hashable tuple"""
    return tuple((k, pauli_dist[k]) for k in sorted(pauli_dist))


def _validate_pauli_dist(pauli_dist: dict) -> bool:
    """Validate that the pauli_dist is formatted as expected"""
    for pauli_word, freq in pauli_dist.items():
        if (not isinstance(pauli_word, str)) or (
            not all(char in {"X", "Y", "Z"} for char in pauli_word)
        ):
            raise ValueError(
                f"The keys represent Pauli words and should be strings containing either 'X','Y' or 'Z' characters only. Got {pauli_word} : {freq}"
            )

        if not (isinstance(freq, int) and (not isinstance(freq, bool)) and (freq >= 0)):
            raise ValueError(
                f"The values represent frequencies and should be positive integers, got {pauli_word} : {freq}"
            )


def _pauli_dist_from_commuting_groups(commuting_groups: tuple[dict]):
    """Construct the total Pauli word distribution from the commuting groups."""
    total_pauli_dist = Counter()
    for group in commuting_groups:
        total_pauli_dist.update(group)
    return dict(total_pauli_dist)


def _preprocess_inputs(
    num_qubits: int,
    num_pauli_words: int | None,
    max_weight: int | None,
    pauli_dist: dict | None,
    commuting_groups: tuple[dict] | None,
) -> tuple:
    """Helper function to validate the inputs of PauliHamiltonian"""
    if commuting_groups is not None:
        for group in commuting_groups:
            _validate_pauli_dist(group)  #  ensure the groups are formatted correctly

        final_commuting_groups = commuting_groups
        final_pauli_dist = _pauli_dist_from_commuting_groups(commuting_groups)
        final_max_weight = max(len(pw) for pw in final_pauli_dist.keys())
        final_num_pauli_words = sum(final_pauli_dist.values())
        return (
            final_max_weight,
            final_num_pauli_words,
            final_pauli_dist,
            final_commuting_groups,
        )

    if pauli_dist is not None:
        _validate_pauli_dist(pauli_dist)

        final_pauli_dist = pauli_dist
        final_commuting_groups = None
        final_max_weight = max(len(pw) for pw in pauli_dist.keys())
        final_num_pauli_words = sum(pauli_dist.values())

        return (
            final_max_weight,
            final_num_pauli_words,
            final_pauli_dist,
            final_commuting_groups,
        )

    if num_pauli_words is None:
        raise ValueError(
            "One of the following sets of inputs must be provided (not None) in order to"
            " instantiate a valid PauliHamiltonian:\n - `commuting_groups`\n - `pauli_dist`\n"
            " - `num_pauli_words`"
        )

    if max_weight and (max_weight > num_qubits):
        raise ValueError(
            "`max_weight` represents the maximum number of qubits any Pauli word acts upon,"
            "this value must be less than or equal to the total number of qubits the "
            f"Hamiltonian acts on. Got `num_qubits` = {num_qubits} and `max_weight` = {max_weight}"
        )

    final_max_weight = max_weight or num_qubits
    return (final_max_weight, num_pauli_words, None, None)
