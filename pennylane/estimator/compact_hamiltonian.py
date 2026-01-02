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
from dataclasses import dataclass
from typing import Iterable


def _validate_positive_int(name, value):
    """Helper to validate positive integers."""
    if not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer, got {value}")


@dataclass(frozen=True)
class CDFHamiltonian:
    """For a compressed double-factorized (CDF) Hamiltonian, stores the minimum necessary information pertaining to resource estimation.

    The form of this Hamiltonian is described in `arXiv:2506.15784 <https://arxiv.org/abs/2506.15784>`_.

    Args:
        num_orbitals (int): number of spatial orbitals
        num_fragments (int): number of fragments in the compressed double-factorized (CDF) representation
        one_norm (float | None): the one-norm of the Hamiltonian

    Raises:
        TypeError: if ``num_orbitals``, or ``num_fragments`` is not a positive integer
        TypeError: if ``one_norm`` is provided but is not a non-negative float or integer

    .. seealso::
        :class:`~.estimator.templates.trotter.TrotterCDF`
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

    The form of this Hamiltonian is described in `arXiv:2407.04432 <https://arxiv.org/abs/2407.04432>`_.

    Args:
        num_orbitals (int): number of spatial orbitals
        tensor_rank (int):  tensor rank of two-body integrals in the tensor hypercontracted (THC) representation
        one_norm (float | None): the one-norm of the Hamiltonian

    Raises:
        TypeError: if ``num_orbitals``, or ``tensor_rank`` is not a positive integer
        TypeError: if ``one_norm`` is provided but is not a non-negative float or integer


    .. seealso::
        :class:`~.estimator.templates.trotter.TrotterTHC`
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

    The form of this Hamiltonian is described in `arXiv:2504.10602 <https://arxiv.org/pdf/2504.10602>`_.

    Args:
        num_modes (int): number of vibrational modes
        grid_size (int): number of grid points used to discretize each mode
        taylor_degree (int): degree of the Taylor expansion used in the vibrational representation
        one_norm (float | None): the one-norm of the Hamiltonian

    Raises:
        TypeError: if ``num_modes``, ``grid_size``, or ``taylor_degree`` is not a positive integer
        TypeError: if ``one_norm`` is provided but is not a non-negative float or integer

    .. seealso::
        :class:`~.estimator.templates.trotter.TrotterVibrational`
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

    The form of this Hamiltonian is described in `arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`_.

    Args:
        num_modes (int): number of vibronic modes
        num_states (int): number of vibronic states
        grid_size (int): number of grid points used to discretize each mode
        taylor_degree (int): degree of the Taylor expansion used in the vibronic representation
        one_norm (float | None): the one-norm of the Hamiltonian

    Raises:
        TypeError: if ``num_modes``, ``num_states``, ``grid_size``, or ``taylor_degree`` is not a positive integer
        TypeError: if ``one_norm`` is provided but is not a non-negative float or integer

    .. seealso::
        :class:`~.estimator.templates.trotter.TrotterVibronic`
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
    r"""Stores the minimum necessary information required to estimate resources for a Hamiltonian
    expressed as a linear combination of tensor products of Pauli operators.

    Args:
        num_qubits (int): total number of qubits the Hamiltonian acts on
        pauli_terms (dict[str, int] | Iterable[dict]): A dictionary representing the Hamiltonian terms
            where the keys are Pauli strings, e.g ``"XY"``, and the values are integers denoting
            how frequently a Pauli string appears in the Hamiltonian. When a list of dictionaries is
            provided, each dictionary is interpreted as a commuting group of terms. See the
            Usage Details section for more information.
        one_norm (float | int | None): the one-norm of the Hamiltonian

    Raises:
        TypeError: if ``pauli_terms`` is not a dictionary
        ValueError: if ``one_norm`` is provided but is not a non-negative float or integer
        ValueError: if ``pauli_terms`` contains invalid keys (not Pauli strings) or values (not integers)

    .. seealso::
        :class:`~.estimator.templates.trotter.TrotterPauli`, :class:`~.estimator.templates.select.SelectPauli`

    **Example**

    A ``PauliHamiltonian`` is a compact representation which can be used with compatible templates
    to obtain resource estimates. Consider for example the Hamiltonian:

    .. math::

        \hat{H} = 0.1 \cdot \Sigma^{30}_{j=1} \hat{X}_{j} \hat{X}_{j+1}
        - 0.05 \cdot \Sigma^{30}_{k=1} \hat{Y}_{k} \hat{Y}_{k+1} + 0.25 \cdot \Sigma^{40}_{l=1} \hat{X}_{l}

    This Hamiltonian is represented in a compact form using ``PauliHamiltonian``:

    >>> import pennylane.estimator as qre
    >>> pauli_ham = qre.PauliHamiltonian(
    ...     num_qubits = 40,
    ...     pauli_terms = {"X":40, "XX":30, "YY":30},
    ...     one_norm = 14.5,  # (|0.1| * 30) + (|-0.05| * 30) + (|0.25| * 40)
    ... )
    >>> pauli_ham
    PauliHamiltonian(num_qubits=40, one_norm=14.5, pauli_terms={'X': 40, 'XX': 30, 'YY': 30})

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
     Total gates : 9.400E+4
       'T': 8.800E+4,
       'CNOT': 2.400E+3,
       'Z': 1.200E+3,
       'S': 2.400E+3

    .. details::
        :title: Usage Details

        The terms of the Hamiltonian can also be separated into groups such that all operators in
        the group commute. Users can instantiate the ``PauliHamiltonian`` by specifying these
        groups of terms directly.

        >>> import pennylane.estimator as qre
        >>> commuting_groups = [
        ...     {"X": 40, "XX": 30}, # first commuting group
        ...     {"YY": 30}, # second commuting group
        ... ]
        >>> pauli_ham = qre.PauliHamiltonian(
        ...     num_qubits = 40,
        ...     pauli_terms = commuting_groups,
        ...     one_norm = 14.5,  # (|0.1| * 30) + (|-0.05| * 30) + (|0.25| * 40)
        ... )
        >>> pauli_ham
        PauliHamiltonian(num_qubits=40, one_norm=14.5, pauli_terms=[{'X': 40, 'XX': 30}, {'YY': 30}])

        Note that providing more information will generally lead to more accurate resource estimates.

        >>> num_steps, order = (10, 2)
        >>> res = qre.estimate(qre.TrotterPauli(pauli_ham, num_steps, order))
        >>> print(res)
        --- Resources: ---
         Total wires: 40
           algorithmic wires: 40
           allocated wires: 0
             zero state: 0
             any state: 0
         Total gates : 5.014E+4
           'T': 4.708E+4,
           'CNOT': 1.260E+3,
           'Z': 600,
           'S': 1.200E+3

    """

    def __init__(
        self,
        num_qubits: int,
        pauli_terms: dict | Iterable[dict],
        one_norm: int | float | None = None,
    ):
        self._num_qubits = num_qubits
        if one_norm is not None and not (isinstance(one_norm, (float, int)) and one_norm >= 0):
            raise ValueError(
                f"one_norm, if provided, must be a positive float or integer. Instead received {one_norm}"
            )

        if isinstance(pauli_terms, dict):
            _validate_pauli_terms(pauli_terms)
        else:
            for group in pauli_terms:
                _validate_pauli_terms(group)

        self._one_norm = one_norm
        self._pauli_terms = pauli_terms

    def __repr__(self):
        """The repr dunder method for the PauliHamiltonian class."""
        return f"PauliHamiltonian(num_qubits={self.num_qubits}, one_norm={self.one_norm}, pauli_terms={self.pauli_terms})"

    def __eq__(self, other: "PauliHamiltonian"):
        """Check if two PauliHamiltonians are identical"""
        return all(
            (
                self._num_qubits == other._num_qubits,
                self._pauli_terms == other._pauli_terms,
                self._one_norm == other._one_norm,
            )
        )

    def __hash__(self):
        """Hash function for the compact Hamiltonian representation"""
        if isinstance(self._pauli_terms, dict):
            hashable_param = _sort_and_freeze(self._pauli_terms)
        else:
            hashable_param = tuple(_sort_and_freeze(group) for group in self._pauli_terms)

        hashable_params = (
            self._num_qubits,
            hashable_param,
            self._one_norm,
        )
        return hash(hashable_params)

    @property
    def num_qubits(self):
        """The number of qubits the Hamiltonian acts on"""
        return self._num_qubits

    @property
    def one_norm(self):
        """The one-norm of the Hamiltonian"""
        return self._one_norm

    @property
    def pauli_terms(self):
        """A dictionary representing the distribution of Pauli words in the Hamiltonian"""
        return copy.deepcopy(self._pauli_terms)

    @property
    def num_terms(self) -> int:
        """The total number of Pauli words in the Hamiltonian"""
        if isinstance(self._pauli_terms, dict):
            return sum(self._pauli_terms.values())

        # Commuting groups are provided
        return sum(sum(group.values()) for group in self._pauli_terms)


def _sort_and_freeze(pauli_terms: dict) -> tuple[tuple]:
    """Map a dictionary into a sorted and hashable tuple"""
    return tuple((k, pauli_terms[k]) for k in sorted(pauli_terms))


def _validate_pauli_terms(pauli_terms: dict) -> bool:
    """Validate that the ``pauli_terms`` is formatted as expected"""
    if not isinstance(pauli_terms, dict):
        raise TypeError(
            f"Expected `pauli_terms` to be a dictionary or an iterable of dictionaries. got {pauli_terms}"
        )
    for pauli_word, freq in pauli_terms.items():
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
