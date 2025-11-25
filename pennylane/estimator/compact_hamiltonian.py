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

        if not isinstance(self.num_orbitals, int):
            raise TypeError(
                f"num_orbitals must be an integer, got {type(self.num_orbitals).__name__}"
            )
        if not isinstance(self.num_fragments, int):
            raise TypeError(
                f"num_fragments must be an integer, got {type(self.num_fragments).__name__}"
            )
        if self.one_norm is not None and not isinstance(self.one_norm, (float, int)):
            raise TypeError(
                f"one_norm must be a float or integer (or None), "
                f"but received type {type(self.one_norm).__name__}"
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

        if not isinstance(self.num_orbitals, int):
            raise TypeError(
                f"num_orbitals must be an integer, got {type(self.num_orbitals).__name__}"
            )
        if not isinstance(self.tensor_rank, int):
            raise TypeError(
                f"tensor_rank must be an integer, got {type(self.tensor_rank).__name__}"
            )

        if self.one_norm is not None and not isinstance(self.one_norm, (float, int)):
            raise TypeError(
                f"one_norm must be a float or integer (or None), "
                f"but received type {type(self.one_norm).__name__}"
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

        if not isinstance(self.num_modes, int):
            raise TypeError(f"num_modes must be an integer, got {type(self.num_modes).__name__}")
        if not isinstance(self.grid_size, int):
            raise TypeError(f"grid_size must be an integer, got {type(self.grid_size).__name__}")
        if not isinstance(self.taylor_degree, int):
            raise TypeError(
                f"taylor_degree must be an integer, got {type(self.taylor_degree).__name__}"
            )

        if self.one_norm is not None and not isinstance(self.one_norm, (float, int)):
            raise TypeError(
                f"one_norm must be a float or integer (or None), "
                f"but received type {type(self.one_norm).__name__}"
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

        if not isinstance(self.num_modes, int):
            raise TypeError(f"num_modes must be an integer, got {type(self.num_modes).__name__}")
        if not isinstance(self.num_states, int):
            raise TypeError(f"num_states must be an integer, got {type(self.num_states).__name__}")
        if not isinstance(self.grid_size, int):
            raise TypeError(f"grid_size must be an integer, got {type(self.grid_size).__name__}")
        if not isinstance(self.taylor_degree, int):
            raise TypeError(
                f"taylor_degree must be an integer, got {type(self.taylor_degree).__name__}"
            )

        if self.one_norm is not None and not isinstance(self.one_norm, (float, int)):
            raise TypeError(
                f"one_norm must be a float or integer (or None), "
                f"but received type {type(self.one_norm).__name__}"
            )

        if isinstance(self.one_norm, int):
            object.__setattr__(self, "one_norm", float(self.one_norm))
