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
    """Constructs a compressed double factorized Hamiltonian instance

    Args:
        num_orbitals (int): number of spatial orbitals
        num_fragments (int): number of fragments in the compressed double factorization (CDF) representation

    Returns:
        CDFHamiltonian: An instance of CDFHamiltonian
    """

    num_orbitals: int
    num_fragments: int


@dataclass(frozen=True)
class THCHamiltonian:
    """Constructs a tensor hypercontracted Hamiltonian instance

    Args:
        num_orbitals (int): number of spatial orbitals
        tensor_rank (int):  tensor rank of two-body integrals in the tensor hypercontracted (THC) representation

    Returns:
        THCHamiltonian: An instance of THCHamiltonian
    """

    num_orbitals: int
    tensor_rank: int


@dataclass(frozen=True)
class VibrationalHamiltonian:
    """Constructs a vibrational Hamiltonian instance

    Args:
        num_modes (int): number of vibrational modes
        grid_size (int): number of grid points used to discretize each mode
        taylor_degree (int): degree of the Taylor expansion used in the vibrational representation

    Returns:
        VibrationalHamiltonian: An instance of VibrationalHamiltonian
    """

    num_modes: int
    grid_size: int
    taylor_degree: int


@dataclass(frozen=True)
class VibronicHamiltonian:
    """Constructs a vibronic Hamiltonian instance

    Args:
        num_modes (int): number of vibronic modes
        num_states (int): number of vibronic states
        grid_size (int): number of grid points used to discretize each mode
        taylor_degree (int): degree of the Taylor expansion used in the vibronic representation

    Returns:
        VibronicHamiltonian: An instance of VibronicHamiltonian
    """

    num_modes: int
    num_states: int
    grid_size: int
    taylor_degree: int
