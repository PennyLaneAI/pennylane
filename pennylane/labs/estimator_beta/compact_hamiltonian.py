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
class DFHamiltonian:
    """For a double-factorized (DF) Hamiltonian, stores the minimum necessary information pertaining to resource estimation.

    Args:
        num_orbitals (int): number of spatial orbitals
        num_fragments (int): number of fragments in the compressed double-factorized (CDF) representation
        num_eigenvectors (int): number of eigenvectors used in the double-factorized (DF) representation

    Returns:
        CDFHamiltonian: An instance of CDFHamiltonian

    .. seealso::
        :class:`~.estimator.templates.TrotterCDF`
    """

    num_orbitals: int
    num_fragments: int
    num_eigenvectors: int
