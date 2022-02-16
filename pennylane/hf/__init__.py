# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This subpackage provides the functionality to perform differentiable
Hartree-Fock calculations.
"""
from .basis_data import STO3G, atomic_numbers
from .basis_set import BasisFunction, atom_basis_data, mol_basis_data
from .hamiltonian import (
    generate_electron_integrals,
    generate_fermionic_hamiltonian,
    generate_hamiltonian,
    simplify,
)
from .hartree_fock import generate_scf, hf_energy, nuclear_energy
from .integrals import (
    contracted_norm,
    electron_repulsion,
    expansion,
    gaussian_kinetic,
    gaussian_overlap,
    generate_attraction,
    generate_kinetic,
    generate_overlap,
    generate_repulsion,
    nuclear_attraction,
    primitive_norm,
)
from .matrices import (
    generate_attraction_matrix,
    generate_core_matrix,
    generate_kinetic_matrix,
    generate_overlap_matrix,
    generate_repulsion_tensor,
    molecular_density_matrix,
)
from .molecule import Molecule
from .tapering import (
    clifford,
    generate_paulis,
    generate_symmetries,
    get_generators,
    optimal_sector,
    transform_hamiltonian,
    transform_hf,
)
