# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This subpackage provides the functionality to perform quantum chemistry calculations.
"""
from .openfermion_obs import (
    observable,
    one_particle,
    two_particle,
    dipole_of,
    meanfield,
    decompose,
    molecular_hamiltonian,
)
from .basis_set import BasisFunction, atom_basis_data, mol_basis_data
from .convert import import_operator
from .dipole import dipole_integrals, fermionic_dipole, dipole_moment
from .factorization import basis_rotation, factorize
from .hamiltonian import electron_integrals, fermionic_hamiltonian, diff_hamiltonian
from .hartree_fock import scf, nuclear_energy, hf_energy
from .integrals import (
    primitive_norm,
    contracted_norm,
    expansion,
    gaussian_overlap,
    overlap_integral,
    hermite_moment,
    gaussian_moment,
    moment_integral,
    gaussian_kinetic,
    kinetic_integral,
    nuclear_attraction,
    attraction_integral,
    electron_repulsion,
    repulsion_integral,
)
from .matrices import (
    mol_density_matrix,
    overlap_matrix,
    moment_matrix,
    kinetic_matrix,
    attraction_matrix,
    repulsion_tensor,
    core_matrix,
)
from .molecule import Molecule
from .observable_hf import fermionic_observable, qubit_observable, jordan_wigner
from .number import particle_number
from .spin import spin2, spinz
from .structure import read_structure, active_space, excitations, hf_state, excitations_to_wires
from .tapering import (
    clifford,
    paulix_ops,
    symmetry_generators,
    optimal_sector,
    taper,
    taper_hf,
    taper_operation,
)
