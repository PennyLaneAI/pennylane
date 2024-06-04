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
from .basis_data import load_basisset
from .basis_set import BasisFunction, atom_basis_data, mol_basis_data
from .convert import import_operator, import_state
from .dipole import dipole_integrals, dipole_moment, fermionic_dipole
from .factorization import basis_rotation, factorize
from .givens_decomposition import givens_decomposition
from .hamiltonian import diff_hamiltonian, electron_integrals, fermionic_hamiltonian
from .hartree_fock import hf_energy, nuclear_energy, scf
from .integrals import (
    attraction_integral,
    contracted_norm,
    electron_repulsion,
    expansion,
    gaussian_kinetic,
    gaussian_moment,
    gaussian_overlap,
    hermite_moment,
    kinetic_integral,
    moment_integral,
    nuclear_attraction,
    overlap_integral,
    primitive_norm,
    repulsion_integral,
)
from .matrices import (
    attraction_matrix,
    core_matrix,
    kinetic_matrix,
    mol_density_matrix,
    moment_matrix,
    overlap_matrix,
    repulsion_tensor,
)
from .molecule import Molecule
from .number import particle_number
from .observable_hf import fermionic_observable, qubit_observable
from .openfermion_obs import (
    decompose,
    dipole_of,
    meanfield,
    molecular_hamiltonian,
    observable,
    one_particle,
    two_particle,
)
from .spin import spin2, spinz
from .structure import (
    active_space,
    excitations,
    excitations_to_wires,
    hf_state,
    mol_data,
    read_structure,
)
from .tapering import (
    clifford,
    optimal_sector,
    paulix_ops,
    symmetry_generators,
    taper,
    taper_hf,
    taper_operation,
)
