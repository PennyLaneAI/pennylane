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
from .convert import (
    _openfermion_pennylane_equivalent,
    _openfermion_to_pennylane,
    _pennylane_to_openfermion,
    _process_wires,
    import_operator,
)
from .dipole import dipole_integrals, dipole_moment, fermionic_dipole
from .hamiltonian import electron_integrals, fermionic_hamiltonian, mol_hamiltonian
from .hartree_fock import hf_energy, nuclear_energy, scf
from .integrals import (
    _boys,
    _diff2,
    _generate_params,
    _hermite_coulomb,
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
from .observable import _pauli_mult, fermionic_observable, jordan_wigner, qubit_observable, simplify
from .particle_number import particle_number
from .spin import _spin2_matrix_elements, spin2, spinz
