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
This submodule provides the functionality to perform quantum chemistry calculations.
"""

from pennylane.math.decomposition import givens_decomposition
from .convert import import_operator, import_state
from .convert_openfermion import from_openfermion, to_openfermion
from .dipole import molecular_dipole
from .factorization import basis_rotation, factorize, symmetry_shift
from .hamiltonian import molecular_hamiltonian, molecular_integrals
from .molecule import Molecule
from .number import particle_number
from .observable_hf import fermionic_observable, qubit_observable
from .openfermion_pyscf import (
    decompose,
    dipole_of,
    meanfield,
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
from .vibrational import (
    VibrationalPES,
    christiansen_bosonic,
    christiansen_dipole,
    christiansen_hamiltonian,
    christiansen_integrals,
    christiansen_integrals_dipole,
    localize_normal_modes,
    optimize_geometry,
    taylor_bosonic,
    taylor_coeffs,
    taylor_dipole_coeffs,
    taylor_hamiltonian,
    vibrational_pes,
    vscf_integrals,
)
