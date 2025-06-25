# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This submodule provides the functionality to calculate vibrational Hamiltonians.
"""

from .christiansen_ham import christiansen_bosonic, christiansen_dipole, christiansen_hamiltonian
from .christiansen_utils import christiansen_integrals, christiansen_integrals_dipole
from .localize_modes import localize_normal_modes
from .pes_generator import vibrational_pes
from .taylor_ham import taylor_bosonic, taylor_coeffs, taylor_dipole_coeffs, taylor_hamiltonian
from .vibrational_class import VibrationalPES, optimize_geometry
from .vscf import vscf_integrals
