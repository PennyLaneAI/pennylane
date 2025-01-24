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

from .taylor_ham import (
    taylor_hamiltonian,
    taylor_bosonic,
    taylor_coeffs,
    taylor_dipole_coeffs,
)
from .localize_modes import localize_normal_modes
from .vibrational_class import VibrationalPES, optimize_geometry
from .vscf import vscf_integrals
