# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Contains the Chemdata class.
"""

from .dataset import Dataset


class Chemdata(Dataset):
    """Class for quantum chemistry dataset"""

    def __init__(
        self,
        molecule,
        hamiltonian=None,
        symmetries=None,
        paulix_ops=None,
        optimal_sector=None,
        num_op=None,
        spin2_op=None,
        spinz_op=None,
        tapered_hamiltonian=None,
        tapered_num_op=None,
        tapered_spin2_op=None,
        tapered_spinz_op=None,
        adaptive_params=None,
        adaptive_excitations=None,
        adaptive_energy=None,
    ):
        self.molecule = molecule
        self.hamiltonian = hamiltonian
        self.symmetries = symmetries
        self.paulix_ops = paulix_ops
        self.optimal_sector = optimal_sector
        self.num_op = num_op
        self.spin2_op = spin2_op
        self.spinz_op = spinz_op
        self.tapered_hamiltonian = tapered_hamiltonian
        self.tapered_num_op = tapered_num_op
        self.tapered_spin2_op = tapered_spin2_op
        self.taperes_spinz_op = tapered_spinz_op
        self.adaptive_params = adaptive_params
        self.adaptive_excitations = adaptive_excitations
        self.adaptive_energy = adaptive_energy
