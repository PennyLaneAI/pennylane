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
Contains the ChemDataset class.
"""

from .dataset import Dataset

# pylint: disable= too-many-instance-attributes too-many-arguments
class ChemDataset(Dataset):
    """Class for quantum chemistry dataset"""

    def __init__(
        self,
        molecule=None,
        hamiltonian=None,
        meas_groupings=None,
        symmetries=None,
        paulix_ops=None,
        optimal_sector=None,
        dipole_op=None,
        num_op=None,
        spin2_op=None,
        spinz_op=None,
        tapered_hamiltonian=None,
        tapered_dipole_op=None,
        tapered_num_op=None,
        tapered_spin2_op=None,
        tapered_spinz_op=None,
        vqe_params=None,
        vqe_circuit=None,
        vqe_energy=None,
        fci_energy=None,
        data_file=None,
    ):
        self._molecule = molecule
        self._hamiltonian = hamiltonian
        self._meas_groupings = meas_groupings
        self._symmetries = symmetries
        self._paulix_ops = paulix_ops
        self._optimal_sector = optimal_sector
        self._num_op = num_op
        self._spin2_op = spin2_op
        self._spinz_op = spinz_op
        self._dipole_op = dipole_op
        self._tapered_hamiltonian = tapered_hamiltonian
        self._tapered_dipole_op = tapered_dipole_op
        self._tapered_num_op = tapered_num_op
        self._tapered_spin2_op = tapered_spin2_op
        self._taperes_spinz_op = tapered_spinz_op
        self._vqe_params = vqe_params
        self._vqe_circuit = vqe_circuit
        self._vqe_energy = vqe_energy
        self._classical_energy = fci_energy
        super().__init__(data_file=data_file)

    @property
    def molecule(self):
        """Molecule property of ChemDataset"""
        if self._molecule is None:
            if self._data_file:
                self._molecule = self.read_data(f"{self._data_file}_molecule.dat")
        return self._molecule

    @molecule.setter
    def molecule(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_molecule.dat", value)
        self._molecule = value

    @property
    def hamiltonian(self):
        """Hamiltonian property of ChemDataset"""
        if self._hamiltonian is None:
            if self._data_file:
                self._hamiltonian = self.read_data(f"{self._data_file}_hamiltonian.dat")
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_hamiltonian.dat", value)
        self._hamiltonian = value

    @property
    def meas_groupings(self):
        """Measurement groupings property of ChemDataset"""
        if self._meas_groupings is None:
            if self._data_file:
                self._meas_groupings = self.read_data(f"{self._data_file}_meas_groupings.dat")
        return self._meas_groupings

    @meas_groupings.setter
    def meas_groupings(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_meas_groupings.dat", value)
        self._meas_groupings = value

    @property
    def symmetries(self):
        """Symmetries property of ChemDataset"""
        if self._symmetries is None:
            if self._data_file:
                self._symmetries = self.read_data(f"{self._data_file}_symmetries.dat")
        return self._symmetries

    @symmetries.setter
    def symmetries(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_symmetries.dat", value)
        self._symmetries = value

    @property
    def paulix_ops(self):
        """PauliX operators property of ChemDataset"""
        if self._paulix_ops is None:
            if self._data_file:
                self._paulix_ops = self.read_data(f"{self._data_file}_paulix_ops.dat")
        return self._paulix_ops

    @paulix_ops.setter
    def paulix_ops(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_paulix_ops.dat", value)
        self._paulix_ops = value

    @property
    def optimal_sector(self):
        """Optimal sector property of ChemDataset"""
        if self._optimal_sector is None:
            if self._data_file:
                self._optimal_sector = self.read_data(f"{self._data_file}_optimal_sector.dat")
        return self._optimal_sector

    @optimal_sector.setter
    def optimal_sector(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_optimal_sector.dat", value)
        self._optimal_sector = value

    @property
    def dipole_op(self):
        """Dipole moment operator property of ChemDataset"""
        if self._dipole_op is None:
            if self._data_file:
                self._dipole_op = self.read_data(f"{self._data_file}_dipole_op.dat")
        return self._dipole_op

    @dipole_op.setter
    def dipole_op(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_dipole_op.dat", value)
        self._dipole_op = value

    @property
    def num_op(self):
        """Number operator property of ChemDataset"""
        if self._num_op is None:
            if self._data_file:
                self._num_op = self.read_data(f"{self._data_file}_num_op.dat")
        return self._num_op

    @num_op.setter
    def num_op(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_num_op.dat", value)
        self._num_op = value

    @property
    def spin2_op(self):
        """Spin:math:`^2` operator property of ChemDataset"""
        if self._spin2_op is None:
            if self._data_file:
                self._spin2_op = self.read_data(f"{self._data_file}_spin2_op.dat")
        return self._spin2_op

    @spin2_op.setter
    def spin2_op(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_spin2_op.dat", value)
        self._spin2_op = value

    @property
    def spinz_op(self):
        """Spin:math:`_Z` operator property of ChemDataset"""
        if self._spinz_op is None:
            if self._data_file:
                self._spinz_op = self.read_data(f"{self._data_file}_spinz_op.dat")
        return self._spinz_op

    @spinz_op.setter
    def spinz_op(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_spinz_op.dat", value)
        self._spinz_op = value

    @property
    def tapered_hamiltonian(self):
        """Tapered Hamiltonian property of ChemDataset"""
        if self._tapered_hamiltonian is None:
            if self._data_file:
                self._tapered_hamiltonian = self.read_data(
                    f"{self._data_file}_tapered_hamiltonian.dat"
                )
        return self._tapered_hamiltonian

    @tapered_hamiltonian.setter
    def tapered_hamiltonian(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_tapered_hamiltonian.dat", value)
        self._tapered_hamiltonian = value

    @property
    def tapered_dipole_op(self):
        """Tapered dipole moment operator property of ChemDataset"""
        if self._tapered_dipole_op is None:
            if self._data_file:
                self._tapered_dipole_op = self.read_data(f"{self._data_file}_tapered_dipole_op.dat")
        return self._tapered_dipole_op

    @tapered_dipole_op.setter
    def tapered_dipole_op(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_tapered_dipole_op.dat", value)
        self._tapered_dipole_op = value

    @property
    def tapered_num_op(self):
        """Tapered number operator property of ChemDataset"""
        if self._tapered_num_op is None:
            if self._data_file:
                self._tapered_num_op = self.read_data(f"{self._data_file}_tapered_num_op.dat")
        return self._tapered_num_op

    @tapered_num_op.setter
    def tapered_num_op(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_tapered_num_op.dat", value)
        self._tapered_num_op = value

    @property
    def tapered_spin2_op(self):
        """Tapered spin:math:`^2` property of ChemDataset"""
        if self._tapered_spin2_op is None:
            if self._data_file:
                self._tapered_spin2_op = self.read_data(f"{self._data_file}_tapered_spin2_op.dat")
        return self._tapered_spin2_op

    @tapered_spin2_op.setter
    def tapered_spin2_op(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_tapered_spin2_op.dat", value)
        self._tapered_spin2_op = value

    @property
    def tapered_spinz_op(self):
        """Tapered spin:math:`_Z` property of ChemDataset"""
        if self._tapered_spinz_op is None:
            if self._data_file:
                self._tapered_spinz_op = self.read_data(f"{self._data_file}_tapered_spinz_op.dat")
        return self._tapered_spinz_op

    @tapered_spinz_op.setter
    def tapered_spinz_op(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_tapered_spinz_op.dat", value)
        self._tapered_spinz_op = value

    @property
    def vqe_params(self):
        """VQE parameters property of ChemDataset"""
        if self._vqe_params is None:
            if self._data_file:
                self._vqe_params = self.read_data(f"{self._data_file}_vqe_params.dat")
        return self._vqe_params

    @vqe_params.setter
    def vqe_params(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_vqe_params.dat", value)
        self._vqe_params = value

    @property
    def vqe_circuit(self):
        """VQE circuit property of ChemDataset"""
        if self._vqe_circuit is None:
            if self._data_file:
                self._vqe_circuit = self.read_data(f"{self._data_file}_vqe_circuit.dat")
        return self._vqe_circuit

    @vqe_circuit.setter
    def vqe_circuit(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_vqe_circuit.dat", value)
        self._vqe_circuit = value

    @property
    def vqe_energy(self):
        """VQE energy property of ChemDataset"""
        if self._vqe_energy is None:
            if self._data_file:
                self._vqe_energy = self.read_data(f"{self._data_file}_vqe_energy.dat")
        return self._vqe_energy

    @vqe_energy.setter
    def vqe_energy(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_vqe_energy.dat", value)
        self._vqe_energy = value

    @property
    def fci_energy(self):
        """Classical ground-state energy property of ChemDataset"""
        if self._classical_energy is None:
            if self._data_file:
                self._classical_energy = self.read_data(f"{self._data_file}_classical_energy.dat")
        return self._classical_energy

    @fci_energy.setter
    def fci_energy(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_classical_energy.dat", value)
        self._classical_energy = value
