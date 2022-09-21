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
Contains the spindata class.
"""

from .dataset import Dataset

# pylint: disable= too-many-instance-attributes too-many-arguments
class SpinDataset(Dataset):
    """Class for quantum spin systems dataset"""

    def __init__(
        self,
        spin_system=None,
        parameters=None,
        hamiltonians=None,
        phase_labels=None,
        order_parameters=None,
        ground_energies=None,
        ground_states=None,
        classical_shadows=None,
        data_file=None,
    ):
        self._spin_system = spin_system
        self._parameters = parameters
        self._hamiltonians = hamiltonians
        self._phase_labels = phase_labels
        self._order_parameters = order_parameters
        self._ground_energies = ground_energies
        self._ground_states = ground_states
        self._classical_shadows = classical_shadows
        super().__init__(data_file=data_file)

    @property
    def spin_system(self):
        """Spin system property of SpinDataset"""
        if self._spin_system is None:
            if self._data_file:
                self._spin_system = self.read_data(f"{self._data_file}_spin_system.dat")
        return self._spin_system

    @spin_system.setter
    def spin_system(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_spin_system.dat", value)
        self._spin_system = value

    @property
    def parameters(self):
        """Parameters property of SpinDataset"""
        if self._parameters is None:
            if self._data_file:
                self._parameters = self.read_data(f"{self._data_file}_parameters.dat")
        return self._parameters

    @parameters.setter
    def parameters(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_parameters.dat", value)
        self._parameters = value

    @property
    def hamiltonians(self):
        """Hamiltonian property of SpinDataset"""
        if self._hamiltonians is None:
            if self._data_file:
                self._hamiltonians = self.read_data(f"{self._data_file}_hamiltonians.dat")
        return self._hamiltonians

    @hamiltonians.setter
    def hamiltonians(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_hamiltonians.dat", value)
        self._hamiltonians = value

    @property
    def phase_labels(self):
        """Phase labels property of SpinDataset"""
        if self._phase_labels is None:
            if self._data_file:
                self._phase_labels = self.read_data(f"{self._data_file}_phase_labels.dat")
        return self._phase_labels

    @phase_labels.setter
    def phase_labels(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_phase_labels.dat", value)
        self._phase_labels = value

    @property
    def order_parameters(self):
        """Order parameters property of SpinDataset"""
        if self._order_parameters is None:
            if self._data_file:
                self._order_parameters = self.read_data(f"{self._data_file}_order_parameters.dat")
        return self._order_parameters

    @order_parameters.setter
    def order_parameters(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_order_parameters.dat", value)
        self._order_parameters = value

    @property
    def ground_energies(self):
        """Ground-state energy property of SpinDataset"""
        if self._ground_energies is None:
            if self._data_file:
                self._ground_energies = self.read_data(f"{self._data_file}_ground_energies.dat")
        return self._ground_energies

    @ground_energies.setter
    def ground_energies(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_ground_energies.dat", value)
        self._ground_energies = value

    @property
    def ground_states(self):
        """Ground-state property of SpinDataset"""
        if self._ground_states is None:
            if self._data_file:
                self._ground_states = self.read_data(f"{self._data_file}_ground_states.dat")
        return self._ground_states

    @ground_states.setter
    def ground_states(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_ground_states.dat", value)
        self._ground_states = value

    @property
    def classical_shadows(self):
        """Classical shadow property of SpinDataset"""
        if self._classical_shadows is None:
            if self._data_file:
                self._classical_shadows = self.read_data(f"{self._data_file}_classical_shadows.dat")
        return self._classical_shadows

    @classical_shadows.setter
    def classical_shadows(self, value, write=False):
        if self._data_file and write:
            self.write_data(f"{self._data_file}_classical_shadows.dat", value)
        self._classical_shadows = value
