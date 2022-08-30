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


class Spindata(Dataset):
    """Class for quantum spin systems dataset"""

    def __init__(
        self,
        parameters,
        hamiltonians,
        phase_labels,
        ground_energies,
        ground_states,
        correlation_matrix,
        classical_shadows,
    ):
        self.parameters = parameters
        self.hamiltonians = hamiltonians
        self.phase_labels = phase_labels
        self.ground_energies = ground_energies
        self.ground_states = ground_states
        self.correlation_matrix = correlation_matrix
        self.classical_shadows = classical_shadows
