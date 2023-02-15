# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for the RydbergMachine class.
"""
import numpy as np

import pennylane as qml
from pennylane.ops import Sum
from pennylane.pulse import ParametrizedHamiltonian, RydbergMachine
from pennylane.wires import Wires

atom_coordinates = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
wires = [1, 6, 9, 2, 4, 3]


# pylint: disable=protected-access
def test_initialization():
    """Test the RydbergMachine class is initialized correctly."""
    rm = RydbergMachine(coordinates=atom_coordinates, wires=wires)

    assert qml.math.allequal(rm.coordinates, atom_coordinates)
    assert rm.wires == Wires(wires)
    assert isinstance(rm.driving_interaction, ParametrizedHamiltonian)
    assert rm.driving_interaction.H_parametrized([], 0) == 0
    assert rm.driving_interaction.H_fixed() == 0
    assert rm._rydberg_interaction is None
    assert rm.interaction_coeff == 862690 * np.pi
    assert rm._local_drives == {"rabi": [], "detunings": [], "phases": [], "wires": []}
    assert rm._global_drive == {"rabi": [], "detunings": [], "phases": []}


class TestProperties:
    """Unit tests for the properties of the RydbergMachine class."""

    def test_rydberg_interaction(self):
        """Test that the rydberg_interaction property returns a Hamiltonian class, and that it
        contains the correct amount of coeffs and ops."""
        rm = RydbergMachine(coordinates=atom_coordinates, wires=wires)

        N = len(wires)
        num_combinations = N * (N - 1) / 2  # number of terms on the rydberg_interaction hamiltonian
        assert isinstance(rm.rydberg_interaction, Sum)
        assert len(rm.rydberg_interaction.operands) == num_combinations

    def test_driving_interaction(self):
        """Test the driving_interaction property."""
        rm = RydbergMachine(coordinates=atom_coordinates, wires=wires)

        assert isinstance(rm.driving_interaction, ParametrizedHamiltonian)
        assert rm.driving_interaction.H_parametrized([], 0) == 0
        assert rm.driving_interaction.H_fixed() == 0

    def test_hamiltonian(self):
        """Test the hamiltonian property."""
        rm = RydbergMachine(coordinates=atom_coordinates, wires=wires)

        assert isinstance(rm.hamiltonian, ParametrizedHamiltonian)
        assert rm.hamiltonian.H_parametrized([], 0) == 0
        assert qml.equal(
            qml.simplify(rm.hamiltonian.H_fixed()), qml.simplify(rm.rydberg_interaction)
        )

        rm.local_drive(rabi=[lambda p, t: 1], detunings=[0], phases=[0], wires=[3])

        assert rm.hamiltonian.H_parametrized([1], 1) != 0
