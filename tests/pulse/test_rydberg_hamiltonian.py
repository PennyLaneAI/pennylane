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
Unit tests for the RydbergHamiltonian class.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.ops import Sum
from pennylane.pulse import ParametrizedHamiltonian, RydbergHamiltonian
from pennylane.wires import Wires

atom_coordinates = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
wires = [1, 6, 9, 2, 4, 3]


# pylint: disable=protected-access
def test_initialization():
    """Test the RydbergHamiltonian class is initialized correctly."""
    rm = RydbergHamiltonian(coordinates=atom_coordinates, wires=wires)

    assert qml.math.allequal(rm.coordinates, atom_coordinates)
    assert rm.wires == Wires(wires)
    assert isinstance(rm.driving_interaction, ParametrizedHamiltonian)
    assert rm.driving_interaction.H_parametrized([], 0) == 0
    assert rm.driving_interaction.H_fixed() == 0
    assert rm._rydberg_interaction is None
    assert rm.interaction_coeff == 862690 * np.pi
    assert rm._local_drives == {"rabi": [], "detunings": [], "phases": [], "wires": []}
    assert rm._global_drive is None


class TestProperties:
    """Unit tests for the properties of the RydbergHamiltonian class."""

    def test_rydberg_interaction(self):
        """Test that the rydberg_interaction property returns a Hamiltonian class, and that it
        contains the correct amount of coeffs and ops."""
        rm = RydbergHamiltonian(coordinates=atom_coordinates, wires=wires)

        N = len(wires)
        num_combinations = N * (N - 1) / 2  # number of terms on the rydberg_interaction hamiltonian
        assert isinstance(rm.rydberg_interaction, Sum)
        assert len(rm.rydberg_interaction.operands) == num_combinations

    def test_driving_interaction(self):
        """Test the driving_interaction property."""
        rm = RydbergHamiltonian(coordinates=atom_coordinates, wires=wires)

        assert isinstance(rm.driving_interaction, ParametrizedHamiltonian)
        assert rm.driving_interaction.H_parametrized([], 0) == 0
        assert rm.driving_interaction.H_fixed() == 0

    def test_hamiltonian(self):
        """Test the hamiltonian property."""
        rm = RydbergHamiltonian(coordinates=atom_coordinates, wires=wires)

        assert isinstance(rm.hamiltonian, ParametrizedHamiltonian)
        assert rm.hamiltonian.H_parametrized([], 0) == 0
        assert qml.equal(
            qml.simplify(rm.hamiltonian.H_fixed()), qml.simplify(rm.rydberg_interaction)
        )


class TestMethods:
    """Unit tests for the RydbergHamiltonian methods."""

    def test_local_drive_updates_dictionaries(self):
        """Test that the local_drive method updates the internal dictionaries."""
        rm = RydbergHamiltonian(atom_coordinates, wires)

        assert rm._local_drives == {"rabi": [], "detunings": [], "phases": [], "wires": []}

        rm.local_drive(rabi=[0, 1, 2], detunings=[3, 4, 5], phases=[6, 7, 8], wires=[1, 2, 3])

        assert rm._local_drives == {
            "rabi": [0, 1, 2],
            "detunings": [3, 4, 5],
            "phases": [6, 7, 8],
            "wires": [1, 2, 3],
        }

    def test_local_drive_updates_driving_hamiltonian(self):
        """Test that the local_drive method updates the driving_interaction term of the Hamiltonian."""
        rm = RydbergHamiltonian(coordinates=atom_coordinates, wires=wires)

        assert isinstance(rm.driving_interaction, ParametrizedHamiltonian)
        assert rm.driving_interaction([], 0) == 0

        rm.local_drive(rabi=[lambda p, t: 1], detunings=[0], phases=[0], wires=[3])

        assert rm.driving_interaction([1], 1) != 0

    def test_local_drive_wrong_lengths_raises_error(self):
        """Test that the local_drive method raises an error when the inputs have different lengths."""
        rm = RydbergHamiltonian(atom_coordinates, wires)

        with pytest.raises(
            ValueError, match="The lists containing the driving parameters must all have the same"
        ):
            rm.local_drive(rabi=[1, 2, 3], detunings=[0], phases=[0], wires=[1])

    def test_local_drive_wrong_wires_raises_error(self):
        """Test that the local_drive method raises an error when a wire value is not present in the
        RydbergHamiltonian."""
        rm = RydbergHamiltonian(atom_coordinates, wires)

        with pytest.raises(
            ValueError,
            match="The wires list contains a wire value that is not present in the RydbergHamiltonian",
        ):
            rm.local_drive(rabi=[1, 2, 3], detunings=[0, 1, 2], phases=[0, -1, 0], wires=[1, 5, 0])

    def test_global_drive_updates_dictionaries(self):
        """Test that the global_drive method updates the internal dictionaries."""
        rm = RydbergHamiltonian(atom_coordinates, wires)

        assert rm._global_drive is None

        rm.global_drive(rabi=1, detuning=2, phase=3)

        assert rm._global_drive == (1, 2, 3)

    def test_global_drive_updates_driving_hamiltonian(self):
        """Test that the global_drive method updates the driving_interaction term of the Hamiltonian."""
        rm = RydbergHamiltonian(coordinates=atom_coordinates, wires=wires)

        assert isinstance(rm.driving_interaction, ParametrizedHamiltonian)
        assert rm.driving_interaction([], 0) == 0

        rm.global_drive(rabi=1, detuning=2, phase=3)

        assert rm.driving_interaction([], 0) != 0
