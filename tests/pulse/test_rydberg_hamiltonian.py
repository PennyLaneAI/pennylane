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
# pylint: disable=protected-access
import numpy as np
import pytest

import pennylane as qml
from pennylane.ops import Sum
from pennylane.pulse import RydbergHamiltonian, global_drive, local_drive
from pennylane.wires import Wires

atom_coordinates = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
wires = [1, 6, 9, 2, 4, 3]


class TestRydbergHamiltonian:
    """Unit tests for the RydbergHamiltonian class."""

    def test_initialization(self):
        """Test the RydbergHamiltonian class is initialized correctly."""
        rm = RydbergHamiltonian(coordinates=atom_coordinates, wires=wires)

        assert qml.math.allequal(rm.coordinates, atom_coordinates)
        assert rm.wires == Wires(wires)
        assert rm._hamiltonian is None
        assert rm.interaction_coeff == 862690 * np.pi
        assert rm._local_drives == {"rabi": [], "detunings": [], "phases": [], "wires": []}
        assert rm._global_drive is None

    def test_hamiltonian(self):
        """Test that the hamiltonian property returns a Hamiltonian class, and that it
        contains the correct amount of coeffs and ops."""
        rm = RydbergHamiltonian(coordinates=atom_coordinates, wires=wires)

        N = len(wires)
        num_combinations = N * (N - 1) / 2  # number of terms on the rydberg_interaction hamiltonian
        assert isinstance(rm.hamiltonian, Sum)
        assert len(rm.hamiltonian.operands) == num_combinations


class TestLocalDrive:
    """Unit tests for the ``local_drive`` function."""

    def test_local_drive_updates_dictionaries(self):
        """Test that the local_drive function updates the internal dictionaries."""
        rm = RydbergHamiltonian(atom_coordinates, wires)

        assert rm._local_drives == {"rabi": [], "detunings": [], "phases": [], "wires": []}

        H = local_drive(rm, rabi=[0, 1, 2], detunings=[3, 4, 5], phases=[6, 7, 8], wires=[1, 2, 3])

        assert isinstance(H, Sum)
        assert rm._local_drives == {
            "rabi": [0, 1, 2],
            "detunings": [3, 4, 5],
            "phases": [6, 7, 8],
            "wires": [1, 2, 3],
        }

    def test_local_drive_updates_hamiltonian(self):
        """Test that the local_drive function returns an updated Hamiltonian that contains the
        driving interaction term."""
        rm = RydbergHamiltonian(coordinates=atom_coordinates, wires=wires)

        H = local_drive(rm, rabi=[lambda p, t: 1], detunings=[0], phases=[0], wires=[3])

        assert H is not rm.hamiltonian

    def test_local_drive_wrong_lengths_raises_error(self):
        """Test that the local_drive function raises an error when the inputs have different lengths."""
        rm = RydbergHamiltonian(atom_coordinates, wires)

        with pytest.raises(
            ValueError, match="The lists containing the driving parameters must all have the same"
        ):
            local_drive(rm, rabi=[1, 2, 3], detunings=[0], phases=[0], wires=[1])

    def test_local_drive_wrong_wires_raises_error(self):
        """Test that the local_drive method raises an error when a wire value is not present in the
        RydbergHamiltonian."""
        rm = RydbergHamiltonian(atom_coordinates, wires)

        with pytest.raises(
            ValueError,
            match="The wires list contains a wire value that is not present in the RydbergHamiltonian",
        ):
            local_drive(rm, rabi=[1, 2, 3], detunings=[0, 1, 2], phases=[0, -1, 0], wires=[1, 5, 0])


class TestGlobalDrive:
    """Unit tests for the ``global_drive`` function."""

    def test_global_drive_updates_dictionaries(self):
        """Test that the global_drive function updates the internal dictionaries."""
        rm = RydbergHamiltonian(atom_coordinates, wires)

        assert rm._global_drive is None

        global_drive(rm, rabi=1, detuning=2, phase=3)

        assert rm._global_drive == (1, 2, 3)

    def test_global_drive_updates_hamiltonian(self):
        """Test that the global_drive function returns an updated Hamiltonian that contains the
        driving interaction term."""
        rm = RydbergHamiltonian(coordinates=atom_coordinates, wires=wires)

        H = global_drive(rm, rabi=1, detuning=2, phase=3)

        assert H is not rm.hamiltonian
