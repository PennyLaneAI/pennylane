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
Tests for everything related to rydberg system specific functionality.
"""
# pylint: disable=too-few-public-methods
import numpy as np
import pytest

import pennylane as qml
from pennylane.pulse import HardwareHamiltonian, rydberg_interaction

from pennylane.wires import Wires
from pennylane.pulse.rydberg import RydbergSettings

atom_coordinates = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
wires = [1, 6, 0, 2, 4, 3]


class TestRydbergInteraction:
    """Unit tests for the ``rydberg_interaction`` function."""

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
        ``rydberg_interaction`` are correct."""
        Hd = rydberg_interaction(register=atom_coordinates, wires=wires, interaction_coeff=1)
        settings = RydbergSettings(atom_coordinates, 1)

        assert isinstance(Hd, HardwareHamiltonian)
        assert Hd.wires == Wires(wires)
        N = len(wires)
        num_combinations = N * (N - 1) / 2  # number of terms on the rydberg_interaction hamiltonian
        assert len(Hd.ops) == num_combinations
        assert Hd.pulses == []
        assert Hd.settings == settings

    def test_wires_is_none(self):
        """Test that when wires is None the wires correspond to an increasing list of values with
        the same length as the atom coordinates."""
        Hd = rydberg_interaction(register=atom_coordinates)

        assert Hd.wires == Wires(list(range(len(atom_coordinates))))

    def test_coeffs(self):
        """Test that the generated coefficients are correct."""
        coords = [[0, 0], [0, 1], [1, 0]]
        Hd = rydberg_interaction(coords, interaction_coeff=1)
        assert Hd.coeffs == [1, 1, 1 / np.sqrt(2) ** 6]

    def test_different_lengths_raises_error(self):
        """Test that using different lengths for the wires and the register raises an error."""
        with pytest.raises(ValueError, match="The length of the wires and the register must match"):
            _ = rydberg_interaction(register=atom_coordinates, wires=[0])

    def test_max_distance(self):
        """Test that specifying a maximum distance affects the number of elements in the interaction term
        as expected."""
        # This threshold will remove interactions between atoms more than 5 micrometers away from each other
        max_distance = 5
        coords = [[0, 0], [2.5, 0], [5, 0], [6, 6]]
        h_wires = [1, 0, 2, 3]

        # Set interaction_coeff to one for easier comparison
        H_res = rydberg_interaction(
            register=coords, wires=h_wires, interaction_coeff=1, max_distance=max_distance
        )
        H_exp = rydberg_interaction(register=coords[:3], wires=h_wires[:3], interaction_coeff=1)

        # Only 3 of the interactions will be non-negligible
        assert H_res.coeffs == [2.5**-6, 5**-6, 2.5**-6]
        assert qml.equal(H_res([], t=5), H_exp([], t=5))


# For rydberg settings test
register0 = [[0.0, 1.0], [0.0, 2.0]]
register1 = [[2.0, 0.3], [1.0, 4.0], [0.5, 0.4]]


class TestRydbergSettings:
    """Unit tests for TransmonSettings dataclass"""

    def test_init(self):
        """Test the initialization of the ``RydbergSettings`` class."""
        settings = RydbergSettings(register0)
        assert settings.register == register0
        assert settings.interaction_coeff == 0.0

    def test_equal(self):
        """Test the ``__eq__`` method of the ``RydbergSettings`` class."""
        settings0 = RydbergSettings(register0)
        settings1 = RydbergSettings(register1, interaction_coeff=2.0)
        settings2 = RydbergSettings(register0, interaction_coeff=0.0)
        assert settings0 != settings1
        assert settings1 != settings2
        assert settings0 == settings2

    def test_add_two_settings(
        self,
    ):
        """Test that two RydbergSettings are correctly added"""

        settings0 = RydbergSettings(register0, interaction_coeff=2.0)
        settings1 = None

        settings01 = settings0 + settings1
        settings10 = settings1 + settings0
        assert settings01.register == register0
        assert settings01.interaction_coeff == 2.0
        assert settings10.register == register0
        assert settings10.interaction_coeff == 2.0

    def test_raises_error_two_interaction_terms(self,):
        """Raises error when attempting to add two non-trivial RydbergSettings"""
        settings0 = RydbergSettings(register0)
        settings1 = RydbergSettings(register1)
        with pytest.raises(
            ValueError, match="Cannot add two"
        ):
            settings0 + settings1