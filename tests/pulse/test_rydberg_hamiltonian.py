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
from pennylane.pulse import RydbergHamiltonian, rydberg_interaction
from pennylane.pulse.rydberg_hamiltonian import RydbergPulse
from pennylane.wires import Wires

atom_coordinates = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
wires = [1, 6, 9, 2, 4, 3]


class TestRydbergHamiltonian:
    """Unit tests for the properties of the RydbergHamiltonian class."""

    # pylint: disable=protected-access
    def test_initialization(self):
        """Test the RydbergHamiltonian class is initialized correctly."""
        rm = RydbergHamiltonian(coeffs=[], observables=[], register=atom_coordinates)

        assert qml.math.allequal(rm.register, atom_coordinates)
        assert rm.pulses == []
        assert rm.wires == Wires([])
        assert rm.interaction_coeff == 862690 * np.pi

    def test_add(self):
        """Test that the __add__ dunder method works correctly."""
        rm1 = RydbergHamiltonian(
            coeffs=[1],
            observables=[qml.PauliX(0)],
            register=atom_coordinates,
            pulses=[RydbergPulse(1, 2, 3, 4)],
        )
        rm2 = RydbergHamiltonian(
            coeffs=[2],
            observables=[qml.PauliY(1)],
            pulses=[RydbergPulse(5, 6, 7, 8)],
        )

        sum_rm = rm1 + rm2
        assert isinstance(sum_rm, RydbergHamiltonian)
        assert qml.math.allequal(sum_rm.coeffs, [1, 2])
        assert all(
            qml.equal(op1, op2) for op1, op2 in zip(sum_rm.ops, [qml.PauliX(0), qml.PauliY(1)])
        )
        assert qml.math.allequal(sum_rm.register, atom_coordinates)
        assert sum_rm.pulses == [RydbergPulse(1, 2, 3, 4), RydbergPulse(5, 6, 7, 8)]

    def test_radd(self):
        """Test that the __radd__ dunder method works correctly."""
        rm1 = RydbergHamiltonian(
            coeffs=[1],
            observables=[qml.PauliX(0)],
            register=atom_coordinates,
            pulses=[RydbergPulse(1, 2, 3, 4)],
        )
        rm2 = RydbergHamiltonian(
            coeffs=[2],
            observables=[qml.PauliY(1)],
            pulses=[RydbergPulse(5, 6, 7, 8)],
        )
        sum_rm2 = rm2 + rm1
        assert isinstance(sum_rm2, RydbergHamiltonian)
        assert qml.math.allequal(sum_rm2.coeffs, [2, 1])
        assert all(
            qml.equal(op1, op2) for op1, op2 in zip(sum_rm2.ops, [qml.PauliY(1), qml.PauliX(0)])
        )
        assert qml.math.allequal(sum_rm2.register, atom_coordinates)
        assert sum_rm2.pulses == [RydbergPulse(5, 6, 7, 8), RydbergPulse(1, 2, 3, 4)]

    def test_add_raises_error(self):
        """Test that an error is raised if two RydbergHamiltonians with registers are added."""
        rm1 = RydbergHamiltonian(
            coeffs=[1],
            observables=[qml.PauliX(0)],
            register=atom_coordinates,
            pulses=[RydbergPulse(1, 2, 3, 4)],
        )
        with pytest.raises(
            ValueError, match="We cannot add two Hamiltonians with an interaction term"
        ):
            _ = rm1 + rm1

    def test_add_raises_warning(self):
        """Test that an error is raised when adding two RydbergHamiltonians where one Hamiltonian
        contains pulses on wires that are not present in the register."""
        Hd = RydbergHamiltonian(coeffs=[1], observables=[qml.PauliX(0)], register=atom_coordinates)
        Ht = RydbergHamiltonian(
            coeffs=[2],
            observables=[qml.PauliY(1)],
            pulses=[RydbergPulse(1, 2, 3, 4)],
        )
        with pytest.warns(
            UserWarning,
            match="The wires of the laser fields are not present in the Rydberg ensemble",
        ):
            _ = Hd + Ht

        with pytest.warns(
            UserWarning,
            match="The wires of the laser fields are not present in the Rydberg ensemble",
        ):
            _ = Ht + Hd


class TestRydbergInteraction:
    """Unit tests for the ``rydberg_interaction`` function."""

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
        ``rydberg_interaction`` are correct."""
        Hd = rydberg_interaction(register=atom_coordinates, wires=wires, interaction_coeff=1)

        assert isinstance(Hd, RydbergHamiltonian)
        assert Hd.interaction_coeff == 1
        assert Hd.wires == Wires(wires)
        assert qml.math.allequal(Hd.register, atom_coordinates)
        N = len(wires)
        num_combinations = N * (N - 1) / 2  # number of terms on the rydberg_interaction hamiltonian
        assert len(Hd.ops) == num_combinations

    def test_wires_is_none(self):
        """Test that when wires is None the wires correspond to an increasing list of values with
        the same length as the atom coordinates."""
        Hd = rydberg_interaction(register=atom_coordinates)

        assert Hd.wires == Wires(list(range(len(atom_coordinates))))
