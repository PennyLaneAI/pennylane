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

import pennylane as qml
from pennylane.pulse import RydbergHamiltonian
from pennylane.pulse.rydberg_hamiltonian import RydbergPulses
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
        assert isinstance(rm.pulses, RydbergPulses)
        assert len(rm.pulses) == 0
        assert rm.wires == Wires([])
        assert rm.interaction_coeff == 862690 * np.pi

    def test_add(self):
        """Test that the __add__ dunder method works correctly."""
        rm1 = RydbergHamiltonian(
            coeffs=[1],
            observables=[qml.PauliX(0)],
            register=atom_coordinates,
            pulses=RydbergPulses([1], [2], [3], [4]),
        )
        rm2 = RydbergHamiltonian(
            coeffs=[2],
            observables=[qml.PauliY(1)],
            pulses=RydbergPulses([5], [6], [7], [8]),
        )

        sum_rm = rm1 + rm2
        assert isinstance(sum_rm, RydbergHamiltonian)
        assert qml.math.allequal(sum_rm.coeffs, [1, 2])
        assert all(
            qml.equal(op1, op2) for op1, op2 in zip(sum_rm.ops, [qml.PauliX(0), qml.PauliY(1)])
        )
        assert qml.math.allequal(sum_rm.register, atom_coordinates)
        assert sum_rm.pulses == RydbergPulses([1, 5], [2, 6], [3, 7], [4, 8])
        sum_rm2 = rm2 + rm1
        assert isinstance(sum_rm2, RydbergHamiltonian)
        assert qml.math.allequal(sum_rm2.coeffs, [2, 1])
        assert all(
            qml.equal(op1, op2) for op1, op2 in zip(sum_rm2.ops, [qml.PauliY(1), qml.PauliX(0)])
        )
        assert qml.math.allequal(sum_rm2.register, atom_coordinates)
        assert sum_rm2.pulses == RydbergPulses([5, 1], [6, 2], [7, 3], [8, 4])
