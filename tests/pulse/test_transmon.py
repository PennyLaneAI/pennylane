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
Unit tests for the HardwareHamiltonian class.
"""
import warnings

# pylint: disable=too-few-public-methods
import numpy as np
import pytest

import pennylane as qml
from pennylane.pulse import HardwareHamiltonian, transmon_interaction, drive
from pennylane.pulse.hardware_hamiltonian import (
    HardwarePulse,
    AmplitudeAndPhase,
)
from pennylane.pulse.transmon import (
    a,
    ad,
)

from pennylane.wires import Wires

connections = [[0, 1], [1, 3], [2, 1], [4, 5]]
wires = [0, 1, 2, 3, 4, 5]
omega = 0.5 * np.arange(len(wires))
g = 0.1 * np.arange(len(connections))
delta = 0.3 * np.arange(len(wires))


class TestTransmonInteraction:
    """Unit tests for the ``transmon_interaction`` function."""

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
        ``transmon_interaction`` are correct."""
        Hd = transmon_interaction(
            connections=connections, omega=omega, g=g, delta=None, wires=wires, d=2
        )

        assert isinstance(Hd, HardwareHamiltonian)
        assert all(Hd.omega == omega)
        assert Hd.wires == Wires(wires)
        assert qml.math.allequal(Hd.connections, connections)

        num_combinations = len(wires) + len(connections)
        assert len(Hd.ops) == num_combinations
        assert Hd.pulses == []

    def test_wires_is_none(self):
        """Test that when wires is None the wires correspond to an increasing list of values with
        the same as the unique connections."""
        Hd = transmon_interaction(connections=connections, omega=0.3, g=0.3, delta=0.3)

        assert Hd.wires == Wires(np.unique(connections))

    def test_coeffs(self):
        """Test that the generated coefficients are correct."""
        Hd = qml.pulse.transmon_interaction(connections, omega, g, delta=delta, d=2)
        assert all(Hd.coeffs == np.concatenate([omega, g]))

    @pytest.mark.skip
    def test_coeffs_d(self):
        """Test that generated coefficients are correct for d>2"""
        Hd2 = qml.pulse.transmon_interaction(connections, omega, g, delta=delta, d=3)
        assert all(Hd2.coeffs == np.concatenate([omega, g, delta]))

    def test_d_neq_2_raises_error(self):
        """Test that setting d != 2 raises error"""
        with pytest.raises(NotImplementedError, match="Currently only supporting qubits."):
            _ = transmon_interaction(connections=connections, omega=0.1, g=0.2, d=3)

    def test_different_lengths_raises_error(self):
        """Test that using wires that are not fully contained by the connections raises an error"""
        with pytest.raises(ValueError, match="There are wires in connections"):
            _ = transmon_interaction(connections=connections, omega=0.1, g=0.2, wires=[0])

    def test_wrong_omega_len_raises_error(self):
        """Test that providing list of omegas with wrong length raises error"""
        with pytest.raises(ValueError, match="Number of qubit frequencies omega"):
            _ = transmon_interaction(
                connections=connections,
                omega=[0.1, 0.2],
                g=0.2,
            )

    def test_wrong_g_len_raises_error(self):
        """Test that providing list of g with wrong length raises error"""
        with pytest.raises(ValueError, match="Number of coupling terms"):
            _ = transmon_interaction(
                connections=connections,
                omega=0.1,
                g=[0.2, 0.2],
            )


# class TestTransmonDrive:
#     """Unit tests for the ``transmon_drive`` function."""

#     def test_attributes_and_number_of_terms(self):
#         """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
#         ``transmon_drive`` are correct."""

#         Hd = transmon_drive(amplitude=1, phase=2, wires=[1, 2])

#         assert isinstance(Hd, HardwareHamiltonian)
#         assert Hd.wires == Wires([1, 2])
#         assert Hd.connections is None
#         assert len(Hd.ops) == 2  # one for a and one of a^\dagger
#         assert Hd.pulses == [TransmonPulse(1, 2, [1, 2])]

#     # odd behavior when adding two drives/HardwareHamiltonians
#     # @pytest.mark.xfail
#     def test_multiple_local_drives(self):
#         """Test that adding multiple drive terms behaves as expected"""

#         def fa(p, t):
#             return np.sin(p * t)

#         def fb(p, t):
#             return np.cos(p * t)

#         H1 = transmon_drive(amplitude=fa, phase=1, wires=[0, 3])
#         H2 = transmon_drive(amplitude=1, phase=3, wires=[1, 2])
#         Hd = H1 + H2

#         ops_expected = [a(1) + a(2), ad(1) + ad(2), a(0) + a(3), ad(0) + ad(3)]
#         coeffs_expected = [
#             1.0 * qml.math.exp(1j * 3.0),
#             1.0 * qml.math.exp(-1j * 3.0),
#             AmplitudeAndPhase(1, fa, 1),
#             AmplitudeAndPhase(-1, fa, 1),
#         ]
#         H_expected = HardwareHamiltonian(coeffs_expected, ops_expected)

#         # structure of Hamiltonian is as expected
#         assert isinstance(Hd, HardwareHamiltonian)
#         assert Hd.wires == Wires([1, 2, 0, 3])  # TODO: Why is the order reversed?
#         assert Hd.connections is None
#         assert len(Hd.ops) == 4  # 2 terms for amplitude/phase and one detuning for each drive

#         # coefficients are correct
#         # Callable coefficients are shifted to the end of the list.
#         assert Hd.coeffs[:2] == coeffs_expected[:2]
#         assert isinstance(Hd.coeffs[2], AmplitudeAndPhase)
#         assert isinstance(Hd.coeffs[3], AmplitudeAndPhase)

#         # # pulses were added correctly
#         assert len(Hd.pulses) == 2
#         assert Hd.pulses == H1.pulses + H2.pulses

#         # # Hamiltonian is as expected
#         assert qml.equal(Hd([0.5, -0.5], t=5), H_expected([0.5, -0.5], t=5))
