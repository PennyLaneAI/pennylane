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
# pylint: disable=too-few-public-methods,redefined-outer-name
import numpy as np
import pytest

import pennylane as qml
from pennylane.pulse import HardwareHamiltonian, transmon_interaction
from pennylane.pulse.transmon import TransmonSettings

from pennylane.wires import Wires

connections = [[0, 1], [1, 3], [2, 1], [4, 5]]
wires = [0, 1, 2, 3, 4, 5]
omega = 0.5 * np.arange(len(wires))
g = 0.1 * np.arange(len(connections))
anharmonicity = 0.3 * np.arange(len(wires))


class TestTransmonInteraction:
    """Unit tests for the ``transmon_interaction`` function."""

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
        ``transmon_interaction`` are correct."""
        Hd = transmon_interaction(
            connections=connections, omega=omega, g=g, anharmonicity=None, wires=wires, d=2
        )
        settings = TransmonSettings(connections, omega, g, anharmonicity=[0.0] * len(wires))

        assert isinstance(Hd, HardwareHamiltonian)
        assert Hd.settings == settings
        assert Hd.wires == Wires(wires)

        num_combinations = len(wires) + len(connections)
        assert len(Hd.ops) == num_combinations
        assert Hd.pulses == []

    def test_wires_is_none(self):
        """Test that when wires is None the wires correspond to an increasing list of values with
        the same as the unique connections."""
        Hd = transmon_interaction(connections=connections, omega=[0.3], g=0.3, anharmonicity=0.3)

        assert Hd.wires == Wires.all_wires(connections + [0])

    def test_coeffs(self):
        """Test that the generated coefficients are correct."""
        Hd = qml.pulse.transmon_interaction(omega, connections, g, anharmonicity=anharmonicity, d=2)
        assert all(Hd.coeffs == np.concatenate([omega, g]))

    @pytest.mark.skip
    def test_coeffs_d(self):
        """Test that generated coefficients are correct for d>2"""
        Hd2 = qml.pulse.transmon_interaction(
            connections, omega, g, anharmonicity=anharmonicity, d=3
        )
        assert all(Hd2.coeffs == np.concatenate([omega, g, anharmonicity]))

    def test_d_neq_2_raises_error(self):
        """Test that setting d != 2 raises error"""
        with pytest.raises(NotImplementedError, match="Currently only supporting qubits."):
            _ = transmon_interaction(connections=connections, omega=[0.1], g=0.2, d=3)

    def test_float_omega_with_no_explicit_wires(self):
        """Test that raises warning when omega is float and wires not explicit"""
        with pytest.raises(ValueError, match="Cannot instantiate wires automatically."):
            _ = transmon_interaction(connections=connections, omega=0.1, g=0.2)

    def test_wrong_g_len_raises_error(self):
        """Test that providing list of g with wrong length raises error"""
        with pytest.raises(ValueError, match="Number of coupling terms"):
            _ = transmon_interaction(
                connections=connections,
                omega=[0.1],
                g=[0.2, 0.2],
            )


# For transmon settings test
connections0 = [[0, 1], [0, 2]]
omega0 = [1.0, 2.0, 3.0]
g0 = [0.5, 0.3]


connections1 = [[2, 3], [1, 4], [5, 4]]
omega1 = [4.0, 5.0, 6.0]
g1 = [0.1, 0.2, 0.3]


class TestTransmonSettings:
    """Unit tests for TransmonSettings dataclass"""

    def test_init(self):
        """Test the initialization of the ``TransmonSettings`` class."""
        settings = TransmonSettings(connections0, omega0, g0, [0.0] * len(omega0))
        assert settings.connections == connections0
        assert settings.omega == omega0
        assert settings.g == g0
        assert settings.anharmonicity == [0.0] * len(omega0)

    def test_equal(self):
        """Test the ``__eq__`` method of the ``TransmonSettings`` class."""
        settings0 = TransmonSettings(connections0, omega0, g0, [0.0] * len(omega0))
        settings1 = TransmonSettings(connections1, omega1, g1, [0.0] * len(omega1))
        settings2 = TransmonSettings(connections0, omega0, g0, [0.0] * len(omega0))
        assert settings0 != settings1
        assert settings1 != settings2
        assert settings0 == settings2

    def test_add_two_settings(
        self,
    ):
        """Test that two settings are correctly added"""
        settings0 = TransmonSettings(connections0, omega0, g0, [0.0] * len(omega0))
        settings1 = TransmonSettings(connections1, omega1, g1, [0.0] * len(omega1))
        settings = settings0 + settings1
        assert settings.connections == connections0 + connections1
        assert settings.omega == omega0 + omega1
        assert settings.g == g0 + g1

    def test_add_two_settings_with_one_anharmonicity_None(
        self,
    ):
        """Test that two settings are correctly added when one has non-trivial anharmonicity"""
        anharmonicity = [1.0] * len(omega0)
        settings0 = TransmonSettings(connections0, omega0, g0, anharmonicity=anharmonicity)
        settings1 = TransmonSettings(connections1, omega1, g1, [0.0] * len(omega1))

        settings01 = settings0 + settings1
        assert settings01.anharmonicity == anharmonicity + [0.0] * len(omega1)

        settings10 = settings1 + settings0
        assert settings10.anharmonicity == [0.0] * len(omega0) + anharmonicity
