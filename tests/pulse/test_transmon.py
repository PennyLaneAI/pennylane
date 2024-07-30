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
# pylint: disable=too-few-public-methods,redefined-outer-name,too-many-arguments
# pylint: disable=import-outside-toplevel
import numpy as np
import pytest

import pennylane as qml
from pennylane.pulse import HardwareHamiltonian, transmon_drive, transmon_interaction
from pennylane.pulse.hardware_hamiltonian import HardwarePulse
from pennylane.pulse.transmon import (
    AmplitudeAndPhaseAndFreq,
    TransmonSettings,
    _reorder_AmpPhaseFreq,
    a,
    ad,
    callable_freq_to_angular,
)
from pennylane.wires import Wires


def amp_phase_freq(amp, phase, freq, t, wire=0):
    """Compute the transmon drive term for given amplitude, phase and frequency."""
    return amp * (np.sin(phase + freq * t) * qml.PauliY(wire))


class TestTransmonDrive:
    """Tests for the transmon drive Hamiltonian."""

    def test_d_neq_2_raises_error(self):
        """Test that setting d != 2 raises error"""
        with pytest.raises(NotImplementedError, match="Currently only supports qubits"):
            _ = transmon_drive(0.5, 0.5, 0.5, [0], d=3)

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
        ``drive`` are correct."""

        Hd = transmon_drive(amplitude=1, phase=2, freq=3, wires=[1, 2])

        assert isinstance(Hd, HardwareHamiltonian)
        assert Hd.wires == Wires([1, 2])
        assert len(Hd.ops) == 1
        assert Hd.pulses == [HardwarePulse(1, 2, 3, wires=[1, 2])]

    @pytest.mark.parametrize("amp", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("phase", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("freq", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("t", 0.5 * np.arange(1, 3, dtype=float))
    def test_all_constant_parameters(self, amp, phase, freq, t):
        """Test that transmon drive with all constant parameters yields the expected Hamiltonian"""
        H = transmon_drive(amp, phase, freq, wires=[0])

        def expected(amp, phase, freq, t, wire=0):
            a = amp * (2 * np.pi)
            fr = freq * (2 * np.pi)
            return amp_phase_freq(a, phase, fr, t, wire)

        assert H.coeffs[0].func.__name__ == "no_callable"
        assert qml.math.allclose(qml.matrix(H([], t)), qml.matrix(expected(amp, phase, freq, t)))

    @pytest.mark.parametrize("amp", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("phase", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("freq", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("p", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("t", 0.5 * np.arange(1, 3, dtype=float))
    def test_amp_callable(self, amp, phase, freq, p, t):
        """Test callable amplitude works as expected"""
        H = transmon_drive(amp, phase, freq, wires=[0])

        def expected(amp, phase, freq, p, t, wire=0):
            a = amp(p[0], t) * (2 * np.pi)
            fr = freq * (2 * np.pi)
            return amp_phase_freq(a, phase, fr, t, wire)

        params = [p]

        assert H.coeffs[0].func.__name__ == "callable_amp"
        assert qml.math.allclose(
            qml.matrix(H(params, t)), qml.matrix(expected(amp, phase, freq, params, t))
        )

    @pytest.mark.parametrize("amp", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("phase", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("freq", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("p", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("t", 0.5 * np.arange(1, 3, dtype=float))
    def test_phase_callable(self, amp, phase, freq, p, t):
        """Test callable phase works as expected"""
        H = transmon_drive(amp, phase, freq, wires=[0])

        def expected(amp, phase, freq, p, t, wire=0):
            a = amp * (2 * np.pi)
            ph = phase(p[0], t)
            fr = freq * (2 * np.pi)
            return amp_phase_freq(a, ph, fr, t, wire)

        params = [p]

        assert H.coeffs[0].func.__name__ == "callable_phase"
        assert qml.math.allclose(
            qml.matrix(H(params, t)), qml.matrix(expected(amp, phase, freq, params, t))
        )

    @pytest.mark.parametrize("amp", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("phase", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("freq", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("p", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("t", 0.5 * np.arange(1, 3, dtype=float))
    def test_freq_callable(self, amp, phase, freq, p, t):
        """Test callable freq works as expected"""
        H = transmon_drive(amp, phase, freq, wires=[0])

        def expected(amp, phase, freq, p, t, wire=0):
            a = amp * (2 * np.pi)
            fr = freq(p[0], t) * (2 * np.pi)
            return amp_phase_freq(a, phase, fr, t, wire)

        params = [p]

        assert H.coeffs[0].func.__name__ == "callable_freq"
        assert qml.math.allclose(
            qml.matrix(H(params, t)), qml.matrix(expected(amp, phase, freq, params, t))
        )

    @pytest.mark.parametrize("amp", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("phase", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("freq", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("p1", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("p2", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("t", 0.5 * np.arange(1, 3, dtype=float))
    def test_amp_and_freq_callable(self, amp, phase, freq, p1, p2, t):
        """Test callable amplitude and freq works as expected"""
        H = transmon_drive(amp, phase, freq, wires=[0])

        def expected(amp, phase, freq, p1, p2, t, wire=0):
            a = amp(p1, t) * (2 * np.pi)
            fr = freq(p2, t) * (2 * np.pi)
            return amp_phase_freq(a, phase, fr, t, wire)

        params = [p1, p2]

        assert H.coeffs[0].func.__name__ == "callable_amp_and_freq"
        assert qml.math.allclose(
            qml.matrix(H(params, t)), qml.matrix(expected(amp, phase, freq, *params, t))
        )

    @pytest.mark.parametrize("amp", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("phase", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("freq", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("p1", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("p2", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("t", 0.5 * np.arange(1, 3, dtype=float))
    def test_amp_and_phase_callable(self, amp, phase, freq, p1, p2, t):
        """Test callable amplitude and phase works as expected"""
        H = transmon_drive(amp, phase, freq, wires=[0])

        def expected(amp, phase, freq, p1, p2, t, wire=0):
            a = amp(p1, t) * (2 * np.pi)
            ph = phase(p2, t)
            fr = freq * (2 * np.pi)
            return amp_phase_freq(a, ph, fr, t, wire)

        params = [p1, p2]

        assert H.coeffs[0].func.__name__ == "callable_amp_and_phase"
        assert qml.math.allclose(
            qml.matrix(H(params, t)), qml.matrix(expected(amp, phase, freq, *params, t))
        )

    @pytest.mark.parametrize("amp", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("phase", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("freq", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("p1", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("p2", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("t", 0.5 * np.arange(1, 3, dtype=float))
    def test_phase_and_freq_callable(self, amp, phase, freq, p1, p2, t):
        """Test callable phase and freq works as expected"""
        H = transmon_drive(amp, phase, freq, wires=[0])

        def expected(amp, phase, freq, p1, p2, t, wire=0):
            a = amp * (2 * np.pi)
            ph = phase(p1, t)
            fr = freq(p2, t) * (2 * np.pi)
            return amp_phase_freq(a, ph, fr, t, wire)

        params = [p1, p2]

        assert H.coeffs[0].func.__name__ == "callable_phase_and_freq"
        assert qml.math.allclose(
            qml.matrix(H(params, t)), qml.matrix(expected(amp, phase, freq, *params, t))
        )

    @pytest.mark.parametrize("amp", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("phase", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("freq", [lambda p, t: p * t, lambda p, t: np.sin(p * t)])
    @pytest.mark.parametrize("p0", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("p1", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("p2", 0.5 * np.arange(2, dtype=float))
    @pytest.mark.parametrize("t", 0.5 * np.arange(1, 3, dtype=float))
    def test_amplitude_phase_and_freq_callable(self, amp, phase, freq, p0, p1, p2, t):
        """Test callable amplitude, phase and freq works as expected"""
        H = transmon_drive(amp, phase, freq, wires=[0])

        def expected(params, t, wire=0):
            a = amp(params[0], t) * (2 * np.pi)
            ph = phase(params[1], t)
            fr = freq(params[2], t) * (2 * np.pi)
            return amp_phase_freq(a, ph, fr, t, wire)

        params = [p0, p1, p2]

        assert H.coeffs[0].func.__name__ == "callable_amp_and_phase_and_freq"
        assert qml.math.allclose(qml.matrix(H(params, t)), qml.matrix(expected(params, t)))

    def test_multiple_drives(self):
        """Test that the sum of two transmon drive Hamiltonians behaves correctly."""

        def amp(p, t):
            return np.sin(p * t)

        phase0 = 3.0
        phase1 = 5.0
        freq0 = 0.1
        freq1 = 0.5

        H1 = transmon_drive(amplitude=amp, phase=phase0, freq=freq0, wires=[0, 3])
        H2 = transmon_drive(amplitude=1, phase=phase1, freq=freq1, wires=[1, 2])
        Hd = H1 + H2

        ops_expected = [
            qml.dot([1.0, 1.0], [qml.PauliY(1), qml.PauliY(2)]),
            qml.dot([1.0, 1.0], [qml.PauliY(0), qml.PauliY(3)]),
        ]
        coeffs_expected = [
            AmplitudeAndPhaseAndFreq(np.sin, amp, phase0, freq0),
            AmplitudeAndPhaseAndFreq(np.sin, 1, phase1, freq1),
        ]
        H_expected = HardwareHamiltonian(
            coeffs_expected, ops_expected, reorder_fn=_reorder_AmpPhaseFreq
        )
        # structure of Hamiltonian is as expected
        assert isinstance(Hd, HardwareHamiltonian)
        assert Hd.wires == Wires([0, 3, 1, 2])
        assert Hd.settings is None
        assert len(Hd.ops) == 2  # 2 terms for amplitude/phase

        for coeff in Hd.coeffs:
            assert isinstance(coeff, AmplitudeAndPhaseAndFreq)
        assert Hd.coeffs[0].func.__name__ == "callable_amp"
        assert Hd.coeffs[1].func.__name__ == "no_callable"

        # pulses were added correctly
        assert Hd.pulses == [
            HardwarePulse(amp, phase0, freq0, wires=[0, 3]),
            HardwarePulse(1, phase1, freq1, wires=[1, 2]),
        ]
        # Hamiltonian is as expected
        assert qml.math.allclose(qml.matrix(Hd([0.5], t=6)), qml.matrix(H_expected([0.5], t=6)))

    def test_transmon_drive_with_regular_Parametrized_Hamiltonian(self):
        """Test call works as expected for regular parametrized Hamiltonian"""

        def f(p, t):
            return np.sin(p * t)

        H = transmon_drive(f, f, f, wires=[0])
        H += np.polyval * qml.PauliZ(0)

        t = 5.0
        params = [1.0, 2.0, 3.0, [0.5]]
        expected = amp_phase_freq(
            f(params[0], t) * (2 * np.pi), f(params[1], t), f(params[2], t) * (2 * np.pi), t, wire=0
        )
        expected += np.polyval(params[3], t) * qml.PauliZ(0)

        assert qml.math.allclose(qml.matrix(H(params, t)), qml.matrix(expected))


connections = [[0, 1], [1, 3], [2, 1], [4, 5]]
wires = [0, 1, 2, 3, 4, 5]
qubit_freq = 0.5 * np.arange(len(wires))
coupling = 0.1 * np.arange(len(connections))
anharmonicity = 0.3 * np.arange(len(wires))


class TestTransmonInteraction:
    """Unit tests for the ``transmon_interaction`` function."""

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
        ``transmon_interaction`` are correct."""
        Hd = transmon_interaction(
            connections=connections,
            qubit_freq=qubit_freq,
            coupling=coupling,
            anharmonicity=None,
            wires=wires,
            d=2,
        )
        settings = TransmonSettings(
            connections, qubit_freq, coupling, anharmonicity=[0.0] * len(wires)
        )

        assert isinstance(Hd, HardwareHamiltonian)
        assert Hd.settings == settings
        assert Hd.wires == Wires(wires)

        num_combinations = len(wires) + len(connections)
        assert len(Hd.ops) == num_combinations
        assert Hd.pulses == []

    def test_coeffs(self):
        """Test that the generated coefficients are correct."""
        Hd = qml.pulse.transmon_interaction(
            qubit_freq, connections, coupling, wires=wires, anharmonicity=anharmonicity, d=2
        )
        omega = 2 * np.pi * qubit_freq
        g = 2 * np.pi * coupling
        assert all(Hd.coeffs == np.concatenate([omega, g]))

    @pytest.mark.skip
    def test_coeffs_d(self):
        """Test that generated coefficients are correct for d>2"""
        Hd2 = qml.pulse.transmon_interaction(
            qubit_freq=qubit_freq,
            connections=connections,
            coupling=coupling,
            wires=wires,
            anharmonicity=anharmonicity,
            d=3,
        )
        assert all(Hd2.coeffs == np.concatenate([qubit_freq, coupling, anharmonicity]))

    def test_float_qubit_freq_with_explicit_wires(self):
        """Test that a single float qubit_freq with explicit wires yields the correct Hamiltonian"""
        wires = range(6)
        H = qml.pulse.transmon_interaction(
            qubit_freq=1.0, connections=connections, coupling=coupling, wires=wires
        )

        g = 2 * np.pi * coupling
        assert H.coeffs[:6] == [2 * np.pi] * 6
        assert all(H.coeffs[6:] == g)
        for o1, o2 in zip(H.ops[:6], [ad(i, 2) @ a(i, 2) for i in wires]):
            qml.assert_equal(o1, o2)

    def test_single_callable_qubit_freq_with_explicit_wires(self):
        """Test that a single callable qubit_freq with explicit wires yields the correct Hamiltonian"""
        wires0 = np.arange(10)
        coupling = 0.5

        H = qml.pulse.transmon_interaction(
            qubit_freq=np.polyval,
            connections=[[i, (i + 1) % 10] for i in wires0],
            coupling=coupling,
            wires=wires0,
        )

        g = 2 * np.pi * coupling
        omega = callable_freq_to_angular(np.polyval)

        assert qml.math.allclose(H.coeffs[:10], g)
        for coeff in H.coeffs[10:]:
            assert coeff([3, 4, 5], 6) == omega([3, 4, 5], 6)
        for o1, o2 in zip(H.ops[10:], [ad(i, 2) @ a(i, 2) for i in wires0]):
            qml.assert_equal(o1, o2)

    def test_d_neq_2_raises_error(self):
        """Test that setting d != 2 raises error"""
        with pytest.raises(NotImplementedError, match="Currently only supporting qubits."):
            _ = transmon_interaction(
                connections=connections, qubit_freq=[0.1], wires=[0], coupling=0.2, d=3
            )

    def test_wrong_g_len_raises_error(self):
        """Test that providing list of coupling with wrong length raises error"""
        with pytest.raises(ValueError, match="Number of coupling terms"):
            _ = transmon_interaction(
                connections=connections, qubit_freq=[0.1], coupling=[0.2, 0.2], wires=[0]
            )

    def test_qubit_freq_and_wires_dont_match(self):
        """Test that providing missmatching qubit_freq and wires raises error"""
        with pytest.raises(ValueError, match="Number of qubit frequencies"):
            _ = transmon_interaction(
                qubit_freq=[1, 2, 3], wires=[0, 1], connections=[], coupling=[]
            )

    def test_wires_and_connections_and_not_containing_each_other_raise_warning(
        self,
    ):
        """Test that when wires and connections to not contain each other, a warning is raised"""
        with pytest.warns(UserWarning, match="Caution, wires and connections do not match."):
            _ = qml.pulse.transmon_interaction(
                qubit_freq=0.5, connections=[[0, 1], [2, 3]], coupling=0.5, wires=[4, 5, 6]
            )

        with pytest.warns(UserWarning, match="Caution, wires and connections do not match."):
            _ = qml.pulse.transmon_interaction(
                qubit_freq=0.5, connections=[[0, 1], [2, 3]], coupling=0.5, wires=[0, 1, 2]
            )

        with pytest.warns(UserWarning, match="Caution, wires and connections do not match."):
            connections = [["a", "b"], ["a", "c"], ["d", "e"], ["e", "f"]]
            wires = ["a", "b", "c", "d", "e"]
            qubit_freq = 0.5 * np.arange(len(wires))
            coupling = 0.1 * np.arange(len(connections))

            H = qml.pulse.transmon_interaction(qubit_freq, connections, coupling, wires)
            assert H.wires == Wires(["a", "b", "c", "d", "e", "f"])

    def test_functional_form(self):
        """Test that the matrices of the transmon drive and a manually constructed
        operator with the same functional form are equal."""
        Hd = transmon_drive(1.0, 1.0, 1.0, wires=[0])
        expected = qml.s_prod(2 * np.pi * np.sin(2 * np.pi * 1.0 + 1), qml.PauliY(0))

        assert qml.math.allclose(qml.matrix(Hd([], 1.0)), qml.matrix(expected))


# For transmon settings test
connections0 = [[0, 1], [0, 2]]
qubit_freq0 = [1.0, 2.0, 3.0]
g0 = [0.5, 0.3]


connections1 = [[2, 3], [1, 4], [5, 4]]
qubit_freq1 = [4.0, 5.0, 6.0]
g1 = [0.1, 0.2, 0.3]


class TestTransmonSettings:
    """Unit tests for TransmonSettings dataclass"""

    def test_init(self):
        """Test the initialization of the ``TransmonSettings`` class."""
        settings = TransmonSettings(connections0, qubit_freq0, g0, [0.0] * len(qubit_freq0))
        assert settings.connections == connections0
        assert settings.qubit_freq == qubit_freq0
        assert settings.coupling == g0
        assert settings.anharmonicity == [0.0] * len(qubit_freq0)

    def test_equal(self):
        """Test the ``__eq__`` method of the ``TransmonSettings`` class."""
        settings0 = TransmonSettings(connections0, qubit_freq0, g0, [0.0] * len(qubit_freq0))
        settings1 = TransmonSettings(connections1, qubit_freq1, g1, [0.0] * len(qubit_freq1))
        settings2 = TransmonSettings(connections0, qubit_freq0, g0, [0.0] * len(qubit_freq0))
        assert settings0 != settings1
        assert settings1 != settings2
        assert settings0 == settings2

    def test_add_two_settings(
        self,
    ):
        """Test that two settings are correctly added"""
        settings0 = TransmonSettings(connections0, qubit_freq0, g0, [0.0] * len(qubit_freq0))
        settings1 = TransmonSettings(connections1, qubit_freq1, g1, [0.0] * len(qubit_freq1))
        settings = settings0 + settings1
        assert settings.connections == connections0 + connections1
        assert settings.qubit_freq == qubit_freq0 + qubit_freq1
        assert settings.coupling == g0 + g1

    def test_add_two_settings_with_one_anharmonicity_None(
        self,
    ):
        """Test that two settings are correctly added when one has non-trivial anharmonicity"""
        anharmonicity = [1.0] * len(qubit_freq0)
        settings0 = TransmonSettings(connections0, qubit_freq0, g0, anharmonicity=anharmonicity)
        settings1 = TransmonSettings(connections1, qubit_freq1, g1, [0.0] * len(qubit_freq1))

        settings01 = settings0 + settings1
        assert settings01.anharmonicity == anharmonicity + [0.0] * len(qubit_freq1)

        settings10 = settings1 + settings0
        assert settings10.anharmonicity == [0.0] * len(qubit_freq0) + anharmonicity


class TestIntegration:
    """Test integration with PennyLane."""

    @pytest.mark.jax
    def test_jitted_qnode(self):
        """Test that regular and jitted qnode yield same result"""
        import jax
        import jax.numpy as jnp

        Hi = transmon_interaction(qubit_freq, connections, coupling, wires=wires)

        def fa(p, t):
            return jnp.polyval(p, t)

        def fb(p, t):
            return p[0] * jnp.sin(p[1] * t)

        Hd = transmon_drive(amplitude=fa, phase=fb, freq=0.5, wires=[0])
        H = Hi + Hd

        dev = qml.device("default.qubit")

        ts = jnp.array([0.0, 3.0])
        H_obj = sum(qml.PauliZ(i) for i in range(2))

        @qml.qnode(dev, interface="jax")
        def qnode(params):
            qml.evolve(H)(params, ts)
            return qml.expval(H_obj)

        qnode_jit = jax.jit(qnode)

        params = (jnp.ones(5), jnp.array([1.0, jnp.pi]))

        res = qnode(params)
        res_jit = qnode_jit(params)

        assert isinstance(res, jax.Array)
        assert qml.math.isclose(res, res_jit)

    @pytest.mark.jax
    def test_jitted_qnode_multidrive(
        self,
    ):
        """Test that a transmon system with multiple drive terms can be
        executed within a jitted qnode."""
        import jax
        import jax.numpy as jnp

        Hi = transmon_interaction(qubit_freq, connections, coupling, wires=wires)

        def fa(p, t):
            return jnp.polyval(p, t)

        def fb(p, t):
            return p[0] * jnp.sin(p[1] * t)

        def fc(p, t):
            return p[0] * jnp.sin(t) + jnp.cos(p[1] * t)

        def fd(p, t):
            return p * jnp.cos(t)

        H1 = transmon_drive(amplitude=fa, phase=fb, freq=0.5, wires=wires)
        H2 = transmon_drive(amplitude=fc, phase=3 * jnp.pi, freq=0, wires=4)
        H3 = transmon_drive(amplitude=0, phase=fd, freq=3.0, wires=[3, 0])

        dev = qml.device("default.qubit", wires=wires)

        ts = jnp.array([0.0, 3.0])
        H_obj = sum(qml.PauliZ(i) for i in range(2))

        @qml.qnode(dev, interface="jax")
        def qnode(params):
            qml.evolve(Hi + H1 + H2 + H3)(params, ts)
            return qml.expval(H_obj)

        qnode_jit = jax.jit(qnode)

        params = (
            jnp.ones(5),
            jnp.array([1.0, jnp.pi]),
            jnp.array([jnp.pi / 2, 0.5]),
            jnp.array(-0.5),
        )
        res = qnode(params)
        res_jit = qnode_jit(params)

        assert isinstance(res, jax.Array)
        assert qml.math.isclose(res, res_jit)

    @pytest.mark.jax
    def test_jitted_qnode_all_coeffs_callable(self):
        """Test that a transmons system can be simulated within a
        jitted qnode when all coeffs are callable."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        H_drift = transmon_interaction(qubit_freq, connections, coupling, wires=wires)

        def fa(p, t):
            return jnp.polyval(p, t)

        def fb(p, t):
            return p[0] * jnp.sin(p[1] * t)

        def fc(p, t):
            return p[0] * jnp.sin(t) + jnp.cos(p[1] * t)

        H_drive = transmon_drive(amplitude=fa, phase=fb, freq=fc, wires=1)

        dev = qml.device("default.qubit", wires=wires)

        ts = jnp.array([0.0, 3.0])
        H_obj = sum(qml.PauliZ(i) for i in range(2))

        @qml.qnode(dev, interface="jax")
        def qnode(params):
            qml.evolve(H_drift + H_drive)(params, ts)
            return qml.expval(H_obj)

        qnode_jit = jax.jit(qnode)

        params = (jnp.ones(5), jnp.array([1.0, jnp.pi]), jnp.array([jnp.pi / 2, 0.5]))
        res = qnode(params)
        res_jit = qnode_jit(params)

        assert isinstance(res, jax.Array)
        assert qml.math.isclose(res, res_jit)
