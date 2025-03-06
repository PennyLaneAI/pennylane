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
# pylint: disable=too-few-public-methods, import-outside-toplevel, redefined-outer-name
# pylint: disable=reimported, wrong-import-position
import numpy as np
import pytest

import pennylane as qml
from pennylane.pulse import rydberg_drive, rydberg_interaction
from pennylane.pulse.hardware_hamiltonian import (
    AmplitudeAndPhase,
    HardwareHamiltonian,
    HardwarePulse,
)
from pennylane.pulse.rydberg import RydbergSettings
from pennylane.wires import Wires

atom_coordinates = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
wires = [1, 6, 0, 2, 4, 3]


class TestRydbergInteraction:
    """Unit tests for the ``rydberg_interaction`` function."""

    def test_queuing(self):
        """Test that the function does not queue any objects."""
        with qml.queuing.AnnotatedQueue() as q:
            rydberg_interaction(register=atom_coordinates, wires=wires, interaction_coeff=1)

        assert len(q) == 0

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
        # factor (2 * np.pi) to convert between angular and standard frequency
        Hd = rydberg_interaction(coords, interaction_coeff=1 / (2 * np.pi))
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
        # factor (2 * np.pi) to convert between angular and standard frequency
        H_res = rydberg_interaction(
            register=coords,
            wires=h_wires,
            interaction_coeff=1 / (2 * np.pi),
            max_distance=max_distance,
        )
        H_exp = rydberg_interaction(
            register=coords[:3], wires=h_wires[:3], interaction_coeff=1 / (2 * np.pi)
        )

        # Only 3 of the interactions will be non-negligible
        assert H_res.coeffs == [2.5**-6, 5**-6, 2.5**-6]
        qml.assert_equal(H_res([], t=5), H_exp([], t=5))


class TestRydbergDrive:
    """Unit tests for the ``rydberg_drive`` function"""

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
        ``rydberg_drive`` are correct."""

        Hd = rydberg_drive(amplitude=1, phase=2, detuning=3, wires=[1, 2])

        assert isinstance(Hd, HardwareHamiltonian)
        assert Hd.settings is None
        assert Hd.wires == Wires([1, 2])
        assert len(Hd.ops) == 3  # 2 amplitude/phase terms and one detuning term
        assert Hd.pulses == [HardwarePulse(1, 2, 3, [1, 2])]

    def test_multiple_local_drives(self):
        """Test that adding multiple drive terms behaves as expected"""

        # factors (2 * np.pi) to convert between angular and standard frequency
        def fa(p, t):
            return np.sin(p * t) / (2 * np.pi)

        def fb(p, t):
            return np.cos(p * t)

        H1 = rydberg_drive(amplitude=fa, phase=1, detuning=3, wires=[0, 3])
        H2 = rydberg_drive(amplitude=1 / (2 * np.pi), phase=3, detuning=fb, wires=[1, 2])
        Hd = H1 + H2

        ops_expected = [
            qml.Hamiltonian(
                [-0.5 * (2 * np.pi), -0.5 * (2 * np.pi), 0.5 * (2 * np.pi), 0.5 * (2 * np.pi)],
                [qml.Identity(0), qml.Identity(3), qml.PauliZ(0), qml.PauliZ(3)],
            ),
            qml.Hamiltonian([0.5, 0.5], [qml.PauliX(1), qml.PauliX(2)]),
            qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(1), qml.PauliY(2)]),
            qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliX(3)]),
            qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(0), qml.PauliY(3)]),
            qml.Hamiltonian(
                [-0.5 * (2 * np.pi), -0.5 * (2 * np.pi), 0.5 * (2 * np.pi), 0.5 * (2 * np.pi)],
                [qml.Identity(1), qml.Identity(2), qml.PauliZ(1), qml.PauliZ(2)],
            ),
        ]
        coeffs_expected = [
            3,
            np.cos(3),
            np.sin(3),
            AmplitudeAndPhase(np.cos, fa, 1),
            AmplitudeAndPhase(np.sin, fa, 1),
            fb,
        ]
        H_expected = HardwareHamiltonian(coeffs_expected, ops_expected)

        # structure of Hamiltonian is as expected
        assert isinstance(Hd, HardwareHamiltonian)
        assert Hd.wires == Wires([0, 3, 1, 2])
        assert Hd.settings is None
        assert len(Hd.ops) == 6  # 2 terms for amplitude/phase and one detuning for each drive

        # coefficients are correct
        # Callable coefficients are shifted to the end of the list.
        assert Hd.coeffs[0:3] == [3, np.cos(3), np.sin(3)]
        assert isinstance(Hd.coeffs[3], AmplitudeAndPhase)
        assert isinstance(Hd.coeffs[4], AmplitudeAndPhase)
        assert Hd.coeffs[5] is fb

        # pulses were added correctly
        assert len(Hd.pulses) == 2
        assert Hd.pulses == H1.pulses + H2.pulses

        # Hamiltonian is as expected
        actual = Hd([0.5, -0.5], t=5).simplify()
        expected = H_expected([0.5, -0.5], t=5).simplify()
        qml.assert_equal(actual, expected)

    def test_no_amplitude(self):
        """Test that when amplitude is not specified, the drive term is correctly defined."""

        # factors (2 * np.pi) to convert between angular and standard frequency
        def f(p, t):
            return np.cos(p * t) / (2 * np.pi)

        Hd = rydberg_drive(amplitude=0, phase=1, detuning=f, wires=[0, 3])

        ops_expected = [
            qml.Hamiltonian(
                [-0.5 * (2 * np.pi), -0.5 * (2 * np.pi), 0.5 * (2 * np.pi), 0.5 * (2 * np.pi)],
                [qml.Identity(0), qml.Identity(3), qml.PauliZ(0), qml.PauliZ(3)],
            )
        ]
        coeffs_expected = [f]
        H_expected = HardwareHamiltonian(coeffs_expected, ops_expected)

        actual = Hd([0.1], 10).simplify()
        expected = H_expected([0.1], 10).simplify()
        qml.assert_equal(actual, expected)
        assert isinstance(Hd, HardwareHamiltonian)
        assert Hd.wires == Wires([0, 3])
        assert Hd.settings is None
        assert len(Hd.coeffs) == 1
        assert Hd.coeffs[0] is f
        assert len(Hd.ops) == 1
        qml.assert_equal(Hd.ops[0], ops_expected[0])

    def test_no_detuning(self):
        """Test that when detuning not specified, the drive term is correctly defined."""

        def f(p, t):
            return np.cos(p * t)

        Hd = rydberg_drive(amplitude=f, phase=1, detuning=0, wires=[0, 3])

        ops_expected = [
            qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliX(3)]),
            qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(0), qml.PauliY(3)]),
        ]
        coeffs_expected = [
            AmplitudeAndPhase(np.cos, f, 1),
            AmplitudeAndPhase(np.sin, f, 1),
        ]
        H_expected = HardwareHamiltonian(coeffs_expected, ops_expected)

        qml.assert_equal(Hd([0.1], 10), H_expected([0.1], 10))
        assert isinstance(Hd, HardwareHamiltonian)
        assert Hd.wires == Wires([0, 3])
        assert Hd.settings is None
        assert all(isinstance(coeff, AmplitudeAndPhase) for coeff in Hd.coeffs)
        assert len(Hd.coeffs) == 2
        for op, op_expected in zip(Hd.ops, ops_expected):
            qml.assert_equal(op, op_expected)

    def test_no_amplitude_no_detuning(self):
        """Test that the correct error is raised if both amplitude and detuning are trivial."""
        with pytest.raises(ValueError, match="Expected non-zero value for at least one of either"):
            _ = rydberg_drive(0, np.pi, 0, wires=[0])


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

    # pylint: disable=unused-variable
    def test_raises_error_two_interaction_terms(
        self,
    ):
        """Raises error when attempting to add two non-trivial RydbergSettings"""
        settings0 = RydbergSettings(register0)
        settings1 = RydbergSettings(register1)
        with pytest.raises(ValueError, match="Cannot add two"):
            res = settings0 + settings1


class TestIntegration:
    """Integration tests for Rydberg system Hamiltonians."""

    @pytest.mark.jax
    def test_jitted_qnode(self):
        """Test that a Rydberg ensemble can be simulated within a jitted qnode."""
        import jax
        import jax.numpy as jnp

        Hd = rydberg_interaction(register=atom_coordinates, wires=wires)

        def fa(p, t):
            return jnp.polyval(p, t)

        def fb(p, t):
            return p[0] * jnp.sin(p[1] * t)

        Ht = rydberg_drive(amplitude=fa, phase=0, detuning=fb, wires=1)

        dev = qml.device("default.qubit", wires=wires)

        ts = jnp.array([0.0, 3.0])
        H_obj = sum(qml.PauliZ(i) for i in range(2))

        @qml.qnode(dev, interface="jax")
        def qnode(params):
            qml.evolve(Hd + Ht)(params, ts)
            return qml.expval(H_obj)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def qnode_jit(params):
            qml.evolve(Hd + Ht)(params, ts)
            return qml.expval(H_obj)

        params = (jnp.ones(5), jnp.array([1.0, jnp.pi]))
        res = qnode(params)
        res_jit = qnode_jit(params)

        assert isinstance(res, jax.Array)
        assert np.allclose(res, res_jit)

    @pytest.mark.jax
    def test_jitted_qnode_multidrive(self):
        """Test that a Rydberg ensemble with multiple drive terms can be
        executed within a jitted qnode."""
        import jax
        import jax.numpy as jnp

        Hd = rydberg_interaction(register=atom_coordinates, wires=wires)

        def fa(p, t):
            return jnp.polyval(p, t)

        def fb(p, t):
            return p[0] * jnp.sin(p[1] * t)

        def fc(p, t):
            return p[0] * jnp.sin(t) + jnp.cos(p[1] * t)

        def fd(p, t):
            return p * jnp.cos(t)

        H1 = rydberg_drive(amplitude=fa, phase=0, detuning=fb, wires=wires)
        H2 = rydberg_drive(amplitude=fc, phase=3 * jnp.pi, detuning=0, wires=4)
        H3 = rydberg_drive(amplitude=0, phase=0, detuning=fd, wires=[3, 0])

        dev = qml.device("default.qubit", wires=wires)

        ts = jnp.array([0.0, 3.0])
        H_obj = sum(qml.PauliZ(i) for i in range(2))

        @qml.qnode(dev, interface="jax")
        def qnode(params):
            qml.evolve(Hd + H1 + H2 + H3)(params, ts)
            return qml.expval(H_obj)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def qnode_jit(params):
            qml.evolve(Hd + H1 + H2 + H3)(params, ts)
            return qml.expval(H_obj)

        params = (
            jnp.ones(5),
            jnp.array([1.0, jnp.pi]),
            jnp.array([jnp.pi / 2, 0.5]),
            jnp.array(-0.5),
        )
        res = qnode(params)
        res_jit = qnode_jit(params)

        assert isinstance(res, jax.Array)
        assert np.allclose(res, res_jit)

    @pytest.mark.jax
    def test_jitted_qnode_all_coeffs_callable(self):
        """Test that a Rydberg ensemble can be simulated within a
        jitted qnode when all coeffs are callable."""
        import jax
        import jax.numpy as jnp

        H_drift = rydberg_interaction(register=atom_coordinates, wires=wires)

        def fa(p, t):
            return jnp.polyval(p, t)

        def fb(p, t):
            return p[0] * jnp.sin(p[1] * t)

        def fc(p, t):
            return p[0] * jnp.sin(t) + jnp.cos(p[1] * t)

        H_drive = rydberg_drive(amplitude=fa, phase=fb, detuning=fc, wires=1)

        dev = qml.device("default.qubit", wires=wires)

        ts = jnp.array([0.0, 3.0])
        H_obj = sum(qml.PauliZ(i) for i in range(2))

        @qml.qnode(dev, interface="jax")
        def qnode(params):
            qml.evolve(H_drift + H_drive)(params, ts)
            return qml.expval(H_obj)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def qnode_jit(params):
            qml.evolve(H_drift + H_drive)(params, ts)
            return qml.expval(H_obj)

        params = (jnp.ones(5), jnp.array([1.0, jnp.pi]), jnp.array([jnp.pi / 2, 0.5]))
        res = qnode(params)
        res_jit = qnode_jit(params)

        assert isinstance(res, jax.Array)
        assert np.allclose(res, res_jit)

    @pytest.mark.jax
    def test_pennylane_and_exact_solution_correspond(self):
        """Test that the results of PennyLane simulation match (within reason) the exact solution"""
        import jax
        import jax.numpy as jnp

        def exact(H, H_obj, t):
            psi0 = jnp.eye(2 ** len(H.wires))[0]
            U_exact = jax.scipy.linalg.expm(-1j * t * qml.matrix(H([], 1)))
            return (
                psi0 @ U_exact.conj().T @ qml.matrix(H_obj, wire_order=[0, 1, 2]) @ U_exact @ psi0
            )

        default_qubit = qml.device("default.qubit", wires=3)

        coordinates = [[0, 0], [0, 5], [5, 0]]

        H_i = qml.pulse.rydberg_interaction(coordinates)

        H = H_i + qml.pulse.rydberg_drive(3, 2, 4, [0, 1, 2])

        H_obj = qml.PauliZ(0)

        @jax.jit
        @qml.qnode(default_qubit, interface="jax")
        def circuit(t):
            qml.evolve(H)([], t)
            return qml.expval(H_obj)

        t = jnp.linspace(0.05, 1.55, 151)

        circuit_results = np.array([circuit(_t) for _t in t])
        exact_results = np.array([exact(H, H_obj, _t) for _t in t])

        # all results are approximately the same
        np.allclose(circuit_results, exact_results, atol=0.07)
