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
import warnings

# pylint: disable=too-few-public-methods
import numpy as np
import pytest

import pennylane as qml
from pennylane.pulse import RydbergHamiltonian, rydberg_drive, rydberg_interaction
from pennylane.pulse.rydberg_hamiltonian import (
    RydbergPulse,
    AmplitudeAndPhase,
    amplitude_and_phase,
    _rydberg_reorder_parameters,
)
from pennylane.wires import Wires

atom_coordinates = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
wires = [1, 6, 0, 2, 4, 3]


def f1(p, t):
    return p * np.sin(t) * (t - 1)


def f2(p, t):
    return p * np.cos(t**2)


param = [1.2, 2.3]


class TestRydbergHamiltonian:
    """Unit tests for the properties of the RydbergHamiltonian class."""

    # pylint: disable=protected-access
    def test_initialization(self):
        """Test the RydbergHamiltonian class is initialized correctly."""
        rm = RydbergHamiltonian(coeffs=[], observables=[], register=atom_coordinates)

        assert qml.math.allequal(rm.register, atom_coordinates)
        assert rm.pulses == []
        assert rm.wires == Wires([])
        assert rm.interaction_coeff == 862690

    def test_add_rydberg_hamiltonian(self):
        """Test that the __add__ dunder method works correctly."""
        rm1 = RydbergHamiltonian(
            coeffs=[1, 2],
            observables=[qml.PauliX(4), qml.PauliZ(8)],
            register=atom_coordinates,
            pulses=[RydbergPulse(1, 2, 3, [4, 8])],
        )
        rm2 = RydbergHamiltonian(
            coeffs=[2],
            observables=[qml.PauliY(8)],
            pulses=[RydbergPulse(5, 6, 7, 8)],
        )
        with warnings.catch_warnings():
            # We make sure that no warning is raised
            warnings.simplefilter("error")
            sum_rm = rm1 + rm2
        assert isinstance(sum_rm, RydbergHamiltonian)
        assert qml.math.allequal(sum_rm.coeffs, [1, 2, 2])
        assert all(
            qml.equal(op1, op2)
            for op1, op2 in zip(sum_rm.ops, [qml.PauliX(4), qml.PauliZ(8), qml.PauliY(8)])
        )
        assert qml.math.allequal(sum_rm.register, atom_coordinates)
        assert sum_rm.pulses == [
            RydbergPulse(1, 2, 3, [4, 8]),
            RydbergPulse(5, 6, 7, 8),
        ]

    def test_add_parametrized_hamiltonian(self):
        """Tests that adding a `RydbergHamiltonian` and `ParametrizedHamiltonian` works as
        expected."""
        coeffs = [2, 3]
        ops = [qml.PauliZ(0), qml.PauliX(2)]
        h_wires = [0, 2]

        rh = RydbergHamiltonian(
            coeffs=[coeffs[0]],
            observables=[ops[0]],
            pulses=[RydbergPulse(5, 6, 7, 8)],
        )
        ph = qml.pulse.ParametrizedHamiltonian(coeffs=[coeffs[1]], observables=[ops[1]])

        res1 = rh + ph
        res2 = ph + rh

        assert isinstance(res1, RydbergHamiltonian)
        assert res1.coeffs_fixed == coeffs
        assert res1.coeffs_parametrized == []
        assert all(qml.equal(op1, op2) for op1, op2 in zip(res1.ops_fixed, ops))
        assert res1.ops_parametrized == []
        assert res1.wires == qml.wires.Wires(h_wires)

        coeffs.reverse()
        ops.reverse()
        h_wires.reverse()

        assert isinstance(res2, RydbergHamiltonian)
        assert res2.coeffs_fixed == coeffs
        assert res2.coeffs_parametrized == []
        assert all(qml.equal(op1, op2) for op1, op2 in zip(res2.ops_fixed, ops))
        assert res2.ops_parametrized == []
        assert res2.wires == qml.wires.Wires(h_wires)

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
        coords = [[0, 0], [0, 5], [5, 0]]

        Hd = rydberg_interaction(register=coords, wires=[0, 1, 2])
        Ht = rydberg_drive(2, 3, 4, wires=3)

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

    def test_hamiltonian_callable_after_addition_right(self):
        """Tests that if a ParametrizedHamiltonian is added onto a
        RydbergHamiltonian with callable coefficients from the right, the
        resulting object is a RydbergHamiltonian that can be called without
        raising an error"""

        def amp(p, t):
            return p[0] * np.exp(-((t - p[1]) ** 2) / (2 * p[2] ** 2))

        def phase(p, t):
            return p * t

        def detuning(p, t):
            return np.cos(t) * p

        H_global = qml.pulse.rydberg_drive(amp, phase, detuning, wires=[0])
        H_global += qml.pulse.rydberg_drive(amp, phase, detuning, wires=[0])
        H_global += np.polyval * qml.PauliX(0)

        # start with RydbergHamiltonians, then add ParametrizedHamiltonian
        params = [np.array([1.2, 2.3, 3.4]), 4.5, 5.6]
        params += [np.array([1.2, 2.3, 3.4]), 4.5, 5.6]
        params += [np.ones(2)]

        assert isinstance(H_global, RydbergHamiltonian)
        H_global(params, 2)  # no error raised

    def test_hamiltonian_callable_after_addition_left(self):
        """Tests that if a ParametrizedHamiltonian is added onto a
        RydbergHamiltonian with callable coefficients from the left, the
        resulting object is a RydbergHamiltonian that can be called without
        raising an error"""

        def amp(p, t):
            return p[0] * np.exp(-((t - p[1]) ** 2) / (2 * p[2] ** 2))

        def phase(p, t):
            return p * t

        def detuning(p, t):
            return np.cos(t) * p

        # start with ParametrizedHamiltonian, add on RydbergHamiltonians
        H_global = np.polyval * qml.PauliX(0)
        H_global += qml.pulse.rydberg_drive(amp, phase, detuning, wires=[0])
        H_global += qml.pulse.rydberg_drive(amp, phase, detuning, wires=[0])

        params = [np.ones(2)]
        params += [np.array([1.2, 2.3, 3.4]), 4.5, 5.6]
        params += [np.array([1.2, 2.3, 3.4]), 4.5, 5.6]

        assert isinstance(H_global, RydbergHamiltonian)
        H_global(params, 2)  # no error raised


class TestInteractionWithOperators:
    """Test that the interaction between a ``RydbergHamiltonian`` and other operators work as
    expected."""

    ops_with_coeffs = (
        (qml.Hamiltonian([2], [qml.PauliZ(0)]), 2),
        (qml.Hamiltonian([1.7], [qml.PauliZ(0)]), 1.7),
        (3 * qml.PauliZ(0), 3),
        (qml.ops.SProd(3, qml.PauliZ(0)), 3),
    )
    ops = (
        qml.PauliX(2),
        qml.PauliX(2) @ qml.PauliX(3),
        qml.CNOT([0, 1]),
    )

    @pytest.mark.parametrize("H, coeff", ops_with_coeffs)
    def test_add_special_operators(self, H, coeff):
        """Test that a Hamiltonian and SProd can be added to a RydbergHamiltonian, and
        will be incorporated in the H_fixed term, with their coefficients included in H_coeffs_fixed.
        """
        R = rydberg_drive(amplitude=f1, phase=0, detuning=f2, wires=[0, 1])
        params = [1, 2]
        # Adding on the right
        new_pH = R + H
        assert isinstance(new_pH, RydbergHamiltonian)
        assert R.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(coeff, qml.PauliZ(0)))
        assert new_pH.coeffs_fixed[0] == coeff
        assert qml.math.allequal(new_pH(params, t=0.5).matrix(), qml.matrix(R(params, t=0.5) + H))
        # Adding on the left
        new_pH = H + R
        assert isinstance(new_pH, RydbergHamiltonian)
        assert R.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(coeff, qml.PauliZ(0)))
        assert new_pH.coeffs_fixed[0] == coeff
        assert qml.math.allequal(new_pH(params, t=0.5).matrix(), qml.matrix(R(params, t=0.5) + H))

    @pytest.mark.parametrize("op", ops)
    def test_add_other_operators(self, op):
        """Test that a Hamiltonian, SProd, Tensor or Operator can be added to a
        ParametrizedHamiltonian, and will be incorporated in the H_fixed term"""
        R = rydberg_drive(amplitude=f1, phase=0, detuning=f2, wires=[0, 1])
        params = [1, 2]

        # Adding on the right
        new_pH = R + op
        assert isinstance(new_pH, RydbergHamiltonian)
        assert R.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(1, op))
        new_pH(params, 2)  # confirm calling does not raise error

        # Adding on the left
        new_pH = op + R
        assert isinstance(new_pH, RydbergHamiltonian)
        assert R.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(1, op))
        new_pH(params, 2)  # confirm calling does not raise error

    def test_unknown_type_raises_error(self):
        R = rydberg_drive(amplitude=f1, phase=0, detuning=f2, wires=[0, 1])
        with pytest.raises(TypeError, match="unsupported operand type"):
            R += 3


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
        assert Hd.pulses == []

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


class TestRydbergDrive:
    """Unit tests for the ``rydberg_drive`` function."""

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
        ``rydberg_drive`` are correct."""

        Hd = rydberg_drive(amplitude=1, phase=2, detuning=3, wires=[1, 2])

        assert isinstance(Hd, RydbergHamiltonian)
        assert Hd.interaction_coeff == 862690
        assert Hd.wires == Wires([1, 2])
        assert Hd.register is None
        assert len(Hd.ops) == 3  # 2 amplitude/phase terms and one detuning term of the Hamiltonian
        assert Hd.pulses == [RydbergPulse(1, 2, 3, [1, 2])]

    def test_multiple_local_drives(self):
        """Test that adding multiple drive terms behaves as expected"""

        def fa(p, t):
            return np.sin(p * t)

        def fb(p, t):
            return np.cos(p * t)

        H1 = rydberg_drive(amplitude=fa, phase=1, detuning=3, wires=[0, 3])
        H2 = rydberg_drive(amplitude=1, phase=3, detuning=fb, wires=[1, 2])
        Hd = H1 + H2

        ops_expected = [
            qml.Hamiltonian([-1, -1], [qml.PauliZ(0), qml.PauliZ(3)]),
            qml.Hamiltonian([0.5, 0.5], [qml.PauliX(1), qml.PauliX(2)]),
            qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(1), qml.PauliY(2)]),
            qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliX(3)]),
            qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(0), qml.PauliY(3)]),
            qml.Hamiltonian([-1, -1], [qml.PauliZ(1), qml.PauliZ(2)]),
        ]
        coeffs_expected = [
            3,
            np.cos(3),
            np.sin(3),
            AmplitudeAndPhase(np.cos, fa, 1),
            AmplitudeAndPhase(np.sin, fa, 1),
            fb,
        ]
        H_expected = RydbergHamiltonian(coeffs_expected, ops_expected)

        # structure of Hamiltonian is as expected
        assert isinstance(Hd, RydbergHamiltonian)
        assert Hd.interaction_coeff == 862690
        assert Hd.wires == Wires([0, 3, 1, 2])
        assert Hd.register is None
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
        assert qml.equal(Hd([0.5, -0.5], t=5), H_expected([0.5, -0.5], t=5))

    def test_no_amplitude(self):
        """Test that when amplitude is not specified, the drive term is correctly defined."""

        def f(p, t):
            return np.cos(p * t)

        Hd = rydberg_drive(amplitude=0, phase=1, detuning=f, wires=[0, 3])

        ops_expected = [qml.Hamiltonian([-1, -1], [qml.PauliZ(0), qml.PauliZ(3)])]
        coeffs_expected = [f]
        H_expected = RydbergHamiltonian(coeffs_expected, ops_expected)

        assert qml.equal(Hd([0.1], 10), H_expected([0.1], 10))
        assert isinstance(Hd, RydbergHamiltonian)
        assert Hd.interaction_coeff == 862690
        assert Hd.wires == Wires([0, 3])
        assert Hd.register is None
        assert len(Hd.coeffs) == 1
        assert Hd.coeffs[0] is f
        assert len(Hd.ops) == 1
        assert qml.equal(Hd.ops[0], ops_expected[0])

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
        H_expected = RydbergHamiltonian(coeffs_expected, ops_expected)

        assert qml.equal(Hd([0.1], 10), H_expected([0.1], 10))
        assert isinstance(Hd, RydbergHamiltonian)
        assert Hd.interaction_coeff == 862690
        assert Hd.wires == Wires([0, 3])
        assert Hd.register is None
        assert all(isinstance(coeff, AmplitudeAndPhase) for coeff in Hd.coeffs)
        assert len(Hd.coeffs) == 2
        assert all(qml.equal(op, op_expected) for op, op_expected in zip(Hd.ops, ops_expected))

    def test_no_amplitude_no_detuning(self):
        """Test that the correct error is raised if both amplitude and detuning are trivial."""
        with pytest.raises(ValueError, match="Expected non-zero value for at least one of either"):
            _ = rydberg_drive(0, np.pi, 0, wires=[0])


def callable_amp(p, t):
    return np.polyval(p, t)


def callable_phase(p, t):
    return p[0] * np.sin(p[1] * t)


def sine_func(p, t):
    return np.sin(p * t)


def cosine_fun(p, t):
    return np.cos(p * t)


class TestAmplitudeAndPhase:
    """Test the AmplitudeAndPhase class that provides callable
    phase/amplitude combinations"""

    def test_amplitude_and_phase_no_callables(self):
        """Test that when calling amplitude_and_phase, if neither are callable,
        a float is returned instead of an AmplitudeAndPhase object"""
        f = amplitude_and_phase(np.sin, 3, 4)
        expected_result = 3 * np.sin(4)

        assert isinstance(f, float)
        assert f == expected_result

    def test_amplitude_and_phase_callable_phase(self):
        """Test that when calling amplitude_and_phase, if only phase is callable,
        an AmplitudeAndPhase object with callable phase and fixed amplitude is
        correctly created"""
        f = amplitude_and_phase(np.sin, 2.7, callable_phase)

        # attributes are correct
        assert isinstance(f, AmplitudeAndPhase)
        assert f.amp_is_callable is False
        assert f.phase_is_callable is True
        assert f.func.__name__ == "callable_phase"

        # calling yields expected result
        expected_result = 2.7 * np.sin(callable_phase([1.3, 2.5], 2))
        assert f([1.3, 2.5], 2) == expected_result

    def test_amplitude_and_phase_callable_amplitude(self):
        """Test that when calling amplitude_and_phase, if only amplitude is callable,
        an AmplitudeAndPhase object with callable amplitude and fixed phase is
        correctly created"""
        f = amplitude_and_phase(np.sin, callable_amp, 0.7)

        # attributes are correct
        assert isinstance(f, AmplitudeAndPhase)
        assert f.amp_is_callable is True
        assert f.phase_is_callable is False
        assert f.func.__name__ == "callable_amp"

        # calling yields expected result
        expected_result = callable_amp([1.7], 2) * np.sin(0.7)
        assert f([1.7], 2) == expected_result

    def test_amplitude_and_phase_both_callable(self):
        """Test that when calling amplitude_and_phase, if both are callable,
        an AmplitudeAndPhase object with callable amplitude and phase is
        correctly created"""
        f = amplitude_and_phase(np.sin, callable_amp, callable_phase)

        # attributes are correct
        assert isinstance(f, AmplitudeAndPhase)
        assert f.amp_is_callable is True
        assert f.phase_is_callable is True
        assert f.func.__name__ == "callable_amp_and_phase"

        # calling yields expected result
        expected_result = callable_amp([1.7], 2) * np.sin(callable_phase([1.3, 2.5], 2))
        assert f([[1.7], [1.3, 2.5]], 2) == expected_result

    def test_callable_phase_and_amplitude_hamiltonian(self):
        """Test that using callable amplitude and phase in rydberg_drive
        creates AmplitudeAndPhase callables, and the resulting Hamiltonian
        can be called successfully"""

        detuning = 2

        Hd = rydberg_drive(sine_func, cosine_fun, detuning, wires=[0, 1])

        assert len(Hd.coeffs) == 3
        assert isinstance(Hd.coeffs[1], AmplitudeAndPhase)
        assert isinstance(Hd.coeffs[2], AmplitudeAndPhase)
        t = 1.7

        evaluated_H = Hd([3.4, 5.6], t)

        expected_H_fixed = qml.s_prod(
            detuning, qml.Hamiltonian([-1, -1], [qml.PauliZ(0), qml.PauliZ(1)])
        )

        c1 = np.sin(3.4 * t) * np.cos(np.cos(5.6 * t))
        c2 = np.sin(3.4 * t) * np.sin(np.cos(5.6 * t))
        expected_H_parametrized = qml.sum(
            qml.s_prod(c1, qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliX(1)])),
            qml.s_prod(c2, qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(0), qml.PauliY(1)])),
        )

        assert qml.equal(evaluated_H[0], expected_H_fixed)
        assert qml.equal(evaluated_H[1], expected_H_parametrized)

    def test_callable_phase_hamiltonian(self):
        """Test that using callable phase in rydberg_drive creates AmplitudeAndPhase
        callables, and the resulting Hamiltonian can be called"""

        detuning = 2

        Hd = rydberg_drive(7.2, sine_func, detuning, wires=[0, 1])

        assert len(Hd.coeffs) == 3
        assert isinstance(Hd.coeffs[1], AmplitudeAndPhase)
        assert isinstance(Hd.coeffs[2], AmplitudeAndPhase)
        t = 1.7

        evaluated_H = Hd([5.6], t)

        expected_H_fixed = qml.s_prod(
            detuning, qml.Hamiltonian([-1, -1], [qml.PauliZ(0), qml.PauliZ(1)])
        )

        c1 = 7.2 * np.cos(np.sin(5.6 * t))
        c2 = 7.2 * np.sin(np.sin(5.6 * t))
        expected_H_parametrized = qml.sum(
            qml.s_prod(c1, qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliX(1)])),
            qml.s_prod(c2, qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(0), qml.PauliY(1)])),
        )

        assert qml.equal(evaluated_H[0], expected_H_fixed)
        assert qml.equal(evaluated_H[1], expected_H_parametrized)

    def test_callable_amplitude_hamiltonian(self):
        """Test that using callable amplitude in rydberg_drive creates AmplitudeAndPhase
        callables, and the resulting Hamiltonian can be called"""

        detuning = 2

        Hd = rydberg_drive(sine_func, 4.3, detuning, wires=[0, 1])

        assert len(Hd.coeffs) == 3
        assert isinstance(Hd.coeffs[1], AmplitudeAndPhase)
        assert isinstance(Hd.coeffs[2], AmplitudeAndPhase)
        t = 1.7

        evaluated_H = Hd([3.4], t)

        expected_H_fixed = qml.s_prod(
            detuning, qml.Hamiltonian([-1, -1], [qml.PauliZ(0), qml.PauliZ(1)])
        )

        c1 = np.sin(3.4 * t) * np.cos(4.3)
        c2 = np.sin(3.4 * t) * np.sin(4.3)
        expected_H_parametrized = qml.sum(
            qml.s_prod(c1, qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliX(1)])),
            qml.s_prod(c2, qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(0), qml.PauliY(1)])),
        )

        assert qml.equal(evaluated_H[0], expected_H_fixed)
        assert qml.equal(evaluated_H[1], expected_H_parametrized)

    COEFFS_AND_PARAMS = [
        (
            [AmplitudeAndPhase(np.cos, f1, f2), AmplitudeAndPhase(np.sin, f1, f2), f2],
            [1, 2, 3],
            [[1, 2], [1, 2], 3],
        ),
        (
            [AmplitudeAndPhase(np.cos, f1, 6.3), AmplitudeAndPhase(np.sin, f1, 6.3), f2],
            [1, 3],
            [1, 1, 3],
        ),
        (
            [AmplitudeAndPhase(np.cos, 6.3, f1), AmplitudeAndPhase(np.sin, 6.3, f1), f2],
            [1, 3],
            [1, 1, 3],
        ),
        (
            [f1, AmplitudeAndPhase(np.cos, f1, 5.7), AmplitudeAndPhase(np.sin, f1, 5.7), f2],
            [1, 2, 3],
            [1, 2, 2, 3],
        ),
        (
            [f1, AmplitudeAndPhase(np.cos, f1, f1), AmplitudeAndPhase(np.sin, f1, f1), f2],
            [2, [3, 5], 8, [13, 21]],
            [2, [[3, 5], 8], [[3, 5], 8], [13, 21]],
        ),
        ([f1, f2, f1, f2], [1, 2, 3, 4], [1, 2, 3, 4]),
    ]

    @pytest.mark.parametrize("coeffs, params, expected_output", COEFFS_AND_PARAMS)
    def test_rydberg_reorder_parameters_all(self, coeffs, params, expected_output):
        """Tests that the function organizing the parameters to pass to the
        RydbergHamiltonian works as expected when AmplitudeAndPhase callables
        are included"""

        assert _rydberg_reorder_parameters(params, coeffs) == expected_output


class TestRydbergPulse:
    """Unit tests for the ``RydbergPulse`` class."""

    def test_init(self):
        """Test the initialization of the ``RydbergPulse`` class."""
        p = RydbergPulse(amplitude=4, detuning=9, phase=8, wires=[0, 4, 7])
        assert p.amplitude == 4
        assert p.phase == 8
        assert p.detuning == 9
        assert p.wires == Wires([0, 4, 7])

    def test_equal(self):
        """Test the ``__eq__`` method of the ``RydbergPulse`` class."""
        p1 = RydbergPulse(1, 2, 3, [0, 1])
        p2 = RydbergPulse(1, 2, 3, 0)
        p3 = RydbergPulse(1, 2, 3, [0, 1])
        assert p1 != p2
        assert p2 != p3
        assert p1 == p3


class TestIntegration:
    """Integration tests for the ``RydbergHamiltonian`` class."""

    @pytest.mark.jax
    def test_jitted_qnode(self):
        """Test that a ``RydbergHamiltonian`` class can be executed within a jitted qnode."""
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
        assert res == res_jit

    @pytest.mark.jax
    def test_jitted_qnode_multidrive(self):
        """Test that a ``RydbergHamiltonian`` class with multiple drive terms can be
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
        assert res == res_jit

    @pytest.mark.jax
    def test_jitted_qnode_all_coeffs_callable(self):
        """Test that a ``RydbergHamiltonian`` class can be executed within a
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
        assert res == res_jit
