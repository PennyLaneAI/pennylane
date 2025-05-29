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
# pylint: disable=too-few-public-methods, import-outside-toplevel
import numpy as np
import pytest

import pennylane as qml
from pennylane.pulse import HardwareHamiltonian, drive, rydberg_interaction
from pennylane.pulse.hardware_hamiltonian import (
    AmplitudeAndPhase,
    HardwarePulse,
    _reorder_parameters,
    amplitude_and_phase,
)
from pennylane.pulse.rydberg import RydbergSettings
from pennylane.pulse.transmon import TransmonSettings
from pennylane.wires import Wires

atom_coordinates = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
wires = [1, 6, 0, 2, 4, 3]


def f1(p, t):
    """Compute the function p * sin(t) * (t - 1)."""
    return p * np.sin(t) * (t - 1)


def f2(p, t):
    """Compute the function p * cos(t**2)."""
    return p * np.cos(t**2)


param = [1.2, 2.3]


connections = [[0, 1], [1, 3], [2, 1], [4, 5]]
wires = [0, 1, 2, 3, 4, 5]
omega = 0.5 * np.arange(len(wires))
g = 0.1 * np.arange(len(connections))
anharmonicity = 0.3 * np.arange(len(wires))
transmon_settings = TransmonSettings(connections, omega, g, anharmonicity=None)

atom_coordinates = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
rydberg_settings = RydbergSettings(atom_coordinates, 1)


class TestHardwareHamiltonian:
    """Unit tests for the properties of the HardwareHamiltonian class."""

    # pylint: disable=protected-access, comparison-with-callable
    def test_initialization(self):
        """Test the HardwareHamiltonian class is initialized correctly."""
        rm = HardwareHamiltonian(coeffs=[], observables=[])

        assert rm.pulses == []
        assert rm.wires == Wires([])
        assert rm.settings is None
        assert rm.reorder_fn == _reorder_parameters

    def test_two_different_reorder_fns_raises_error(self):
        """Test that adding two HardwareHamiltonians with different reordering functions
        raises an error."""
        H1 = HardwareHamiltonian(coeffs=[], observables=[])
        H2 = HardwareHamiltonian(coeffs=[], observables=[], reorder_fn=lambda _, y: y[:2])

        with pytest.raises(
            ValueError, match="Cannot add two HardwareHamiltonians with different reorder"
        ):
            _ = H1 + H2

    @pytest.mark.parametrize(
        "settings",
        [
            None,
            transmon_settings,
            rydberg_settings,
        ],
    )
    def test_add_hardware_hamiltonian(self, settings):
        """Test that the __add__ dunder method works correctly."""
        rm1 = HardwareHamiltonian(
            coeffs=[1, 2],
            observables=[qml.PauliX(4), qml.PauliZ(8)],
            pulses=[HardwarePulse(1, 2, 3, [4, 8])],
            settings=settings,
            reorder_fn=_reorder_parameters,
        )
        rm2 = HardwareHamiltonian(
            coeffs=[2],
            observables=[qml.PauliY(8)],
            pulses=[HardwarePulse(5, 6, 7, 8)],
            reorder_fn=_reorder_parameters,
        )

        sum_rm = rm1 + rm2

        assert isinstance(sum_rm, HardwareHamiltonian)
        assert qml.math.allequal(sum_rm.coeffs, [1, 2, 2])
        for op1, op2 in zip(sum_rm.ops, [qml.PauliX(4), qml.PauliZ(8), qml.PauliY(8)]):
            qml.assert_equal(op1, op2)
        assert sum_rm.pulses == [
            HardwarePulse(1, 2, 3, [4, 8]),
            HardwarePulse(5, 6, 7, 8),
        ]
        assert sum_rm.settings == settings

    def test__repr__(self):
        """Test repr method returns expected string"""
        test_example = HardwareHamiltonian(
            coeffs=[2],
            observables=[qml.PauliY(8)],
            pulses=[HardwarePulse(5, 6, 7, 8)],
            reorder_fn=_reorder_parameters,
        )
        str = repr(test_example)
        assert str == "HardwareHamiltonian: terms=1"

    def test_add_parametrized_hamiltonian(self):
        """Tests that adding a `HardwareHamiltonian` and `ParametrizedHamiltonian` works as
        expected."""
        coeffs = [2, 3]
        ops = [qml.PauliZ(0), qml.PauliX(2)]
        h_wires = [0, 2]

        rh = HardwareHamiltonian(
            coeffs=[coeffs[0]],
            observables=[ops[0]],
            pulses=[HardwarePulse(5, 6, 7, 8)],
        )
        ph = qml.pulse.ParametrizedHamiltonian(coeffs=[coeffs[1]], observables=[ops[1]])

        res1 = rh + ph
        res2 = ph + rh

        assert isinstance(res1, HardwareHamiltonian)
        assert res1.coeffs_fixed == coeffs
        assert res1.coeffs_parametrized == []
        for op1, op2 in zip(res1.ops_fixed, ops):
            qml.assert_equal(op1, op2)
        assert res1.ops_parametrized == []
        assert res1.wires == qml.wires.Wires(h_wires)

        coeffs.reverse()
        ops.reverse()
        h_wires.reverse()

        assert isinstance(res2, HardwareHamiltonian)
        assert res2.coeffs_fixed == coeffs
        assert res2.coeffs_parametrized == []
        for op1, op2 in zip(res2.ops_fixed, ops):
            qml.assert_equal(op1, op2)
        assert res2.ops_parametrized == []
        assert res2.wires == qml.wires.Wires(h_wires)

    @pytest.mark.parametrize("scalars", [(0.2, 1), (0.9, 0.2), (1, 5), (3, 0.5)])
    def test_add_scalar(self, scalars):
        """Test that adding a scalar/number to a `HardwareHamiltonian` works as expected."""
        coeffs = [2, 3]
        ops = [qml.PauliZ(2), qml.PauliX(0)]

        H = HardwareHamiltonian(
            coeffs=coeffs,
            observables=ops,
            pulses=[HardwarePulse(qml.pulse.constant, 6, 7, 8)],
        )
        orig_matrix = qml.matrix(H([0.3], 0.1))
        H1 = H + scalars[0]
        assert len(H1.ops) == len(H1.coeffs) == 3
        qml.assert_equal(H1.ops[-1], qml.Identity(2))
        assert qml.math.allclose(qml.matrix(H1([0.3], 0.1)), orig_matrix + np.eye(4) * scalars[0])
        H2 = scalars[1] + H1
        assert len(H2.ops) == len(H2.coeffs) == 4
        qml.assert_equal(H2.ops[-2], qml.Identity(2))
        qml.assert_equal(H2.ops[-1], qml.Identity(2))
        assert qml.math.allclose(
            qml.matrix(H2([0.3], 0.1)), orig_matrix + np.eye(4) * (scalars[0] + scalars[1])
        )
        H += scalars[0]
        assert len(H.ops) == len(H.coeffs) == 3
        qml.assert_equal(H.ops[-1], qml.Identity(2))
        assert qml.math.allclose(qml.matrix(H([0.3], 0.1)), orig_matrix + np.eye(4) * scalars[0])

    def test_add_zero(self):
        """Test that adding an int or a float that is zero to a `HardwareHamiltonian`
        returns an unchanged copy."""
        coeffs = [2, 3]
        ops = [qml.PauliZ(2), qml.PauliX(0)]

        H = HardwareHamiltonian(
            coeffs=coeffs,
            observables=ops,
            pulses=[HardwarePulse(qml.pulse.constant, 6, 7, 8)],
        )
        orig_matrix = qml.matrix(H([0.3], 0.1))
        H1 = H + 0
        assert len(H1.ops) == len(H1.coeffs) == 2
        assert qml.math.allclose(qml.matrix(H1([0.3], 0.1)), orig_matrix)
        assert H1 is not H
        H2 = 0.0 + H
        assert len(H2.ops) == len(H2.coeffs) == 2
        assert qml.math.allclose(qml.matrix(H2([0.3], 0.1)), orig_matrix)
        assert H2 is not H

    @pytest.mark.xfail
    def test_add_raises_warning(self):
        """Test that an error is raised when adding two HardwareHamiltonians where one Hamiltonian
        contains pulses on wires that are not present in the register."""
        coords = [[0, 0], [0, 5], [5, 0]]

        Hd = rydberg_interaction(register=coords, wires=[0, 1, 2])
        Ht = drive(2, 3, wires=3)

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
        HardwareHamiltonian with callable coefficients from the right, the
        resulting object is a HardwareHamiltonian that can be called without
        raising an error"""

        def amp(p, t):
            return p[0] * np.exp(-((t - p[1]) ** 2) / (2 * p[2] ** 2))

        def phase(p, t):
            return p * t

        H_global = qml.pulse.drive(amp, phase, wires=[0])
        H_global += qml.pulse.drive(amp, phase, wires=[0])
        H_global += np.polyval * qml.PauliX(0)

        # start with HardwareHamiltonians, then add ParametrizedHamiltonian
        params = [np.array([1.2, 2.3, 3.4]), 4.5]
        params += [np.array([1.2, 2.3, 3.4]), 4.5]
        params += [np.ones(2)]

        assert isinstance(H_global, HardwareHamiltonian)
        H_global(params, 2)  # no error raised

    def test_hamiltonian_callable_after_addition_left(self):
        """Tests that if a ParametrizedHamiltonian is added onto a
        HardwareHamiltonian with callable coefficients from the left, the
        resulting object is a HardwareHamiltonian that can be called without
        raising an error"""

        def amp(p, t):
            return p[0] * np.exp(-((t - p[1]) ** 2) / (2 * p[2] ** 2))

        def phase(p, t):
            return p * t

        # start with ParametrizedHamiltonian, add on HardwareHamiltonians
        H_global = np.polyval * qml.PauliX(0)
        H_global += qml.pulse.drive(amp, phase, wires=[0])
        H_global += qml.pulse.drive(amp, phase, wires=[0])

        params = [np.ones(2)]
        params += [np.array([1.2, 2.3, 3.4]), 4.5]
        params += [np.array([1.2, 2.3, 3.4]), 4.5]

        assert isinstance(H_global, HardwareHamiltonian)
        H_global(params, 2)  # no error raised


# pylint: disable=no-member
class TestInteractionWithOperators:
    """Test that the interaction between a ``HardwareHamiltonian`` and other operators work as
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
        """Test that a Hamiltonian and SProd can be added to a HardwareHamiltonian, and
        will be incorporated in the H_fixed term, with their coefficients included in H_coeffs_fixed.
        """
        R = drive(amplitude=f1, phase=0, wires=[0, 1])
        params = [1, 2]
        # Adding on the right
        new_pH = R + H
        assert isinstance(new_pH, HardwareHamiltonian)
        assert R.H_fixed() == 0
        qml.assert_equal(new_pH.H_fixed(), qml.s_prod(coeff, qml.PauliZ(0)))
        assert new_pH.coeffs_fixed[0] == coeff
        assert qml.math.allequal(new_pH(params, t=0.5).matrix(), qml.matrix(R(params, t=0.5) + H))
        # Adding on the left
        new_pH = H + R
        assert isinstance(new_pH, HardwareHamiltonian)
        assert R.H_fixed() == 0
        qml.assert_equal(new_pH.H_fixed(), qml.s_prod(coeff, qml.PauliZ(0)))
        assert new_pH.coeffs_fixed[0] == coeff
        assert qml.math.allequal(new_pH(params, t=0.5).matrix(), qml.matrix(R(params, t=0.5) + H))

    @pytest.mark.parametrize("op", ops)
    def test_add_other_operators(self, op):
        """Test that a Hamiltonian, SProd, Tensor or Operator can be added to a
        ParametrizedHamiltonian, and will be incorporated in the H_fixed term"""
        R = drive(amplitude=f1, phase=0, wires=[0, 1])
        params = [1]

        # Adding on the right
        new_pH = R + op
        assert isinstance(new_pH, HardwareHamiltonian)
        assert R.H_fixed() == 0
        qml.assert_equal(new_pH.H_fixed(), qml.s_prod(1, op))
        new_pH(params, 2)  # confirm calling does not raise error

        # Adding on the left
        new_pH = op + R
        assert isinstance(new_pH, HardwareHamiltonian)
        assert R.H_fixed() == 0
        qml.assert_equal(new_pH.H_fixed(), qml.s_prod(1, op))
        new_pH(params, 2)  # confirm calling does not raise error

    def test_adding_scalar_does_not_queue_id(self):
        """Test that no additional Identity operation is queued when adding a scalar."""
        R = drive(amplitude=f1, phase=0, wires=[0, 1])
        with qml.queuing.AnnotatedQueue() as q:
            R += 3
        assert len(q) == 0

    def test_unknown_type_raises_error(self):
        """Test that adding an invalid object to a HardwareHamiltonian raises an error."""
        R = drive(amplitude=f1, phase=0, wires=[0, 1])
        with pytest.raises(TypeError, match="unsupported operand type"):
            R += 3j


class TestDrive:
    """Unit tests for the ``drive`` function."""

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the
        ``ParametrizedHamiltonian`` returned by ``drive`` are correct."""

        Hd = drive(amplitude=1, phase=2, wires=[1, 2])

        assert isinstance(Hd, HardwareHamiltonian)
        assert Hd.wires == Wires([1, 2])
        assert len(Hd.ops) == 2  # 2 amplitude/phase terms

    def test_multiple_local_drives(self):
        """Test that adding multiple drive terms behaves as expected"""

        def fa(p, t):
            return np.sin(p * t) / (2 * np.pi)

        H1 = drive(amplitude=fa, phase=1, wires=[0, 3])
        H2 = drive(amplitude=0.5 / np.pi, phase=3, wires=[1, 2])
        Hd = H1 + H2

        ops_expected = [
            qml.Hamiltonian([0.5, 0.5], [qml.PauliX(1), qml.PauliX(2)]),
            qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(1), qml.PauliY(2)]),
            qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliX(3)]),
            qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(0), qml.PauliY(3)]),
        ]
        coeffs_expected = [
            np.cos(3),
            np.sin(3),
            AmplitudeAndPhase(np.cos, fa, 1),
            AmplitudeAndPhase(np.sin, fa, 1),
        ]
        H_expected = HardwareHamiltonian(
            coeffs_expected, ops_expected, reorder_fn=_reorder_parameters
        )

        # structure of Hamiltonian is as expected
        assert isinstance(Hd, HardwareHamiltonian)
        assert Hd.wires == Wires([1, 2, 0, 3])
        assert Hd.settings is None
        assert len(Hd.ops) == 4  # 2 terms for amplitude/phase

        # coefficients are correct
        # Callable coefficients are shifted to the end of the list.
        assert Hd.coeffs[:2] == [np.cos(3), np.sin(3)]
        assert isinstance(Hd.coeffs[2], AmplitudeAndPhase)
        assert isinstance(Hd.coeffs[3], AmplitudeAndPhase)

        # pulses were added correctly
        assert Hd.pulses == []

        # Hamiltonian is as expected
        qml.assert_equal(Hd([0.5, -0.5], t=5), H_expected([0.5, -0.5], t=5))


def callable_amp(p, t):
    """Compute the polynomial function polyval(p, t)."""
    return np.polyval(p, t)


def callable_phase(p, t):
    """Compute the function p_0 * sin(p_1 * t)."""
    return p[0] * np.sin(p[1] * t)


def sine_func(p, t):
    """Compute the function sin(p * t)."""
    return np.sin(p * t)


def cosine_fun(p, t):
    """Compute the function cos(p * t)."""
    return np.cos(p * t)


class TestAmplitudeAndPhase:
    """Test the AmplitudeAndPhase class that provides callable
    phase/amplitude combinations"""

    def test_amplitude_and_phase_no_callables(self):
        """Test that when calling amplitude_and_phase, if neither are callable,
        a float is returned instead of an AmplitudeAndPhase object"""
        f = amplitude_and_phase(np.sin, 3 / (2 * np.pi), 4)
        expected_result = 3 * np.sin(4)

        assert isinstance(f, float)
        assert f == expected_result

    def test_amplitude_and_phase_callable_phase(self):
        """Test that when calling amplitude_and_phase, if only phase is callable,
        an AmplitudeAndPhase object with callable phase and fixed amplitude is
        correctly created"""
        f = amplitude_and_phase(np.sin, 2.7 / (2 * np.pi), callable_phase)

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
        expected_result = callable_amp([1.7], 2) * (2 * np.pi) * np.sin(0.7)
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
        expected_result = (
            callable_amp([1.7], 2) * (2 * np.pi) * np.sin(callable_phase([1.3, 2.5], 2))
        )
        assert f([[1.7], [1.3, 2.5]], 2) == expected_result

    def test_callable_phase_and_amplitude_hamiltonian(self):
        """Test that using callable amplitude and phase in drive
        creates AmplitudeAndPhase callables, and the resulting Hamiltonian
        can be called successfully"""

        Hd = drive(sine_func, cosine_fun, wires=[0, 1])

        assert len(Hd.coeffs) == 2
        assert isinstance(Hd.coeffs[0], AmplitudeAndPhase)
        assert isinstance(Hd.coeffs[1], AmplitudeAndPhase)
        t = 1.7

        evaluated_H = Hd([3.4, 5.6], t)

        c1 = np.sin(3.4 * t) * (2 * np.pi) * np.cos(np.cos(5.6 * t))
        c2 = np.sin(3.4 * t) * (2 * np.pi) * np.sin(np.cos(5.6 * t))
        expected_H_parametrized = qml.sum(
            qml.s_prod(c1, qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliX(1)])),
            qml.s_prod(c2, qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(0), qml.PauliY(1)])),
        )
        qml.assert_equal(evaluated_H, expected_H_parametrized)

    def test_callable_phase_hamiltonian(self):
        """Test that using callable phase in drive creates AmplitudeAndPhase
        callables, and the resulting Hamiltonian can be called"""

        Hd = drive(7.2 / (2 * np.pi), sine_func, wires=[0, 1])

        assert len(Hd.coeffs) == 2
        assert isinstance(Hd.coeffs[0], AmplitudeAndPhase)
        assert isinstance(Hd.coeffs[1], AmplitudeAndPhase)
        t = 1.7

        evaluated_H = Hd([5.6], t)

        c1 = 7.2 * np.cos(np.sin(5.6 * t))
        c2 = 7.2 * np.sin(np.sin(5.6 * t))
        expected_H_parametrized = qml.sum(
            qml.s_prod(c1, qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliX(1)])),
            qml.s_prod(c2, qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(0), qml.PauliY(1)])),
        )

        qml.assert_equal(evaluated_H, expected_H_parametrized)

    def test_callable_amplitude_hamiltonian(self):
        """Test that using callable amplitude in drive creates AmplitudeAndPhase
        callables, and the resulting Hamiltonian can be called"""

        Hd = drive(sine_func, 4.3, wires=[0, 1])

        assert len(Hd.coeffs) == 2
        assert isinstance(Hd.coeffs[0], AmplitudeAndPhase)
        assert isinstance(Hd.coeffs[1], AmplitudeAndPhase)
        t = 1.7

        evaluated_H = Hd([3.4], t)

        c1 = np.sin(3.4 * t) * (2 * np.pi) * np.cos(4.3)
        c2 = np.sin(3.4 * t) * (2 * np.pi) * np.sin(4.3)
        expected_H_parametrized = qml.sum(
            qml.s_prod(c1, qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliX(1)])),
            qml.s_prod(c2, qml.Hamiltonian([-0.5, -0.5], [qml.PauliY(0), qml.PauliY(1)])),
        )

        qml.assert_equal(evaluated_H, expected_H_parametrized)

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
    def test_reorder_parameters_all(self, coeffs, params, expected_output):
        """Tests that the function organizing the parameters to pass to the
        HardwareHamiltonian works as expected when AmplitudeAndPhase callables
        are included"""

        assert _reorder_parameters(params, coeffs) == expected_output


class TestHardwarePulse:
    """Unit tests for the ``HardwarePulse`` class."""

    def test_init(self):
        """Test the initialization of the ``HardwarePulse`` class."""
        p = HardwarePulse(amplitude=4, phase=8, frequency=3, wires=[0, 4, 7])
        assert p.amplitude == 4
        assert p.phase == 8
        assert p.frequency == 3
        assert p.wires == Wires([0, 4, 7])

    def test_equal(self):
        """Test the ``__eq__`` method of the ``HardwarePulse`` class."""
        p1 = HardwarePulse(1, 2, 3, [0, 1])
        p2 = HardwarePulse(1, 2, 3, 0)
        p3 = HardwarePulse(1, 2, 3, [0, 1])
        assert p1 != p2
        assert p2 != p3
        assert p1 == p3


class TestIntegration:
    """Integration tests for the ``HardwareHamiltonian`` class."""

    @pytest.mark.jax
    def test_jitted_qnode(self):
        """Test that a ``HardwareHamiltonian`` class can be executed within a jitted qnode."""
        import jax
        import jax.numpy as jnp

        Hd = rydberg_interaction(register=atom_coordinates, wires=wires)

        def fa(p, t):
            return jnp.polyval(p, t)

        Ht = drive(amplitude=fa, phase=0, wires=1)

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

        params = (jnp.ones(5), jnp.array([1.0]))
        res = qnode(params)
        res_jit = qnode_jit(params)

        assert isinstance(res, jax.Array)
        assert qml.math.allclose(res, res_jit)

    @pytest.mark.jax
    def test_jitted_qnode_multidrive(self):
        """Test that a ``HardwareHamiltonian`` class with multiple drive terms can be
        executed within a jitted qnode."""
        import jax
        import jax.numpy as jnp

        Hd = rydberg_interaction(register=atom_coordinates, wires=wires)

        def fa(p, t):
            return jnp.polyval(p, t)

        def fc(p, t):
            return p[0] * jnp.sin(t) + jnp.cos(p[1] * t)

        H1 = drive(amplitude=fa, phase=0, wires=wires)
        H2 = drive(amplitude=fc, phase=3 * jnp.pi, wires=4)
        H3 = drive(amplitude=1.0, phase=0, wires=[3, 0])

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
            jnp.array([1.0, jnp.pi]),
            jnp.array([jnp.pi / 2, 0.5]),
        )
        res = qnode(params)
        res_jit = qnode_jit(params)

        assert isinstance(res, jax.Array)
        assert qml.math.allclose(res, res_jit)

    @pytest.mark.jax
    def test_jitted_qnode_all_coeffs_callable(self):
        """Test that a ``HardwareHamiltonian`` class can be executed within a
        jitted qnode when all coeffs are callable."""
        import jax
        import jax.numpy as jnp

        H_drift = rydberg_interaction(register=atom_coordinates, wires=wires)

        def fa(p, t):
            return jnp.polyval(p, t)

        def fb(p, t):
            return p[0] * jnp.sin(p[1] * t)

        H_drive = drive(amplitude=fa, phase=fb, wires=1)

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

        params = (jnp.ones(5), jnp.array([1.0, jnp.pi]))
        res = qnode(params)
        res_jit = qnode_jit(params)

        assert isinstance(res, jax.Array)
        assert qml.math.allclose(res, res_jit)
