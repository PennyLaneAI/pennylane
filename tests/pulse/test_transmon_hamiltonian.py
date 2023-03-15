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
Unit tests for the TransmonHamiltonian class.
"""
import warnings

# pylint: disable=too-few-public-methods
import numpy as np
import pytest

import pennylane as qml
from pennylane.pulse import TransmonHamiltonian, transmon_interaction, transmon_drive
from pennylane.pulse.transmon_hamiltonian import (
    TransmonPulse,
    AmplitudeAndPhase,
    _amplitude_and_phase,
    _transmon_reorder_parameters,
    a,
    ad,
)

from pennylane.wires import Wires

connections = [[0, 1], [1, 3], [2, 1], [4, 5]]
wires = [0, 1, 2, 3, 4, 5]
omega = 0.5 * np.arange(len(wires))
g = 0.1 * np.arange(len(connections))
delta = 0.3 * np.arange(len(wires))


# class TestTransmonHamiltonian:
#     """Unit tests for the properties of the TransmonHamiltonian class."""

#     # pylint: disable=protected-access
#     def test_initialization(self):
#         """Test the TransmonHamiltonian class is initialized correctly."""
#         rm = TransmonHamiltonian(coeffs=[], observables=[], register=atom_coordinates)

#         assert qml.math.allequal(rm.register, atom_coordinates)
#         assert rm.pulses == []
#         assert rm.wires == Wires([])
#         assert rm.interaction_coeff == 862690

#     def test_add_transmon_hamiltonian(self):
#         """Test that the __add__ dunder method works correctly."""
#         rm1 = TransmonHamiltonian(
#             coeffs=[1, 2],
#             observables=[qml.PauliX(4), qml.PauliZ(8)],
#             register=atom_coordinates,
#             pulses=[TransmonPulse(1, 2, 3, [4, 8])],
#         )
#         rm2 = TransmonHamiltonian(
#             coeffs=[2],
#             observables=[qml.PauliY(8)],
#             pulses=[TransmonPulse(5, 6, 7, 8)],
#         )
#         with warnings.catch_warnings():
#             # We make sure that no warning is raised
#             warnings.simplefilter("error")
#             sum_rm = rm1 + rm2
#         assert isinstance(sum_rm, TransmonHamiltonian)
#         assert qml.math.allequal(sum_rm.coeffs, [1, 2, 2])
#         assert all(
#             qml.equal(op1, op2)
#             for op1, op2 in zip(sum_rm.ops, [qml.PauliX(4), qml.PauliZ(8), qml.PauliY(8)])
#         )
#         assert qml.math.allequal(sum_rm.register, atom_coordinates)
#         assert sum_rm.pulses == [TransmonPulse(1, 2, 3, [4, 8]), TransmonPulse(5, 6, 7, 8)]

#     def test_add_parametrized_hamiltonian(self):
#         # ToDo: check returned object is a TransmonHamiltonian and can be called!
#         """Tests that adding a `TransmonHamiltonian` and `ParametrizedHamiltonian` works as
#         expected."""
#         coeffs = [2, 3]
#         ops = [qml.PauliZ(0), qml.PauliX(2)]
#         h_wires = [0, 2]

#         rh = TransmonHamiltonian(
#             coeffs=[coeffs[0]],
#             observables=[ops[0]],
#             pulses=[TransmonPulse(5, 6, 7, 8)],
#         )
#         ph = qml.pulse.ParametrizedHamiltonian(coeffs=[coeffs[1]], observables=[ops[1]])

#         res1 = rh + ph
#         res2 = ph + rh

#         assert res1.coeffs_fixed == coeffs
#         assert res1.coeffs_parametrized == []
#         assert all(qml.equal(op1, op2) for op1, op2 in zip(res1.ops_fixed, ops))
#         assert res1.ops_parametrized == []
#         assert res1.wires == qml.wires.Wires(h_wires)

#         coeffs.reverse()
#         ops.reverse()
#         h_wires.reverse()

#         assert res2.coeffs_fixed == coeffs
#         assert res2.coeffs_parametrized == []
#         assert all(qml.equal(op1, op2) for op1, op2 in zip(res2.ops_fixed, ops))
#         assert res2.ops_parametrized == []
#         assert res2.wires == qml.wires.Wires(h_wires)

#     # def test_radd_parametrized_hamiltonian(self):
#     #     """Tests that adding a `TransmonHamiltonian` and `ParametrizedHamiltonian` works as
#     #     expected."""
#     #     # ToDo: check returned object is a TransmonHamiltonian and can be called!
#     #     pass

#     def test_add_raises_error(self):
#         """Test that an error is raised if two TransmonHamiltonians with registers are added."""
#         rm1 = TransmonHamiltonian(
#             coeffs=[1],
#             observables=[qml.PauliX(0)],
#             register=atom_coordinates,
#             pulses=[TransmonPulse(1, 2, 3, 4)],
#         )
#         with pytest.raises(
#             ValueError, match="We cannot add two Hamiltonians with an interaction term"
#         ):
#             _ = rm1 + rm1

#     def test_add_raises_warning(self):
#         """Test that an error is raised when adding two TransmonHamiltonians where one Hamiltonian
#         contains pulses on wires that are not present in the register."""
#         coords = [[0, 0], [0, 5], [5, 0]]

#         Hd = transmon_interaction(register=coords, wires=[0, 1, 2])
#         Ht = transmon_drive(2, 3, 4, wires=3)

#         with pytest.warns(
#             UserWarning,
#             match="The wires of the laser fields are not present in the Transmon ensemble",
#         ):
#             _ = Hd + Ht

#         with pytest.warns(
#             UserWarning,
#             match="The wires of the laser fields are not present in the Transmon ensemble",
#         ):
#             _ = Ht + Hd


# def f1(p, t):
#     return p * np.sin(t) * (t - 1)


# def f2(p, t):
#     return p * np.cos(t**2)


# param = [1.2, 2.3]


# class TestInteractionWithOperators:
#     """Test that the interaction between a ``TransmonHamiltonian`` and other operators work as
#     expected."""

#     ops_with_coeffs = (
#         (qml.Hamiltonian([2], [qml.PauliZ(0)]), 2),
#         (qml.Hamiltonian([1.7], [qml.PauliZ(0)]), 1.7),
#         (qml.ops.SProd(3, qml.PauliZ(0)), 3),
#     )
#     ops = (
#         qml.PauliX(2),
#         qml.PauliX(2) @ qml.PauliX(3),
#         qml.CNOT([0, 1]),
#     )  # ToDo: maybe add more operators to test here?

#     # ToDo: check returned object is a TransmonHamiltonian and can be called!
#     @pytest.mark.parametrize("H, coeff", ops_with_coeffs)
#     def test_add_special_operators(self, H, coeff):
#         """Test that a Hamiltonian and SProd can be added to a TransmonHamiltonian, and
#         will be incorporated in the H_fixed term, with their coefficients included in H_coeffs_fixed.
#         """
#         R = transmon_drive(amplitude=f1, phase=0, detuning=f2, wires=[0, 1])
#         params = [1, 2]
#         # Adding on the right
#         new_pH = R + H
#         assert R.H_fixed() == 0
#         assert qml.equal(new_pH.H_fixed(), qml.s_prod(coeff, qml.PauliZ(0)))
#         assert new_pH.coeffs_fixed[0] == coeff
#         assert qml.math.allequal(new_pH(params, t=0.5).matrix(), qml.matrix(R(params, t=0.5) + H))
#         # Adding on the left
#         new_pH = H + R
#         assert R.H_fixed() == 0
#         assert qml.equal(new_pH.H_fixed(), qml.s_prod(coeff, qml.PauliZ(0)))
#         assert new_pH.coeffs_fixed[0] == coeff
#         assert qml.math.allequal(new_pH(params, t=0.5).matrix(), qml.matrix(R(params, t=0.5) + H))

#     @pytest.mark.parametrize("op", ops)
#     def test_add_other_operators(self, op):
#         """Test that a Hamiltonian, SProd, Tensor or Operator can be added to a
#         ParametrizedHamiltonian, and will be incorporated in the H_fixed term"""
#         R = transmon_drive(amplitude=f1, phase=0, detuning=f2, wires=[0, 1])

#         # Adding on the right
#         new_pH = R + op
#         assert R.H_fixed() == 0
#         assert qml.equal(new_pH.H_fixed(), qml.s_prod(1, op))

#         # Adding on the left
#         new_pH = op + R
#         assert R.H_fixed() == 0
#         assert qml.equal(new_pH.H_fixed(), qml.s_prod(1, op))


class TestTransmonInteraction:
    """Unit tests for the ``transmon_interaction`` function."""

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
        ``transmon_interaction`` are correct."""
        Hd = transmon_interaction(
            connections=connections, omega=omega, g=g, delta=None, wires=wires, d=2
        )

        assert isinstance(Hd, TransmonHamiltonian)
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


class TestTransmonDrive:
    """Unit tests for the ``transmon_drive`` function."""

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
        ``transmon_drive`` are correct."""

        Hd = transmon_drive(amplitude=1, phase=2, wires=[1, 2])

        assert isinstance(Hd, TransmonHamiltonian)
        assert Hd.wires == Wires([1, 2])
        assert Hd.connections is None
        assert len(Hd.ops) == 2  # one for a and one of a^\dagger
        assert Hd.pulses == [TransmonPulse(1, 2, [1, 2])]

    # odd behavior when adding two drives/TransmonHamiltonians
    # @pytest.mark.xfail
    def test_multiple_local_drives(self):
        """Test that adding multiple drive terms behaves as expected"""

        def fa(p, t):
            return np.sin(p * t)

        def fb(p, t):
            return np.cos(p * t)

        H1 = transmon_drive(amplitude=fa, phase=1, wires=[0, 3])
        H2 = transmon_drive(amplitude=1, phase=3, wires=[1, 2])
        Hd = H1 + H2

        ops_expected = [a(1) + a(2), ad(1) + ad(2), a(0) + a(3), ad(0) + ad(3)]
        coeffs_expected = [
            1.0 * qml.math.exp(1j * 3.0),
            1.0 * qml.math.exp(-1j * 3.0),
            AmplitudeAndPhase(1, fa, 1),
            AmplitudeAndPhase(-1, fa, 1),
        ]
        H_expected = TransmonHamiltonian(coeffs_expected, ops_expected)

        # structure of Hamiltonian is as expected
        assert isinstance(Hd, TransmonHamiltonian)
        assert Hd.wires == Wires([1, 2, 0, 3])  # TODO: Why is the order reversed?
        assert Hd.connections is None
        assert len(Hd.ops) == 4  # 2 terms for amplitude/phase and one detuning for each drive

        # coefficients are correct
        # Callable coefficients are shifted to the end of the list.
        assert Hd.coeffs[:2] == coeffs_expected[:2]
        assert isinstance(Hd.coeffs[2], AmplitudeAndPhase)
        assert isinstance(Hd.coeffs[3], AmplitudeAndPhase)

        # # pulses were added correctly
        assert len(Hd.pulses) == 2
        assert Hd.pulses == H1.pulses + H2.pulses

        # # Hamiltonian is as expected
        assert qml.equal(Hd([0.5, -0.5], t=5), H_expected([0.5, -0.5], t=5))


# def callable_amp(p, t):
#     return np.polyval(p, t)


# def callable_phase(p, t):
#     return p[0] * np.sin(p[1] * t)


# def sine_func(p, t):
#     return np.sin(p * t)


# def cosine_fun(p, t):
#     return np.cos(p * t)


# class TestAmplitudeAndPhase:
#     """Test the AmplitudeAndPhase class that provides callable
#     phase/amplitude combinations"""

#     def test_amplitude_and_phase_no_callables(self):
#         """Test that when calling amplitude_and_phase, if neither are callable,
#         a float is returned instead of an AmplitudeAndPhase object"""
#         f = amplitude_and_phase(np.sin, 3, 4)
#         expected_result = 3 * np.sin(4)

#         assert isinstance(f, float)
#         assert f == expected_result

#     def test_amplitude_and_phase_callable_phase(self):
#         """Test that when calling amplitude_and_phase, if only phase is callable,
#         an AmplitudeAndPhase object with callable phase and fixed amplitude is
#         correctly created"""
#         f = amplitude_and_phase(np.sin, 2.7, callable_phase)

#         # attributes are correct
#         assert isinstance(f, AmplitudeAndPhase)
#         assert f.amp_is_callable is False
#         assert f.phase_is_callable is True
#         assert f.func.__name__ == "callable_phase"

#         # calling yields expected result
#         expected_result = 2.7 * np.sin(callable_phase([1.3, 2.5], 2))
#         assert f([1.3, 2.5], 2) == expected_result

#     def test_amplitude_and_phase_callable_amplitude(self):
#         """Test that when calling amplitude_and_phase, if only amplitude is callable,
#         an AmplitudeAndPhase object with callable amplitude and fixed phase is
#         correctly created"""
#         f = amplitude_and_phase(np.sin, callable_amp, 0.7)

#         # attributes are correct
#         assert isinstance(f, AmplitudeAndPhase)
#         assert f.amp_is_callable is True
#         assert f.phase_is_callable is False
#         assert f.func.__name__ == "callable_amp"

#         # calling yields expected result
#         expected_result = callable_amp([1.7], 2) * np.sin(0.7)
#         assert f([1.7], 2) == expected_result

#     def test_amplitude_and_phase_both_callable(self):
#         """Test that when calling amplitude_and_phase, if both are callable,
#         an AmplitudeAndPhase object with callable amplitude and phase is
#         correctly created"""
#         f = amplitude_and_phase(np.sin, callable_amp, callable_phase)

#         # attributes are correct
#         assert isinstance(f, AmplitudeAndPhase)
#         assert f.amp_is_callable is True
#         assert f.phase_is_callable is True
#         assert f.func.__name__ == "callable_amp_and_phase"

#         # calling yields expected result
#         expected_result = callable_amp([1.7], 2) * np.sin(callable_phase([1.3, 2.5], 2))
#         assert f([[1.7], [1.3, 2.5]], 2) == expected_result

#     def test_callable_phase_and_amplitude_hamiltonian(self):
#         """Test that using callable amplitude and phase in transmon_drive
#         creates AmplitudeAndPhase callables, and the resulting Hamiltonian
#         can be called successfully"""

#         detuning = 2

#         Hd = transmon_drive(sine_func, cosine_fun, detuning, wires=[0, 1])

#         assert len(Hd.coeffs) == 3
#         assert isinstance(Hd.coeffs[1], AmplitudeAndPhase)
#         assert isinstance(Hd.coeffs[2], AmplitudeAndPhase)
#         t = 1.7

#         evaluated_H = Hd([3.4, 5.6], t)

#         expected_H_fixed = qml.s_prod(
#             detuning, qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(1)])
#         )

#         c1 = np.sin(3.4 * t) * np.cos(np.cos(5.6 * t))
#         c2 = np.sin(3.4 * t) * np.sin(np.cos(5.6 * t))
#         expected_H_parametrized = qml.sum(
#             qml.s_prod(c1, qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)])),
#             qml.s_prod(c2, qml.sum(qml.s_prod(-1, qml.PauliY(0)), qml.s_prod(-1, qml.PauliY(1)))),
#         )

#         assert qml.equal(evaluated_H[0], expected_H_fixed)
#         assert qml.equal(evaluated_H[1], expected_H_parametrized)

#     def test_callable_phase_hamiltonian(self):
#         """Test that using callable phase in transmon_drive creates AmplitudeAndPhase
#         callables, and the resulting Hamiltonian can be called"""

#         detuning = 2

#         Hd = transmon_drive(7.2, sine_func, detuning, wires=[0, 1])

#         assert len(Hd.coeffs) == 3
#         assert isinstance(Hd.coeffs[1], AmplitudeAndPhase)
#         assert isinstance(Hd.coeffs[2], AmplitudeAndPhase)
#         t = 1.7

#         evaluated_H = Hd([5.6], t)

#         expected_H_fixed = qml.s_prod(
#             detuning, qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(1)])
#         )

#         c1 = 7.2 * np.cos(np.sin(5.6 * t))
#         c2 = 7.2 * np.sin(np.sin(5.6 * t))
#         expected_H_parametrized = qml.sum(
#             qml.s_prod(c1, qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)])),
#             qml.s_prod(c2, qml.sum(qml.s_prod(-1, qml.PauliY(0)), qml.s_prod(-1, qml.PauliY(1)))),
#         )

#         assert qml.equal(evaluated_H[0], expected_H_fixed)
#         assert qml.equal(evaluated_H[1], expected_H_parametrized)

#     def test_callable_amplitude_hamiltonian(self):
#         """Test that using callable amplitude in transmon_drive creates AmplitudeAndPhase
#         callables, and the resulting Hamiltonian can be called"""

#         detuning = 2

#         Hd = transmon_drive(sine_func, 4.3, detuning, wires=[0, 1])

#         assert len(Hd.coeffs) == 3
#         assert isinstance(Hd.coeffs[1], AmplitudeAndPhase)
#         assert isinstance(Hd.coeffs[2], AmplitudeAndPhase)
#         t = 1.7

#         evaluated_H = Hd([3.4], t)

#         expected_H_fixed = qml.s_prod(
#             detuning, qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(1)])
#         )

#         c1 = np.sin(3.4 * t) * np.cos(4.3)
#         c2 = np.sin(3.4 * t) * np.sin(4.3)
#         expected_H_parametrized = qml.sum(
#             qml.s_prod(c1, qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)])),
#             qml.s_prod(c2, qml.sum(qml.s_prod(-1, qml.PauliY(0)), qml.s_prod(-1, qml.PauliY(1)))),
#         )

#         assert qml.equal(evaluated_H[0], expected_H_fixed)
#         assert qml.equal(evaluated_H[1], expected_H_parametrized)

#     COEFFS_AND_PARAMS = [
#         (
#             [AmplitudeAndPhase(np.cos, f1, f2), AmplitudeAndPhase(np.sin, f1, f2), f2],
#             [1, 2, 3],
#             [[1, 2], [1, 2], 3],
#         ),
#         (
#             [AmplitudeAndPhase(np.cos, f1, 6.3), AmplitudeAndPhase(np.sin, f1, 6.3), f2],
#             [1, 3],
#             [1, 1, 3],
#         ),
#         (
#             [AmplitudeAndPhase(np.cos, 6.3, f1), AmplitudeAndPhase(np.sin, 6.3, f1), f2],
#             [1, 3],
#             [1, 1, 3],
#         ),
#         (
#             [f1, AmplitudeAndPhase(np.cos, f1, 5.7), AmplitudeAndPhase(np.sin, f1, 5.7), f2],
#             [1, 2, 3],
#             [1, 2, 2, 3],
#         ),
#         (
#             [f1, AmplitudeAndPhase(np.cos, f1, f1), AmplitudeAndPhase(np.sin, f1, f1), f2],
#             [2, [3, 5], 8, [13, 21]],
#             [2, [[3, 5], 8], [[3, 5], 8], [13, 21]],
#         ),
#         ([f1, f2, f1, f2], [1, 2, 3, 4], [1, 2, 3, 4]),
#     ]

#     @pytest.mark.parametrize("coeffs, params, expected_output", COEFFS_AND_PARAMS)
#     def test_transmon_reorder_parameters_all(self, coeffs, params, expected_output):
#         """Tests that the function organizing the parameters to pass to the
#         TransmonHamiltonian works as expected when AmplitudeAndPhase callables
#         are included"""

#         assert _transmon_reorder_parameters(params, coeffs) == expected_output


class TestTransmonPulse:
    """Unit tests for the ``TransmonPulse`` class."""

    def test_init(self):
        """Test the initialization of the ``TransmonPulse`` class."""
        p = TransmonPulse(amplitude=4, phase=8, wires=[0, 4, 7])
        assert p.amplitude == 4
        assert p.phase == 8
        assert p.wires == Wires([0, 4, 7])

    def test_equal(self):
        """Test the ``__eq__`` method of the ``TransmonPulse`` class."""
        p1 = TransmonPulse(1, 2, [0, 1])
        p2 = TransmonPulse(1, 2, 0)
        p3 = TransmonPulse(1, 2, [0, 1])
        assert p1 != p2
        assert p2 != p3
        assert p1 == p3


# class TestIntegration:
#     """Integration tests for the ``TransmonHamiltonian`` class."""

#     @pytest.mark.jax
#     def test_jitted_qnode(self):
#         """Test that a ``TransmonHamiltonian`` class can be executed within a jitted qnode."""
#         import jax
#         import jax.numpy as jnp

#         Hd = transmon_interaction(register=atom_coordinates, wires=wires)

#         def fa(p, t):
#             return jnp.polyval(p, t)

#         def fb(p, t):
#             return p[0] * jnp.sin(p[1] * t)

#         Ht = transmon_drive(amplitude=fa, phase=0, detuning=fb, wires=1)

#         dev = qml.device("default.qubit", wires=wires)

#         ts = jnp.array([0.0, 3.0])
#         H_obj = sum(qml.PauliZ(i) for i in range(2))

#         @jax.jit
#         @qml.qnode(dev, interface="jax")
#         def qnode(params):
#             qml.evolve(Hd + Ht)(params, ts)
#             return qml.expval(H_obj)

#         params = (jnp.ones(5), jnp.array([1.0, jnp.pi]))
#         res = qnode(params)

#         assert isinstance(res, jax.Array)

#     @pytest.mark.jax
#     def test_jitted_qnode_multidrive(self):
#         """Test that a ``TransmonHamiltonian`` class with multiple drive terms can be
#         executed within a jitted qnode."""
#         import jax
#         import jax.numpy as jnp

#         Hd = transmon_interaction(register=atom_coordinates, wires=wires)

#         def fa(p, t):
#             return jnp.polyval(p, t)

#         def fb(p, t):
#             return p[0] * jnp.sin(p[1] * t)

#         def fc(p, t):
#             return p[0] * jnp.sin(t) + jnp.cos(p[1] * t)

#         H1 = transmon_drive(amplitude=fa, detuning=fb, phase=0, wires=1)
#         H2 = transmon_drive(amplitude=fc, detuning=jnp.pi / 4, phase=3 * jnp.pi, wires=4)

#         dev = qml.device("default.qubit", wires=wires)

#         ts = jnp.array([0.0, 3.0])
#         H_obj = sum(qml.PauliZ(i) for i in range(2))

#         @jax.jit
#         @qml.qnode(dev, interface="jax")
#         def qnode(params):
#             qml.evolve(Hd + H1 + H2)(params, ts)
#             return qml.expval(H_obj)

#         params = (jnp.ones(5), jnp.array([1.0, jnp.pi]), jnp.array([jnp.pi / 2, 0.5]))
#         res = qnode(params)

#         assert isinstance(res, jax.Array)
