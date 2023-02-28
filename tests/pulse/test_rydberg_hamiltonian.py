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
# pylint: disable=too-few-public-methods
import numpy as np
import pytest

import pennylane as qml
from pennylane.pulse import RydbergHamiltonian, rydberg_interaction, rydberg_transition
from pennylane.pulse.rydberg_hamiltonian import RydbergPulse
from pennylane.wires import Wires

atom_coordinates = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
wires = [1, 6, 0, 2, 4, 3]


class TestRydbergHamiltonian:
    """Unit tests for the properties of the RydbergHamiltonian class."""

    # pylint: disable=protected-access
    def test_initialization(self):
        """Test the RydbergHamiltonian class is initialized correctly."""
        rm = RydbergHamiltonian(coeffs=[], observables=[], register=atom_coordinates)

        assert qml.math.allequal(rm.register, atom_coordinates)
        assert rm.pulses == []
        assert rm.wires == Wires([])
        assert rm.interaction_coeff == 862690 * 2 * np.pi

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


def f1(p, t):
    return p * np.sin(t) * (t - 1)


def f2(p, t):
    return p * np.cos(t**2)


param = [1.2, 2.3]


class TestInteractionWithOperators:
    """Test that the interaction between a ``RydbergHamiltonian`` and other operators work as
    expected."""

    ops_with_coeffs = (
        (qml.Hamiltonian([2], [qml.PauliZ(0)]), 2),
        (qml.Hamiltonian([1.7], [qml.PauliZ(0)]), 1.7),
        (qml.ops.SProd(3, qml.PauliZ(0)), 3),
    )
    ops = (
        qml.PauliX(2),
        qml.PauliX(2) @ qml.PauliX(3),
        qml.CNOT([0, 1]),
    )  # ToDo: maybe add more operators to test here?

    @pytest.mark.parametrize("H, coeff", ops_with_coeffs)
    def test_add_special_operators(self, H, coeff):
        """Test that a Hamiltonian and SProd can be added to a RydbergHamiltonian, and
        will be incorporated in the H_fixed term, with their coefficients included in H_coeffs_fixed.
        """
        R = rydberg_transition(rabi=f1, detuning=f2, phase=0, wires=[0, 1])
        params = [1, 2]
        # Adding on the right
        new_pH = R + H
        assert R.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(coeff, qml.PauliZ(0)))
        assert new_pH.coeffs_fixed[0] == coeff
        assert qml.math.allequal(new_pH(params, t=0.5).matrix(), qml.matrix(R(params, t=0.5) + H))
        # Adding on the left
        new_pH = H + R
        assert R.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(coeff, qml.PauliZ(0)))
        assert new_pH.coeffs_fixed[0] == coeff
        assert qml.math.allequal(new_pH(params, t=0.5).matrix(), qml.matrix(R(params, t=0.5) + H))

    @pytest.mark.parametrize("op", ops)
    def test_add_other_operators(self, op):
        """Test that a Hamiltonian, SProd, Tensor or Operator can be added to a
        ParametrizedHamiltonian, and will be incorporated in the H_fixed term"""
        R = rydberg_transition(rabi=f1, detuning=f2, phase=0, wires=[0, 1])

        # Adding on the right
        new_pH = R + op
        assert R.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(1, op))

        # Adding on the left
        new_pH = op + R
        assert R.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(1, op))


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


class TestRydbergTransition:
    """Unit tests for the ``rydberg_transition`` function."""

    def test_attributes_and_number_of_terms(self):
        """Test that the attributes and the number of terms of the ``ParametrizedHamiltonian`` returned by
        ``rydberg_transition`` are correct."""
        Hd = rydberg_transition(rabi=1, phase=2, detuning=3, wires=[1, 2])

        assert isinstance(Hd, RydbergHamiltonian)
        assert Hd.interaction_coeff == 862690 * 2 * np.pi
        assert Hd.wires == Wires([1, 2])
        assert Hd.register is None
        assert len(Hd.ops) == 2  # rabi and detuning terms of the Hamiltonian
        assert Hd.pulses == [RydbergPulse(1, 2, 3, [1, 2])]


class TestRydbergPulse:
    """Unit tests for the ``RydbergPulse`` class."""

    def test_init(self):
        """Test the initialization of the ``RydbergPulse`` class."""
        p = RydbergPulse(rabi=4, phase=8, detuning=9, wires=[0, 4, 7])
        assert p.rabi == 4
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

        Ht = rydberg_transition(rabi=fa, detuning=fb, phase=0, wires=1)

        dev = qml.device("default.qubit", wires=wires)

        ts = jnp.array([0.0, 3.0])
        H_obj = sum(qml.PauliZ(i) for i in range(2))

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def qnode(params):
            qml.evolve(Hd + Ht)(params, ts)
            return qml.expval(H_obj)

        params = (jnp.ones(5), jnp.array([1.0, jnp.pi]))
        res = qnode(params)

        assert isinstance(res, jax.Array)
