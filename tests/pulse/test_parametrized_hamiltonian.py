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
Unit tests for the ParametrizedHamiltonian class
"""
# pylint: disable=no-member, import-outside-toplevel
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.pulse import ParametrizedHamiltonian
from pennylane.wires import Wires


def f1(p, t):
    """Compute the function p * sin(t) * (t - 1)."""
    return p * np.sin(t) * (t - 1)


def f2(p, t):
    """Compute the function p * cos(t**2)."""
    return p * np.cos(t**2)


param = [1.2, 2.3]

test_example = ParametrizedHamiltonian(
    [1, 2, f1, f2], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2), qml.Hadamard(3)]
)


class TestInitialization:
    """Test initialization of the ParametrizedHamiltonian class"""

    def test_initialization_via_dot(self):
        """Test that using qml.dot initializes a ParametrizedHamiltonian"""

        XX = qml.PauliX(0) @ qml.PauliX(1)
        YY = qml.PauliY(0) @ qml.PauliY(1)
        ZZ = qml.PauliZ(0) @ qml.PauliZ(1)

        coeffs = [2, f1, f2]
        ops = [XX, YY, ZZ]

        H = qml.dot(coeffs, ops)
        expected_H = ParametrizedHamiltonian(coeffs, ops)

        qml.assert_equal(H([1.2, 2.3], 3.4), expected_H([1.2, 2.3], 3.4))

    def test_initialization_via_addition(self):
        """Test that adding combinations of operators and numbers/callables initializes
        a ParametrizedHamiltonian"""

        XX = qml.PauliX(0) @ qml.PauliX(1)
        YY = qml.PauliY(0) @ qml.PauliY(1)
        ZZ = qml.PauliZ(0) @ qml.PauliZ(1)

        H = 2 * XX + f1 * YY + f2 * ZZ

        coeffs = [2, f1, f2]
        ops = [XX, YY, ZZ]
        expected_H = ParametrizedHamiltonian(coeffs, ops)

        qml.assert_equal(H([1.2, 2.3], 3.4), expected_H([1.2, 2.3], 3.4))

    def test_mismatched_coeffs_and_obs_raises_error(self):
        """Test that an error is raised if the length of the list of coefficients
        and the list of observables don't match"""
        coeffs = [1, 2, f1]
        obs = [qml.PauliX(0), qml.PauliY(1)]
        # fmt: off
        with pytest.raises(ValueError, match="number of coefficients and operators does not match."):
            # fmt: on
            ParametrizedHamiltonian(coeffs, obs)

    def test_H_fixed_lists(self):
        """Test that attributes H_fixed_ops and H_fixed_coeffs are as expected"""
        assert test_example.coeffs_fixed == [1, 2]
        for op1, op2 in zip(test_example.ops_fixed, [qml.PauliX(0), qml.PauliY(1)]):
            qml.assert_equal(op1, op2)

    def test_H_parametrized_lists(self):
        """Test that attributes H_parametrized_ops and H_parametrized_coeffs are as expected"""
        assert test_example.coeffs_parametrized == [f1, f2]
        for op1, op2 in zip(test_example.ops_parametrized, [qml.PauliZ(2), qml.Hadamard(3)]):
            qml.assert_equal(op1, op2)

    def test_H_fixed(self):
        """Test that H_fixed() is an Operator of the expected form"""
        H_fixed = test_example.H_fixed()
        op = qml.sum(qml.s_prod(1, qml.PauliX(0)), qml.s_prod(2, qml.PauliY(1)))
        qml.assert_equal(H_fixed, op)

    def test_H_parametrized(self):
        """Test H_parametrized is a function that, when passed parameters,
        returns an Operator of the expected format."""
        H_param = test_example.H_parametrized

        H = H_param([1, 2], 0.5)
        assert len(H) == 2
        assert isinstance(H, qml.ops.Sum)
        assert isinstance(H[0], qml.ops.SProd)
        assert isinstance(H[1], qml.ops.SProd)

    def test__repr__(self):
        """Test repr method returns expected string"""

        coeffs = [2.0, f1, f2]
        ops = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
        H = ParametrizedHamiltonian(coeffs, ops)
        expected = "(\n    2.0 * X(0)\n  + f1(params_0, t) * Y(0)\n  + f2(params_1, t) * Z(0)\n)"

        assert repr(H) == expected

    def test__repr__with_Sum_and_Prod(self):
        """Test repr method returns expected string"""

        h0 = qml.sum(*(qml.X(i) for i in range(5)))
        h1 = qml.prod(qml.X(0), qml.sum(qml.X(1), qml.X(2)))
        H = qml.pulse.constant * h0 + qml.pulse.constant * h1 + qml.pulse.constant * qml.X(1)

        expected = "(\n    constant(params_0, t) * (X(0) + X(1) + X(2) + X(3) + X(4))\n  + constant(params_1, t) * X(0) @ (X(1) + X(2))\n  + constant(params_2, t) * X(1)\n)"

        assert repr(H) == expected

    def test_repr_with_class_objects(self):
        """Test repr method with class objects r
        eturns expected string"""

        class f3:  # pylint: disable=too-few-public-methods
            """Dummy class"""

            def __init__(self, hyper_param):
                self.hyper = hyper_param

            def __call__(self, p, t):
                return self.hyper * p * t

        coeffs = [2.0, f1, f2, f3(0.5)]
        observables = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0), qml.PauliX(0)]
        H = ParametrizedHamiltonian(coeffs, observables)
        expected = "(\n    2.0 * X(0)\n  + f1(params_0, t) * Y(0)\n  + f2(params_1, t) * Z(0)\n  + f3(params_2, t) * X(0)\n)"
        assert repr(H) == expected

    def test_wire_attribute(self):
        """Tests that the wires attribute contains the expected wires, in the expected order"""
        coeffs = [1, f1, 2, f2]
        ops = [qml.PauliX(3), qml.PauliX(2), qml.PauliX(0), qml.PauliX(1)]

        H = ParametrizedHamiltonian(coeffs, ops)
        assert H.wires == H(param, 2).wires
        assert H.wires == Wires([3, 0, 2, 1])


class TestCall:
    """Test that calling the ParametrizedHamiltonian behaves as expected."""

    coeffs_and_ops_and_params = (
        (
            [f1, f2],
            [qml.PauliX(0), qml.PauliY(1)],
            [1, 2],
            [0, 2],
        ),  # no H_fixed term, multiple H_parametrized terms
        ([f1], [qml.PauliX(0)], [1], [0, 1]),  # no H_fixed term, one H_parametrized term
        (
            [1.2, 2.3],
            [qml.PauliX(0), qml.PauliY(1)],
            [],
            [2, 0],
        ),  # no H_parametrized term, multiple H_fixed terms
        ([1.2], [qml.PauliX(0)], [], [1, 0]),  # no H_parametrized term, one H_fixed term
        ([1, f1], [qml.PauliX(3), qml.PauliY(2)], [1], [1, 1]),  # one of each
        (
            [1.2, f1, 2.3, f2],
            [qml.PauliX(0) for i in range(4)],
            [1, 2],
            [2, 2],
        ),  # multiples of each
    )

    @pytest.mark.parametrize("coeffs, ops, params, num_terms", coeffs_and_ops_and_params)
    def test_call_succeeds_for_different_shapes(self, coeffs, ops, params, num_terms):
        """Test that calling the ParametrizedHamiltonian succeeds for
        different numbers of H_fixed and H_parametrized terms"""
        pH = ParametrizedHamiltonian(coeffs, ops)
        pH(params, 0.5)
        assert len(pH.ops_fixed) == num_terms[0]
        assert len(pH.ops_parametrized) == num_terms[1]

    def test_call_returns_expected_results(self):
        """Test result of calling the ParametrizedHamiltonian makes sense"""
        pH = ParametrizedHamiltonian([1.2, f1, 2.3, f2], [qml.PauliX(i) for i in range(4)])
        params = [4.5, 6.7]
        t = 2
        op = pH(params, t)

        assert isinstance(op, qml.ops.Sum)
        assert len(op) == 2

        H_fixed = op[0]
        H_parametrized = op[1]
        expected_H_fixed = qml.sum(qml.s_prod(1.2, qml.PauliX(0)), qml.s_prod(2.3, qml.PauliX(2)))
        expected_H_parametrized = qml.sum(
            qml.s_prod(f1(params[0], t), qml.PauliX(1)), qml.s_prod(f2(params[1], t), qml.PauliX(3))
        )

        qml.assert_equal(H_fixed, expected_H_fixed)
        qml.assert_equal(H_parametrized, expected_H_parametrized)

    def test_call_with_qutrit_operators(self):
        """Test that the ParametrizedHamiltonian can be created and called to initialize an
        operator consisting of qutrit operations"""

        def f(x, t):
            return x + 2 * t

        coeffs = [f, 2]
        obs = [qml.GellMann(wires=0, index=1), qml.GellMann(wires=0, index=2)]

        H = ParametrizedHamiltonian(coeffs, obs)

        assert isinstance(H([2], 4), qml.ops.Sum)
        assert (
            repr(H([2], t=4)) == "(\n    2 * GellMann2(wires=[0])\n  + 10 * GellMann1(wires=[0])\n)"
        )
        assert np.all(
            qml.matrix(H([2], 4))
            == np.array(
                [
                    [0.0 + 0.0j, 10.0 - 2.0j, 0.0 + 0.0j],
                    [10.0 + 2.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ]
            )
        )

    def test_call_raises_error(self):
        """Test that if the user calls a `ParametrizedHamiltonian` with the incorrect number
        of parameters, an error is raised."""

        def f(x, t):
            return x + 2 * t

        coeffs = [f, 2]
        obs = [qml.GellMann(wires=0, index=1), qml.GellMann(wires=0, index=2)]
        H = ParametrizedHamiltonian(coeffs, obs)
        # fmt: off
        with pytest.raises(ValueError, match="The length of the params argument and the number of scalar-valued functions must be the same",):
        # fmt: on
            H(params=[1, 2], t=0.5)


class TestInteractionWithOperators:
    """Test that arithmetic operations involving or creating ParametrizedHamiltonians behave as expected"""

    ops_with_coeffs = (
        (qml.Hamiltonian([2], [qml.PauliZ(0)]), 2),
        (qml.Hamiltonian([1.7], [qml.PauliZ(0)]), 1.7),
        (qml.ops.SProd(3, qml.PauliZ(0)), 3),
        (2 * qml.PauliZ(0), 2),
    )
    ops = (
        qml.PauliX(2),
        qml.PauliX(2) @ qml.PauliX(3),
        qml.CNOT([0, 1]),
    )  # ToDo: maybe add more operators to test here?

    @pytest.mark.parametrize("H, coeff", ops_with_coeffs)
    def test_add_special_operators(self, H, coeff):
        """Test that a Hamiltonian and SProd can be added to a ParametrizedHamiltonian, and
        will be incorporated in the H_fixed term, with their coefficients included in H_coeffs_fixed
        """
        pH = ParametrizedHamiltonian([f1, f2, 2], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)])
        pH_fixed = qml.s_prod(2, qml.PauliZ(2))
        params = [1, 2]
        # Adding on the right
        new_pH = pH + H
        qml.assert_equal(pH.H_fixed(), pH_fixed)
        qml.assert_equal(new_pH.H_fixed(), sum((pH_fixed, qml.s_prod(coeff, qml.PauliZ(0)))))
        assert new_pH.coeffs_fixed == [2, coeff]
        assert qml.math.allclose(
            new_pH(params, t=0.5).matrix(),
            qml.matrix(pH(params, t=0.5) + H, wire_order=new_pH.wires),
        )
        # Adding on the left
        new_pH = H + pH
        qml.assert_equal(pH.H_fixed(), pH_fixed)
        qml.assert_equal(new_pH.H_fixed(), sum((qml.s_prod(coeff, qml.PauliZ(0)), pH_fixed)))
        assert new_pH.coeffs_fixed == [coeff, 2]
        assert qml.math.allclose(
            new_pH(params, t=0.5).matrix(),
            qml.matrix(pH(params, t=0.5) + H, wire_order=new_pH.wires),
        )

    @pytest.mark.parametrize("op", ops)
    def test_add_other_operators(self, op):
        """Test that a Hamiltonian, SProd, Tensor or Operator can be added to a
        ParametrizedHamiltonian, and will be incorporated in the H_fixed term"""
        pH = ParametrizedHamiltonian([f1, f2, 2], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)])
        pH_fixed = qml.s_prod(2, qml.PauliZ(2))

        # Adding on the right
        new_pH = pH + op
        qml.assert_equal(pH.H_fixed(), pH_fixed)
        qml.assert_equal(new_pH.H_fixed(), sum((pH_fixed, qml.s_prod(1, op))))

        # Adding on the left
        new_pH = op + pH
        qml.assert_equal(pH.H_fixed(), pH_fixed)
        qml.assert_equal(new_pH.H_fixed(), sum((qml.s_prod(1, op), pH_fixed)))

    def test_add_invalid_object_raises_error(self):
        """Test that an error is raised when adding a ``ParametrizedHamiltonian`` with an invalid
        object."""
        H = ParametrizedHamiltonian([f1, f2], [qml.PauliX(0), qml.PauliY(1)])

        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = H + object()

        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = object() + H

    def test_multiply_with_scalar(self):
        """Test the __mul__ dunder method with a scalar."""
        H = ParametrizedHamiltonian([f1, f2], [qml.PauliX(0), qml.PauliY(1)])
        new_H = 3 * H
        expected_H = ParametrizedHamiltonian(
            [lambda p, t: 3 * f1(p, t), lambda p, t: 3 * f2(p, t)], [qml.PauliX(0), qml.PauliY(1)]
        )
        assert new_H.H_fixed() == expected_H.H_fixed() == 0
        qml.assert_equal(new_H.H_parametrized([1, 2], t=4), expected_H.H_parametrized([1, 2], t=4))

    def test_multiply_invalid_object_raises_error(self):
        """Test that an error is raised when multiplying a ``ParametrizedHamiltonian`` with an invalid
        object."""
        H = ParametrizedHamiltonian([f1, f2], [qml.PauliX(0), qml.PauliY(1)])

        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = H * object()

        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = object() * H

    def test_adding_two_parametrized_hamiltonians(self):
        """Test that two ParametrizedHamiltonians can be added together and
        the H_fixed and H_parametrized terms are correctly combined."""
        pH1 = ParametrizedHamiltonian([1.2, f2], [qml.PauliX(0), qml.PauliY(1)])
        pH2 = ParametrizedHamiltonian([2.3, f1], [qml.Hadamard(2), qml.PauliZ(3)])

        # H_fixed now contains the fixed terms from both pH1 and pH2
        new_pH = pH1 + pH2
        qml.assert_equal(new_pH.H_fixed()[0], pH1.H_fixed())
        qml.assert_equal(new_pH.H_fixed()[1], pH2.H_fixed())

        # H_parametrized now contained the parametrized terms from both pH1 and pH2
        parametric_term = new_pH.H_parametrized([1.2, 2.3], 0.5)
        qml.assert_equal(parametric_term[0], pH1.H_parametrized([1.2], 0.5))
        qml.assert_equal(parametric_term[1], pH2.H_parametrized([2.3], 0.5))

    def test_fn_times_observable_creates_parametrized_hamiltonian(self):
        """Test a ParametrizedHamiltonian can be created by multiplying a
        function and an Observable"""
        pH = f1 * qml.PauliX(0)
        assert isinstance(pH, ParametrizedHamiltonian)
        assert len(pH.coeffs_fixed) == 0
        assert isinstance(pH.H_parametrized(param, 0.5), qml.ops.SProd)


class TestProperties:
    """Test properties"""

    def test_ops(self):
        """Test stored operator list"""
        ops = test_example.ops
        for op1, op2 in zip(ops, [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2), qml.Hadamard(3)]):
            qml.assert_equal(op1, op2)

    def test_coeffs(self):
        """Test stored coefficients list"""
        coeffs = test_example.coeffs
        assert [1, 2, f1, f2] == coeffs


class TestInterfaces:
    """Test ParametrizedHamiltonian with ML interfaces."""

    @pytest.mark.jax
    def test_call_jax(self):
        """Test result of calling the ParametrizedHamiltonian works with parameters as a jax array"""
        import jax

        pH = ParametrizedHamiltonian([1.2, f1, 2.3, f2], [qml.PauliX(i) for i in range(4)])
        params = jax.numpy.array([4.5, 6.7])
        t = 2
        op = pH(params, t)

        assert isinstance(op, qml.ops.Sum)
        assert len(op) == 2

        H_fixed = op[0]
        H_parametrized = op[1]
        expected_H_fixed = qml.sum(qml.s_prod(1.2, qml.PauliX(0)), qml.s_prod(2.3, qml.PauliX(2)))
        expected_H_parametrized = qml.sum(
            qml.s_prod(f1(params[0], t), qml.PauliX(1)), qml.s_prod(f2(params[1], t), qml.PauliX(3))
        )

        qml.assert_equal(H_fixed, expected_H_fixed)
        qml.assert_equal(H_parametrized, expected_H_parametrized)

    @pytest.mark.torch
    def test_call_torch(self):
        """Test result of calling the ParametrizedHamiltonian works with
        parameters as a torch tensor"""
        import torch

        pH = ParametrizedHamiltonian([1.2, f1, 2.3, f2], [qml.PauliX(i) for i in range(4)])
        params = torch.Tensor([4.5, 6.7])
        t = 2
        op = pH(params, t)

        assert isinstance(op, qml.ops.Sum)
        assert len(op) == 2

        H_fixed = op[0]
        H_parametrized = op[1]
        expected_H_fixed = qml.sum(qml.s_prod(1.2, qml.PauliX(0)), qml.s_prod(2.3, qml.PauliX(2)))
        expected_H_parametrized = qml.sum(
            qml.s_prod(f1(params[0], t), qml.PauliX(1)), qml.s_prod(f2(params[1], t), qml.PauliX(3))
        )

        qml.assert_equal(H_fixed, expected_H_fixed)
        qml.assert_equal(H_parametrized, expected_H_parametrized)

    @pytest.mark.tf
    def test_call_tf(self):
        """Test result of calling the ParametrizedHamiltonian works with
        parameters as a Tensorflow tensor"""
        import tensorflow as tf

        pH = ParametrizedHamiltonian([1.2, f1, 2.3, f2], [qml.PauliX(i) for i in range(4)])
        params = tf.constant([4.5, 6.7])
        t = 2
        op = pH(params, t)

        assert isinstance(op, qml.ops.Sum)
        assert len(op) == 2

        H_fixed = op[0]
        H_parametrized = op[1]
        expected_H_fixed = qml.sum(qml.s_prod(1.2, qml.PauliX(0)), qml.s_prod(2.3, qml.PauliX(2)))
        expected_H_parametrized = qml.sum(
            qml.s_prod(f1(params[0], t), qml.PauliX(1)), qml.s_prod(f2(params[1], t), qml.PauliX(3))
        )

        qml.assert_equal(H_fixed, expected_H_fixed)
        qml.assert_equal(H_parametrized, expected_H_parametrized)
