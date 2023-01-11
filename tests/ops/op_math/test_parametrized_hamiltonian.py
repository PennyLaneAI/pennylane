"""
Unit tests for the ParametrizedHamiltonian class
"""

from inspect import isfunction
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.op_math import ParametrizedHamiltonian


f1 = lambda param, t: param[0] * np.sin(t) * (t - 1)
f2 = lambda param, t: param[1] * np.cos(t**2)
param = [1.2, 2.3]

test_example = ParametrizedHamiltonian(
    [1, 2, f1, f2], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2), qml.Hadamard(3)]
)


class TestInitialization:
    """Test initialization of the ParametrizedHamiltonian class"""

    def test_mismatched_coeffs_and_obs_raises_error(self):
        """Test that an error is raised if the length of the list of coefficients
        and the list of observables don't match"""
        coeffs = [1, 2, f1]
        obs = [qml.PauliX(0), qml.PauliY(1)]
        with pytest.raises(
            ValueError, match="number of coefficients and operators does not match."
        ):
            ParametrizedHamiltonian(coeffs, obs)

    def test_H_fixed_lists(self):
        """Test that attributes H_fixed_ops and H_fixed_coeffs are as expected"""
        assert test_example.H_coeffs_fixed == [1, 2]
        assert np.all(
            [
                qml.equal(op1, op2)
                for op1, op2 in zip(test_example.H_ops_fixed, [qml.PauliX(0), qml.PauliY(1)])
            ]
        )

    def test_H_parametrized_lists(self):
        """Test that attributes H_parametrized_ops and H_parametrized_coeffs are as expected"""
        assert test_example.H_coeffs_parametrized == [f1, f2]
        assert np.all(
            [
                qml.equal(op1, op2)
                for op1, op2 in zip(
                    test_example.H_ops_parametrized, [qml.PauliZ(2), qml.Hadamard(3)]
                )
            ]
        )


class TestCall:

    coeffs_and_ops = (
        (
            [f1, f2],
            [qml.PauliX(0), qml.PauliY(1)],
        ),  # no H_fixed term, multiple H_parametrized terms
        ([f1], [qml.PauliX(0)]),  # no H_fixed term, one H_parametrized term
        (
            [1.2, 2.3],
            [qml.PauliX(0), qml.PauliY(1)],
        ),  # no H_parametrized term, multiple H_fixed terms
        ([1.2], [qml.PauliX(0)]),  # no H_parametrized term, one H_fixed term
        ([1, f1], [qml.PauliX(3), qml.PauliY(2)]),  # one of each
        ([1.2, f1, 2.3, f2], [qml.PauliX(0) for i in range(4)]),  # multiples of each
    )

    @pytest.mark.parametrize("coeffs, ops", coeffs_and_ops)
    def test_call_succeeds_for_different_shapes(self, coeffs, ops):
        """Test that calling the ParametrizedHamiltonian succeeds for
        different numbers of H_fixed and H_parametrized terms"""
        pH = ParametrizedHamiltonian(coeffs, ops)
        pH(param, 0.5)

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
        expected_H_fixed = qml.op_sum(
            qml.s_prod(1.2, qml.PauliX(0)), qml.s_prod(2.3, qml.PauliX(2))
        )
        expected_H_parametrized = qml.op_sum(
            qml.s_prod(f1(params, t), qml.PauliX(1)), qml.s_prod(f2(params, t), qml.PauliX(3))
        )

        assert qml.equal(H_fixed, expected_H_fixed)
        assert qml.equal(H_parametrized, expected_H_parametrized)


class TestInteractionWithOperators:
    """Test that arithmetic operations involving or creating ParametrizedHamiltonians behave as expected"""

    ops = (
        (qml.Hamiltonian([2], [qml.PauliZ(0)]), qml.s_prod(2, qml.PauliZ(0))),
        (qml.ops.SProd(2, qml.PauliZ(0)), qml.s_prod(2, qml.PauliZ(0))),
        (qml.PauliX(2), qml.s_prod(1, qml.PauliX(2))),
        (qml.PauliX(2) @ qml.PauliX(3), qml.s_prod(1, qml.PauliX(2) @ qml.PauliX(3))),
    )

    @pytest.mark.parametrize("H, res", ops)
    def test_add_other_operators(self, H, res):
        """Test that a Hamiltonian, SProd, Tensor or Observable can be added to a
        ParametrizedHamiltonian, and will be incorporated in the H_fixed term"""
        pH = ParametrizedHamiltonian([f1, f2], [qml.PauliX(0), qml.PauliY(1)])
        new_pH = pH + H

        assert pH.H_fixed == 0
        assert qml.equal(new_pH.H_fixed, res)

    def test_adding_two_parametrized_hamiltonians(self):
        """Test that two ParametrizedHamiltonians can be added together and
        the H_fixed and H_parametrized terms are correctly combined."""
        pH1 = ParametrizedHamiltonian([1.2, f2], [qml.PauliX(0), qml.PauliY(1)])
        pH2 = ParametrizedHamiltonian([2.3, f1], [qml.Hadamard(2), qml.PauliZ(3)])

        # H_fixed now contains the fixed terms from both pH1 and pH2
        new_pH = pH1 + pH2
        assert qml.equal(new_pH.H_fixed[0], pH1.H_fixed)
        assert qml.equal(new_pH.H_fixed[1], pH2.H_fixed)

        # H_parametrized now contained the parametrized terms from both pH1 and pH2
        parametric_term = new_pH.H_parametrized([1.2, 2.3], 0.5)
        assert qml.equal(parametric_term[0], pH1.H_parametrized([1.2, 2.3], 0.5))
        assert qml.equal(parametric_term[1], pH2.H_parametrized([1.2, 2.3], 0.5))

    def test_fn_times_observable_creates_parametrized_hamiltonian(self):
        """Test a ParametrizedHamiltonian can be created by multiplying a
        function and an Observable"""
        pH = f1 * qml.PauliX(0)
        assert isinstance(pH, ParametrizedHamiltonian)
        assert len(pH.H_coeffs_fixed) == 0
        assert isinstance(pH.H_parametrized(param, 0.5), qml.ops.SProd)


class TestMiscMethods:
    """Test miscellaneous methods"""

    # pylint: disable=protected-access
    def test_get_terms_for_empty_lists(self):
        """Test that _get_terms can handle being passed
        an empty list, and returns None"""
        op = ParametrizedHamiltonian._get_terms([], [])
        assert op == 0

    # pylint: disable=protected-access
    def test_get_terms_for_single_term_operator(self):
        """Test that _get_terms for a single term operator
        returns a qml.SProd operator"""
        op = ParametrizedHamiltonian._get_terms([2.3], [qml.Hadamard(3)])
        assert qml.equal(op, qml.s_prod(2.3, qml.Hadamard(3)))

    # pylint: disable=protected-access
    def test_get_terms_for_multi_term_operator(self):
        """Test that _get_terms for a single term operator
        returns a qml.Sum of qml.SProd operators"""
        op = ParametrizedHamiltonian._get_terms([2.3, 4.5], [qml.Hadamard(3), qml.PauliX(2)])
        assert qml.equal(
            op, qml.op_sum(qml.s_prod(2.3, qml.Hadamard(3)), qml.s_prod(4.5, qml.PauliX(2)))
        )

    def test__repr__(self):
        """Test repr method returns expected string"""
        str = test_example.__repr__()
        assert str == "ParametrizedHamiltonian: terms=4"


class TestProperties:
    """Test properties"""

    def test_ops(self):
        """Test stored operator list"""
        ops = test_example.ops
        assert np.all(
            [
                qml.equal(op1, op2)
                for op1, op2 in zip(
                    ops, [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2), qml.Hadamard(3)]
                )
            ]
        )

    def test_coeffs(self):
        """Test stored coefficients list"""
        coeffs = test_example.coeffs
        assert [1, 2, f1, f2] == coeffs

    def test_H_fixed(self):
        """Test that H_fixed is an Operator of the expected form"""
        H_fixed = test_example.H_fixed
        op = qml.op_sum(qml.s_prod(1, qml.PauliX(0)), qml.s_prod(2, qml.PauliY(1)))
        assert qml.equal(H_fixed, op)

    def test_H_parametrized(self):
        """Test H_parametrized is a function that, when passed parameters,
        returns an Operator of the expected format."""
        H_param = test_example.H_parametrized
        assert isfunction(H_param)

        H = H_param([1, 2], 0.5)
        assert len(H) == 2
        assert isinstance(H, qml.ops.Sum)
        assert isinstance(H[0], qml.ops.SProd)
        assert isinstance(H[1], qml.ops.SProd)
