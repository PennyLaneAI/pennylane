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

    def test_initialization_via_dot(self):
        """Test that using qml.ops.dot initializes a ParametrizedHamiltonian"""
        f1 = lambda param, t: param[0] * np.sin(t) * (t - 1)
        f2 = lambda param, t: param[1] * np.cos(t**2)

        XX = qml.PauliX(0) @ qml.PauliX(1)
        YY = qml.PauliY(0) @ qml.PauliY(1)
        ZZ = qml.PauliZ(0) @ qml.PauliZ(1)

        coeffs = [2, f1, f2]
        ops = [XX, YY, ZZ]

        H = qml.ops.dot(coeffs, ops)
        expected_H = ParametrizedHamiltonian(coeffs, ops)

        assert qml.equal(H([1.2, 2.3], 3.4), expected_H([1.2, 2.3], 3.4))

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

    def test_call_with_qutrit_operators(self):
        def f1(x, t):
            return x + 2 * t

        coeffs = [f1, 2]
        obs = [qml.GellMann(wires=0, index=1), qml.GellMann(wires=0, index=2)]

        H = ParametrizedHamiltonian(coeffs, obs)

        assert isinstance(H(2, 4), qml.ops.Sum)
        assert (
            H(2, 4).__repr__()
            == "(2*(GellMann(wires=[0], index=2))) + (10*(GellMann(wires=[0], index=1)))"
        )
        assert np.all(
            qml.matrix(H(2, 4))
            == np.array(
                [
                    [0.0 + 0.0j, 10.0 - 2.0j, 0.0 + 0.0j],
                    [10.0 + 2.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ]
            )
        )

    wires = (
        ([0, 1, 2, 3], [10, 12, 2, 7]),  # expected to map 0--> 10, 1 --> 12, 2 --> 2 and 3 --> 7
        (
            ["a", "c", "d", "b"],
            [0, 1, 2, 3],
        ),  # expected to map 'a'--> 0, 'b' --> 1, 'c' --> 2 and 'd' --> 3
        (
            ["a", 2, "b", "c"],
            [2, 10, 11, 12],
        ),  # expected to map 2 --> 2, 'a'--> 10, 'b' --> 11, 'c' --> 12
    )

    @pytest.mark.parametrize("wires1, wires2", wires)
    def test_call_with_wire_mapping(self, wires1, wires2):
        """Test that calling the ParametrizedHamiltonian with the optional
        kwarg wires returns an Operator with wires mapped as expected"""
        coeffs = [1.2, f1, 2.3, f2]
        ops = [
            qml.PauliX(wires1[0]),
            qml.PauliY(wires1[1]),
            qml.PauliZ(wires1[2]),
            qml.Identity(wires1[3]),
        ]
        # note that wires1 are passed to ParametrizedHamiltonian in the order given above, but are
        # sorted when initializing H, so the order when mapping will not necessarily match the order here

        H = ParametrizedHamiltonian(coeffs, ops)
        params = [4.5, 6.7]
        t = 2

        assert np.all(H.wires == np.sort(wires1))

        # get a list of all operators for wire configuration wires1 and wires2
        ops1 = H(params, t).simplify().operands
        ops2 = H(params, t, wires=wires2).simplify().operands

        for w1, w2 in zip(H.wires, wires2):
            # get the operators we expect to correspond in the two wire mappings
            op1 = [op for op in ops1 if op.wires[0] == w1]
            op2 = [op for op in ops2 if op.wires[0] == w2]
            assert len(op1) == len(op2) == 1

            op1 = op1[0]
            op2 = op2[0]

            # operators are not equal unless w1=w2 (in first and last test examples, 2 maps to 2)
            if w1 != w2:
                assert not qml.equal(op1, op2)
            else:
                assert qml.equal(op1, op2)

            # if wire mapping is reversed, the operators are equal, i.e. op1 and op2 match in all aspects except wire
            assert qml.equal(op1, qml.map_wires(op2, {w2: w1}))


class TestInteractionWithOperators:
    """Test that arithmetic operations involving or creating ParametrizedHamiltonians behave as expected"""

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
        """Test that a Hamiltonian and SProd can be added to a ParametrizedHamiltonian, and
        will be incorporated in the H_fixed term, with their coefficients included in H_coeffs_fixed"""
        pH = ParametrizedHamiltonian([f1, f2], [qml.PauliX(0), qml.PauliY(1)])

        # Adding on the right
        new_pH = pH + H
        assert pH.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(coeff, qml.PauliZ(0)))
        assert new_pH.H_coeffs_fixed[0] == coeff

        # Adding on the left
        new_pH = H + pH
        assert pH.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(coeff, qml.PauliZ(0)))
        assert new_pH.H_coeffs_fixed[0] == coeff

    @pytest.mark.parametrize("op", ops)
    def test_add_other_operators(self, op):
        """Test that a Hamiltonian, SProd, Tensor or Operator can be added to a
        ParametrizedHamiltonian, and will be incorporated in the H_fixed term"""
        pH = ParametrizedHamiltonian([f1, f2], [qml.PauliX(0), qml.PauliY(1)])

        # Adding on the right
        new_pH = pH + op
        assert pH.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(1, op))

        # Adding on the left
        new_pH = op + pH
        assert pH.H_fixed() == 0
        assert qml.equal(new_pH.H_fixed(), qml.s_prod(1, op))

    def test_adding_two_parametrized_hamiltonians(self):
        """Test that two ParametrizedHamiltonians can be added together and
        the H_fixed and H_parametrized terms are correctly combined."""
        pH1 = ParametrizedHamiltonian([1.2, f2], [qml.PauliX(0), qml.PauliY(1)])
        pH2 = ParametrizedHamiltonian([2.3, f1], [qml.Hadamard(2), qml.PauliZ(3)])

        # H_fixed now contains the fixed terms from both pH1 and pH2
        new_pH = pH1 + pH2
        assert qml.equal(new_pH.H_fixed()[0], pH1.H_fixed())
        assert qml.equal(new_pH.H_fixed()[1], pH2.H_fixed())

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
        H_fixed = test_example.H_fixed()
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
