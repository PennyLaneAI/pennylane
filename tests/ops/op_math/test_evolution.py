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
"""Unit tests for the ``Evolution`` operator."""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.ops.op_math import Evolution, Exp


def test_basic_validity():
    """Assert the basic validity of an evolution op."""
    base = qml.prod(qml.PauliX(0), qml.PauliY(1))
    op = Evolution(base, 5.2)
    qml.ops.functions.assert_valid(op)


class TestEvolution:
    """Test Evolution(Exp) class that takes a parameter x and a generator G and defines an evolution exp(ixG)"""

    def test_initialization(self):
        """Test initialization with a provided coefficient and a Tensor base."""
        base = qml.PauliZ("b") @ qml.PauliZ("c")
        param = 1.23

        op = Evolution(base, param)

        assert op.base is base
        assert op.coeff == -1j * param
        assert op.name == "Evolution"
        assert isinstance(op, Exp)

        assert op.num_params == 1
        assert op.parameters == [param]
        assert op.data == (param,)

        assert op.wires == qml.wires.Wires(("b", "c"))

    def test_evolution_matches_corresponding_exp(self):
        base_op = 2 * qml.PauliX(0)
        op1 = Exp(base_op, 1j)
        op2 = Evolution(base_op, -1)

        assert np.all(op1.matrix() == op2.matrix())

    def test_has_generator_true(self):
        """Test that has_generator returns True if the coefficient is purely imaginary."""
        U = Evolution(qml.PauliX(0), 3)
        assert U.has_generator is True

    def test_has_generator_false(self):
        """Test that has_generator returns False if the coefficient is not purely imaginary."""
        U = Evolution(qml.PauliX(0), 3j)
        assert U.has_generator is False

        U = Evolution(qml.PauliX(0), 0.01 + 2j)
        assert U.has_generator is False

    def test_generator(self):
        U = Evolution(qml.PauliX(0), 3)
        assert U.generator() == -1 * U.base

    def test_num_params_for_parametric_base(self):
        base_op = 0.5 * qml.PauliY(0) + qml.PauliZ(0) @ qml.PauliX(1)
        op = Evolution(base_op, 1.23)

        assert base_op.num_params == 1
        assert op.num_params == 1

    def test_data(self):
        """Test initializing and accessing the data property."""

        param = np.array(1.234)

        base = qml.PauliX(0)
        op = Evolution(base, param)

        assert op.data == (param,)
        assert op.coeff == -1j * op.data[0]
        assert op.param == op.data[0]

        new_param = np.array(2.345)
        op.data = (new_param,)

        assert op.data == (new_param,)
        assert op.coeff == -1j * op.data[0]
        assert op.data == op.data[0]

    def test_repr_paulix(self):
        """Test the __repr__ method when the base is a simple observable."""
        op = Evolution(qml.PauliX(0), 3)
        assert repr(op) == "Evolution(-3j PauliX)"

    def test_repr_tensor(self):
        """Test the __repr__ method when the base is a tensor."""
        t = qml.PauliX(0) @ qml.PauliX(1)
        isingxx = Evolution(t, 0.25)

        assert repr(isingxx) == "Evolution(-0.25j X(0) @ X(1))"

    def test_repr_deep_operator(self):
        """Test the __repr__ method when the base is any operator with arithmetic depth > 0."""
        base = qml.S(0) @ qml.X(0)
        op = Evolution(base, 3)

        assert repr(op) == "Evolution(-3j S(0) @ X(0))"

    @pytest.mark.parametrize(
        "op,decimals,expected",
        [
            (Evolution(qml.PauliZ(0), 2), None, "Exp(-2j Z)"),
            (Evolution(qml.PauliZ(0), 2), 2, "Exp(-2.00j Z)"),
            (Evolution(qml.prod(qml.PauliZ(0), qml.PauliY(1)), 2), None, "Exp(-2j Z@Y)"),
            (Evolution(qml.prod(qml.PauliZ(0), qml.PauliY(1)), 2), 2, "Exp(-2.00j Z@Y)"),
            (Evolution(qml.RZ(1.234, wires=[0]), 5.678), None, "Exp(-5.678j RZ)"),
            (Evolution(qml.RZ(1.234, wires=[0]), 5.678), 2, "Exp(-5.68j RZ\n(1.23))"),
        ],
    )
    def test_label(self, op, decimals, expected):
        """Test that the label is informative and uses decimals."""
        assert op.label(decimals=decimals) == expected

    def test_simplify(self):
        """Test that the simplify method simplifies the base."""
        orig_base = qml.adjoint(qml.adjoint(qml.PauliX(0)))

        op = Exp(orig_base, coeff=0.2)
        new_op = op.simplify()
        qml.assert_equal(new_op.base, qml.PauliX(0))
        assert new_op.coeff == 0.2

    def test_simplify_s_prod(self):
        """Tests that when simplification of the base results in an SProd,
        the scalar is included in the coeff rather than the base"""
        base = qml.s_prod(2, qml.sum(qml.PauliX(0), qml.PauliX(0)))
        op = Evolution(base, 3)
        new_op = op.simplify()

        qml.assert_equal(new_op.base, qml.PauliX(0))
        assert new_op.coeff == -12j

    @pytest.mark.jax
    def test_parameter_shift_gradient_matches_jax(self):
        import jax

        dev = qml.device("default.qubit", wires=2)
        base = qml.PauliX(0)
        x = np.array(1.234)

        @qml.qnode(dev, diff_method=qml.gradients.param_shift)
        def circ_param_shift(x):
            Evolution(base, -0.5 * x)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("default.qubit"), interface="jax")
        def circ(x):
            Evolution(qml.PauliX(0), -0.5 * x)
            return qml.expval(qml.PauliZ(0))

        grad_param_shift = qml.grad(circ_param_shift)(x)
        grad = jax.grad(circ)(x)

        assert qml.math.allclose(grad, grad_param_shift)

    def test_generator_warns_if_not_hermitian(self):
        base = qml.s_prod(1j, qml.Identity(0))
        op = Evolution(base, 2)
        with pytest.warns(UserWarning, match="may not be hermitian"):
            op.generator()

    def test_simplifying_Evolution_operator(self):
        base = qml.PauliX(0) + qml.PauliX(1) + qml.PauliX(0)
        op = Evolution(base, 2)

        qml.assert_equal(op.simplify(), Evolution(base.simplify(), 2))

    @pytest.mark.parametrize(
        "base",
        [
            qml.pow(qml.PauliX(0) + qml.PauliY(1)),
            qml.adjoint(qml.PauliZ(2)),
            qml.s_prod(0.5, qml.PauliX(0)),
        ],
    )
    def test_generator_not_observable_class(self, base):
        """Test that qml.generator will return generator if it is_hermitian, but is not a subclass of Observable"""
        op = Evolution(base, 1)
        gen, c = qml.generator(op)
        qml.assert_equal(gen if c == 1 else qml.s_prod(c, gen), qml.s_prod(-1, base))

    def test_generator_error_if_not_hermitian(self):
        """Tests that an error is raised if the generator is not hermitian."""
        op = Evolution(qml.RX(np.pi / 3, 0), 1)

        with pytest.raises(QuantumFunctionError, match="of operation Evolution is not hermitian"):
            qml.generator(op)

    def test_generator_undefined_error(self):
        """Tests that an error is raised if the generator of an Evolution operator is requested
        with a non-zero complex term in the operator parameter."""
        param = 1 + 2.5j
        op = Evolution(qml.PauliZ(0), param)

        with pytest.raises(
            qml.operation.GeneratorUndefinedError,
            match="is not imaginary; the expected format is exp",
        ):
            _ = op.generator()

    def test_pow_is_evolution(self):
        """Test that Evolution raised to a pow is another Evolution."""

        op = Evolution(qml.Z(0), -0.5)

        pow_op = op.pow(2.5)
        qml.assert_equal(pow_op, Evolution(qml.Z(0), -0.5 * 2.5))
        assert type(pow_op) == Evolution  # pylint: disable=unidiomatic-typecheck
