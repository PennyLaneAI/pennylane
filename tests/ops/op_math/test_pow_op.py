# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the Pow arithmetic class of qubit operations
"""
from copy import copy

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.exceptions import AdjointUndefinedError, DecompositionUndefinedError
from pennylane.ops.op_math.controlled import ControlledOp
from pennylane.ops.op_math.pow import Pow, PowOperation


# pylint: disable=too-few-public-methods
class TempOperator(qml.operation.Operator):
    """Dummy operator"""

    num_wires = 1


# pylint: disable=unused-argument
def pow_using_dunder_method(base, z, id=None):
    """Helper function which computes the base raised to the power invoking the __pow__ dunder
    method."""
    return base**z


def test_basic_validity():
    """Run basic operator validity checks."""
    op = qml.pow(qml.RX(1.2, wires=0), 3)
    qml.ops.functions.assert_valid(op)

    op = qml.pow(qml.PauliX(0), 2.5)
    qml.ops.functions.assert_valid(op)

    op = qml.pow(qml.Hermitian(np.eye(2), 0), 2)
    qml.ops.functions.assert_valid(op, skip_new_decomp=True)


class TestConstructor:
    def test_lazy_mode(self):
        """Test that by default, the operator is simply wrapped in `Pow`, even if a simplification exists."""

        op = qml.pow(qml.PauliX(0), 2)
        assert isinstance(op, Pow)
        assert op.z == 2
        qml.assert_equal(op.base, qml.PauliX(0))

    def test_nonlazy_product_expansion(self):
        """Test that nonlazy pow returns an expanded product of operators"""

        op = TempOperator(0)
        op_pow = qml.pow(op, 2, lazy=False)
        op_prod = qml.prod(op, op)
        qml.assert_equal(op_pow, op_prod)

    @pytest.mark.parametrize("op", (qml.PauliX(0), qml.CNOT((0, 1))))
    def test_nonlazy_identity_simplification(self, op):
        """Test that nonlazy pow returns a single identity if the power decomposes
        to the identity."""

        op_new = qml.pow(op, 2, lazy=False)
        qml.assert_equal(op_new, qml.Identity(op.wires))

    def test_simplify_with_pow_not_defined(self):
        """Test the simplify method with an operator that has not defined the op.pow method."""
        op = Pow(qml.U2(1, 1, 0), z=1.23)
        simplified_op = op.simplify()
        qml.assert_equal(simplified_op, op)

    def test_simplification_multiple_ops(self):
        """Test that when the simplification method returns a list of multiple operators,
        pow returns a list of multiple operators."""

        # pylint: disable=too-few-public-methods
        class Temp(qml.operation.Operator):
            num_wires = 1

            def pow(self, z):  # pylint: disable=unused-argument
                return [qml.S(0), qml.T(0)]

        new_op = qml.pow(Temp(0), 2, lazy=False)
        assert isinstance(new_op, qml.ops.Prod)  # pylint:disable=no-member
        qml.assert_equal(new_op.operands[0], qml.S(0))
        qml.assert_equal(new_op.operands[1], qml.T(0))

    def test_nonlazy_simplification_queueing(self):
        """Test that if a simpification is accomplished, the metadata for the original op
        and the new simplified op is updated."""

        with qml.queuing.AnnotatedQueue() as q:
            original_op = qml.PauliX(0)
            _ = qml.pow(original_op, 0.5, lazy=False)

        assert original_op not in q.queue

    def test_simplify_squared(self):
        """Test that an op without a special pow method can still be simplified when raised to an integer power."""

        class DummyOp(qml.operation.Operator):
            pass

        simplified = (DummyOp(0) ** 2).simplify()
        qml.assert_equal(simplified, DummyOp(0) @ DummyOp(0))


@pytest.mark.parametrize("power_method", [Pow, pow_using_dunder_method, qml.pow])
class TestInheritanceMixins:
    """Test the inheritance structure and mixin addition through dynamic __new__ method."""

    def test_plain_operator(self, power_method):
        """Test when base directly inherits from Operator only inherits from Operator."""

        base = TempOperator(1.234, wires=0)
        op: Pow = power_method(base=base, z=1.2)

        assert isinstance(op, Pow)
        assert isinstance(op, qml.operation.Operator)
        assert not isinstance(op, qml.operation.Operation)
        assert not isinstance(op, PowOperation)

        # checking we can call `dir` without problems
        assert "num_params" in dir(op)

    def test_operation(self, power_method):
        """When the operation inherits from `Operation`, the `PowOperation` mixin should
        be added and the Pow should now have Operation functionality."""

        class CustomOp(qml.operation.Operation):
            num_wires = 1
            num_params = 1

        base = CustomOp(1.234, wires=0)
        op: Pow = power_method(base=base, z=6.5)

        assert isinstance(op, Pow)
        assert isinstance(op, qml.operation.Operator)
        assert isinstance(op, qml.operation.Operation)
        assert isinstance(op, PowOperation)

        # check operation-specific properties made it into the mapping
        assert "grad_recipe" in dir(op)
        assert "control_wires" in dir(op)


@pytest.mark.parametrize("power_method", [Pow, pow_using_dunder_method, qml.pow])
class TestInitialization:
    """Test the initialization process and standard properties."""

    def test_nonparametric_ops(self, power_method):
        """Test pow initialization for a non parameteric operation."""
        base = qml.PauliX("a")

        op: Pow = power_method(base=base, z=-4.2, id="something")

        assert op.base is base
        assert op.z == -4.2
        assert op.hyperparameters["base"] is base
        assert op.hyperparameters["z"] == -4.2
        assert op.name == "PauliX**-4.2"
        if power_method.__name__ == Pow.__name__:
            assert op.id == "something"

        assert op.num_params == 0
        assert op.parameters == []
        assert op.data == ()

        assert op.wires == qml.wires.Wires("a")
        assert op.num_wires == 1

    def test_parametric_ops(self, power_method):
        """Test pow initialization for a standard parametric operation."""
        params = [1.2345, 2.3456, 3.4567]
        base = qml.Rot(*params, wires="b")

        op: Pow = power_method(base=base, z=-0.766, id="id")

        assert op.base is base
        assert op.z == -0.766
        assert op.hyperparameters["base"] is base
        assert op.hyperparameters["z"] == -0.766
        assert op.name == "Rot**-0.766"
        if power_method.__name__ == Pow.__name__:
            assert op.id == "id"

        assert op.num_params == 3
        assert qml.math.allclose(params, op.parameters)
        assert qml.math.allclose(params, op.data)

        assert op.wires == qml.wires.Wires("b")
        assert op.num_wires == 1

    def test_template_base(self, power_method, seed):
        """Test pow initialization for a template."""
        rng = np.random.default_rng(seed=seed)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        params = rng.random(shape)  # pylint:disable=no-member

        base = qml.StronglyEntanglingLayers(params, wires=[0, 1])
        op: Pow = power_method(base=base, z=2.67)

        assert op.base is base
        assert op.z == 2.67
        assert op.hyperparameters["base"] is base
        assert op.hyperparameters["z"] == 2.67
        assert op.name == "StronglyEntanglingLayers**2.67"

        assert op.num_params == 1
        assert qml.math.allclose(params, op.parameters[0])
        assert qml.math.allclose(params, op.data[0])

        assert op.wires == qml.wires.Wires((0, 1))
        assert op.num_wires == 2


# pylint: disable=too-many-public-methods
@pytest.mark.parametrize("power_method", [Pow, pow_using_dunder_method, qml.pow])
class TestProperties:
    """Test Pow properties."""

    def test_data(self, power_method):
        """Test base data can be get and set through Pow class."""
        x = np.array(1.234)

        base = qml.RX(x, wires="a")
        op: Pow = power_method(base=base, z=3.21)

        assert op.data == (x,)

        # update parameters through pow
        x_new = np.array(2.3456)
        op.data = (x_new,)
        assert base.data == (x_new,)
        assert op.data == (x_new,)

        # update base data updates pow data
        x_new2 = np.array(3.456)
        base.data = (x_new2,)
        assert op.data == (x_new2,)

    def test_has_matrix_true(self, power_method):
        """Test `has_matrix` property carries over when base op defines matrix."""
        base = qml.PauliX(0)
        op: Pow = power_method(base=base, z=-1.1)

        assert op.has_matrix is True

    def test_has_matrix_false(self, power_method):
        """Test has_matrix property carries over when base op does not define a matrix."""

        op: Pow = power_method(base=TempOperator(wires=0), z=2.0)

        assert op.has_matrix is False

    @pytest.mark.parametrize("z", [-2, 3, 2])
    def test_has_adjoint_true(self, z, power_method):
        """Test `has_adjoint` property is true for integer powers."""
        # Note that even if the base would have `base.has_adjoint=False`, `qml.adjoint`
        # would succeed because it would create an `Adjoint(base)` operator.
        base = qml.PauliX(0)
        op: Pow = power_method(base=base, z=z)

        assert op.has_adjoint is True

    @pytest.mark.parametrize("z", [-2.0, 1.0, 0.32])
    def test_has_adjoint_false(self, z, power_method):
        """Test `has_adjoint` property is false for non-integer powers."""
        # Note that the integer power check is a type check, so that floats like 2.
        # are not considered to be integers.

        op: Pow = power_method(base=TempOperator(wires=0), z=z)

        assert op.has_adjoint is False

    @pytest.mark.parametrize("z", [1, 3])
    def test_has_decomposition_true_via_int(self, power_method, z):
        """Test `has_decomposition` property is true if the power is an interger."""
        base = qml.PauliX(0)
        op: Pow = power_method(base=base, z=z)

        assert op.has_decomposition is True

    @pytest.mark.parametrize("z", [1, 3, -0.2, 1.9])
    def test_has_decomposition_true_via_base(self, power_method, z):
        """Test `has_decomposition` property is true if the base operation
        has a working `pow` method, even for non-integer powers."""
        base = qml.RX(0.7, 0)
        op: Pow = power_method(base=base, z=z)

        assert op.has_decomposition is True

    @pytest.mark.parametrize("z", [-0.2, 1.9])
    def test_has_decomposition_false_non_int_no_base_pow(self, power_method, z):
        """Test `has_decomposition` property is false for non-integer powers
        if the base operation does not have a working `pow` method."""
        base = qml.Hadamard(0)
        op: Pow = power_method(base=base, z=z)

        assert op.has_decomposition is False

    def test_no_decomposition_batching_error(self, power_method):
        """Test that if an error occurs with a batched exponent, has_decomposition is False."""

        class MyOp(qml.operation.Operator):

            def pow(self, z):
                return super().pow(z % 2)

        pow_op = power_method(base=MyOp(wires=0), z=np.array([1.0, 2.0]))
        assert not pow_op.has_decomposition

        with pytest.raises(DecompositionUndefinedError):
            pow_op.decomposition()

    def test_error_raised_if_no_batching(self, power_method):
        """Test that if Operator.pow raises an error and no batching is present, the erorr is raised."""

        class MyOp(qml.operation.Operator):

            def pow(self, z):
                raise ValueError

        pow_op = power_method(base=MyOp(0), z=2.5)

        with pytest.raises(ValueError):
            _ = pow_op.has_decomposition

        with pytest.raises(DecompositionUndefinedError):
            _ = pow_op.decomposition()

    @pytest.mark.parametrize("value", (True, False))
    def test_has_diagonalizing_gates(self, value, power_method):
        """Test that Pow defers has_diagonalizing_gates to base operator."""

        class DummyOp(qml.operation.Operator):
            num_wires = 1
            has_diagonalizing_gates = value

        op: Pow = power_method(base=DummyOp("a"), z=2.124)
        assert op.has_diagonalizing_gates is value

    @pytest.mark.parametrize("value", (True, False))
    def test_is_hermitian(self, value, power_method):
        """Test that if the base is hermitian, then the power is hermitian."""

        class DummyOp(qml.operation.Operator):
            """Dummy operator."""

            num_wires = 1
            is_hermitian = value

        op: Pow = power_method(base=DummyOp(1), z=2.5)
        assert op.is_hermitian is value

    def test_queue_category(self, power_method):
        """Test that the queue category `"_ops"` carries over."""
        op: Pow = power_method(base=qml.PauliX(0), z=3.5)
        assert op._queue_category == "_ops"  # pylint: disable=protected-access

    def test_batching_properties(self, power_method):
        """Test the batching properties and methods."""

        # base is batched
        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = power_method(base, z=1)
        assert op.ndim_params == base.ndim_params
        assert op.batch_size == 3

        # coeff is batched
        base = qml.RX(1, 0)
        op = power_method(base, z=np.array([1.2, 2.3, 3.4]))
        assert op.ndim_params == base.ndim_params
        assert op.batch_size == 3

        # both are batched
        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = power_method(base, z=np.array([1.2, 2.3, 3.4]))
        assert op.ndim_params == base.ndim_params
        assert op.batch_size == 3

    def test_different_batch_sizes_raises_error(self, power_method):
        """Test that using different batch sizes for base and scalar raises an error."""
        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = power_method(base, np.array([0.1, 1.2, 2.3, 3.4]))
        with pytest.raises(
            ValueError, match="Broadcasting was attempted but the broadcasted dimensions"
        ):
            _ = op.batch_size

    op_pauli_reps = (
        (qml.PauliZ(wires=0), 1, qml.pauli.PauliSentence({qml.pauli.PauliWord({0: "Z"}): 1})),
        (qml.PauliX(wires=1), 2, qml.pauli.PauliSentence({qml.pauli.PauliWord({}): 1})),  # identity
        (qml.PauliY(wires="a"), 5, qml.pauli.PauliSentence({qml.pauli.PauliWord({"a": "Y"}): 1})),
    )

    @pytest.mark.parametrize("base, exp, rep", op_pauli_reps)
    def test_pauli_rep(self, base, exp, rep, power_method):
        """Test the pauli rep is produced as expected."""
        op = power_method(base, exp)
        assert op.pauli_rep == rep

    def test_pauli_rep_is_none_for_bad_exponents(self, power_method):
        """Test that the _pauli_rep is None if the exponent is not positive or non integer."""
        base = qml.PauliX(wires=0)
        exponents = [1.23, -2]

        for exponent in exponents:
            op = power_method(base, z=exponent)
            assert op.pauli_rep is None

    def test_pauli_rep_none_if_base_pauli_rep_none(self, power_method):
        """Test that None is produced if the base op does not have a pauli rep"""
        base = qml.RX(1.23, wires=0)
        op = power_method(base, z=2)
        assert op.pauli_rep is None

    @pytest.mark.parametrize("z", [-2, 3, 2])
    def test_adjoint_integer_power(self, z, power_method):
        """Test the `adjoint` method for integer powers."""
        base = qml.PauliX(0)
        op: Pow = power_method(base=base, z=z)
        adj_op = op.adjoint()

        assert isinstance(adj_op, Pow)
        assert adj_op.z is op.z
        qml.assert_equal(adj_op.base, qml.ops.Adjoint(qml.X(0)))

    @pytest.mark.parametrize("z", [-2.0, 1.0, 0.32])
    def test_adjoint_non_integer_power_raises(self, z, power_method):
        """Test that the `adjoint` method raises and error for non-integer powers."""

        base = qml.PauliX(0)
        op: Pow = power_method(base=base, z=z)
        with pytest.raises(AdjointUndefinedError, match="The adjoint of Pow operators"):
            _ = op.adjoint()


class TestSimplify:
    """Test Pow simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        pow_op = Pow(base=qml.ops.Adjoint(qml.PauliX(0)), z=2)  # pylint:disable=no-member
        assert pow_op.arithmetic_depth == 2

    def test_simplify_nested_pow_ops(self):
        """Test the simplify method with nested pow operations."""
        pow_op = Pow(base=Pow(base=qml.adjoint(Pow(base=qml.CNOT([1, 0]), z=2)), z=1.2), z=5)
        final_op = qml.Identity([1, 0])
        simplified_op = pow_op.simplify()

        assert isinstance(simplified_op, qml.Identity)
        assert final_op.data == simplified_op.data
        assert final_op.wires == simplified_op.wires
        assert final_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_simplify_zero_power(self):
        """Test that simplifying a matrix raised to the power of 0 returns an Identity matrix."""
        qml.assert_equal(Pow(base=qml.PauliX(0), z=0).simplify(), qml.Identity(0))

    def test_simplify_zero_power_multiple_wires(self):
        """Test that simplifying a multi-wire operator raised to the power of 0 returns a product
        of Identity matrices."""
        pow_op = Pow(base=qml.CNOT([0, 1]), z=0)
        final_op = qml.Identity([0, 1])
        simplified_op = pow_op.simplify()
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        pow_op = Pow(qml.sum(qml.PauliX(0), qml.PauliX(0)) + qml.PauliX(0), 2)
        final_op = qml.s_prod(9, qml.Identity(0))
        simplified_op = pow_op.simplify()
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_method_with_controlled_operation(self):
        """Test simplify method with controlled operation."""
        pow_op = Pow(ControlledOp(base=qml.Hadamard(0), control_wires=1, id=3), z=3)
        final_op = qml.CH([1, 0], id=3)
        simplified_op = pow_op.simplify()
        qml.assert_equal(simplified_op, final_op)


class TestMiscMethods:
    """Test miscellaneous minor Pow methods."""

    def test_repr(self):
        op = Pow(qml.PauliX(0), 2.5)
        assert repr(op) == "X(0)**2.5"

        base = qml.RX(1, 0) + qml.S(1)
        op = Pow(base, 2.5)
        assert repr(op) == "(RX(1, wires=[0]) + S(1))**2.5"

    # pylint: disable=protected-access
    def test_flatten_unflatten(self):
        """Test the _flatten and _unflatten methods."""

        target = qml.S(0)
        z = -0.5
        op = Pow(target, z)
        data, metadata = op._flatten()

        assert len(data) == 2
        assert data[0] is target
        assert data[1] == z

        assert metadata == tuple()

        new_op = type(op)._unflatten(*op._flatten())
        assert new_op is not op
        qml.assert_equal(new_op, op)

    def test_copy(self):
        """Test that a copy of a power operator can have its parameters updated
        independently of the original operator."""
        param1 = 1.2345
        z = 2.3
        base = qml.RX(param1, wires=0)
        op = Pow(base, z)
        copied_op = copy(op)

        assert copied_op.__class__ is op.__class__
        assert copied_op.z == op.z
        assert copied_op.data == (param1,)

        copied_op.data = (6.54,)
        assert op.data == (param1,)

    def test_label(self):
        """Test that the label draws the exponent as superscript."""
        base = qml.RX(1.2, wires=0)
        op = Pow(base, -1.23456789)

        assert op.label() == "RX⁻¹⋅²³⁴⁵⁶⁷⁸⁹"
        assert op.label(decimals=2) == "RX\n(1.20)⁻¹⋅²³⁴⁵⁶⁷⁸⁹"

    def test_label_matrix_param(self):
        """Test that when passed a matrix op, the matrix is cached into passed dictionary."""
        base = qml.QubitUnitary(np.eye(2), wires=0)
        op = Pow(base, -1.2)

        cache = {"matrices": []}
        assert op.label(decimals=2, cache=cache) == "U\n(M0)⁻¹⋅²"
        assert len(cache["matrices"]) == 1

    def test_eigvals(self):
        """Test that the eigenvalues are correct."""
        base = qml.RZ(2.34, wires=0)
        op = Pow(base, 2.5)

        mat_eigvals = qml.math.linalg.eigvals(op.matrix())

        assert qml.math.allclose(mat_eigvals, op.eigvals())

    def test_has_generator_true(self):
        """Test `has_generator` property carries over when base op defines generator."""
        base = qml.RX(0.5, 0)
        op = Pow(base, 0.3)

        assert op.has_generator is True

    def test_has_generator_false(self):
        """Test `has_generator` property carries over when base op does not define a generator."""
        base = qml.PauliX(0)
        op = Pow(base, 0.3)

        assert op.has_generator is False

    def test_generator(self):
        """Test that the generator is the base's generator multiplied by the power."""
        z = 2.5
        base = qml.RX(2.34, wires=0)
        op = Pow(base, z)

        base_gen_op, base_gen_coeff = qml.generator(base, format="prefactor")
        op_gen_op, op_gen_coeff = qml.generator(op, format="prefactor")

        assert qml.math.allclose(base_gen_coeff * z, op_gen_coeff)
        assert base_gen_op.__class__ is op_gen_op.__class__


class TestDiagonalizingGates:
    """Test Pow operators diagonalizing_gates method."""

    @pytest.mark.parametrize("z", (2, -1, 0.25))
    def test_diagonalizing_gates_int_exist(self, z):
        """Test the diagonalizing gates method returns the same thing as the base operator."""

        base = qml.PauliX(0)
        op = Pow(base, z)

        op_gates = op.diagonalizing_gates()
        base_gates = base.diagonalizing_gates()

        assert len(op_gates) == len(base_gates)

        for op1, op2 in zip(op_gates, base_gates):
            assert op1.__class__ is op2.__class__
            assert op1.wires == op2.wires

    def test_base_doesnt_define(self):
        """Test that when the base gate does not define the diagonalizing gates and raises an error,
        that the power operator does as well."""
        base = qml.RX(1.2, wires=0)
        op = Pow(base, 2)

        with pytest.raises(qml.operation.DiagGatesUndefinedError):
            op.diagonalizing_gates()


class TestQueueing:
    """Test that Pow operators queue and update base metadata"""

    def test_queueing(self):
        """Test queuing and metadata when both Pow and base defined inside a recording context."""

        with qml.queuing.AnnotatedQueue() as q:
            base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
            op = Pow(base, 1.2)

        assert q.queue[0] is op
        assert len(q) == 1

    def test_queueing_base_defined_outside(self):
        """Test that base is added to queue even if it's defined outside the recording context."""

        base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
        with qml.queuing.AnnotatedQueue() as q:
            op = Pow(base, 3.4)

        assert len(q) == 1
        assert q.queue[0] is op


class TestMatrix:
    """Test the matrix method for the power operator."""

    def test_base_batching_support(self):
        """Test that Pow matrix has base batching support."""
        x = np.array([-1, -2, -3])
        op = Pow(qml.RX(x, 0), z=3)
        mat = op.matrix()
        true_mat = qml.math.stack([Pow(qml.RX(i, 0), z=3).matrix() for i in x])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    def test_coeff_batching_support(self):
        """Test that Pow matrix has coeff batching support."""
        x = np.array([-1, -2, -3])
        op = Pow(qml.PauliX(0), z=x)
        mat = op.matrix()
        true_mat = qml.math.stack([Pow(qml.PauliX(0), i).matrix() for i in x])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    def test_base_and_coeff_batching_support(self):
        """Test that Pow matrix has base and coeff batching support."""
        x = np.array([-1, -2, -3])
        y = np.array([1, 2, 3])
        op = Pow(qml.RX(x, 0), z=y)
        mat = op.matrix()
        true_mat = qml.math.stack([Pow(qml.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    @pytest.mark.jax
    def test_batching_jax(self):
        """Test that Pow matrix has batching support with the jax interface."""
        import jax.numpy as jnp

        x = jnp.array([-1, -2, -3])
        y = jnp.array([1, 2, 3])
        op = Pow(qml.RX(x, 0), y)
        mat = op.matrix()
        true_mat = qml.math.stack([Pow(qml.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    @pytest.mark.torch
    def test_batching_torch(self):
        """Test that Pow matrix has batching support with the torch interface."""
        import torch

        x = torch.tensor([-1, -2, -3])
        y = torch.tensor([1, 2, 3])
        op = Pow(qml.RX(x, 0), y)
        mat = op.matrix()
        true_mat = qml.math.stack([Pow(qml.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    @pytest.mark.tf
    def test_batching_tf(self):
        """Test that Pow matrix has batching support with the tensorflow interface."""
        import tensorflow as tf

        x = tf.constant([-1.0, -2.0, -3.0])
        y = tf.constant([1.0, 2.0, 3.0])
        op = Pow(qml.RX(x, 0), y)
        mat = op.matrix()
        true_mat = qml.math.stack([Pow(qml.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    def check_matrix(self, param, z):
        """Interface-independent helper function that checks that the matrix of a power op
        of an IsingZZ is the same as the matrix for its decomposition."""
        base = qml.IsingZZ(param, wires=(0, 1))
        op = Pow(base, z)

        mat = op.matrix()
        [shortcut] = base.pow(z)
        shortcut_mat = shortcut.matrix()

        return qml.math.allclose(mat, shortcut_mat)

    @pytest.mark.parametrize("z", (2, -2, 1.23, -0.5))
    def test_matrix_against_shortcut(self, z):
        """Test the matrix method for different exponents and a float parameter."""
        assert self.check_matrix(2.34, z)

    @pytest.mark.autograd
    @pytest.mark.parametrize("z", (2, -2, 1.23, -0.5))
    def test_matrix_against_shortcut_autograd(self, z):
        """Test the matrix using a pennylane numpy array/ autograd numpy."""
        param = qml.numpy.array(2.34)
        assert self.check_matrix(param, z)

    @pytest.mark.jax
    @pytest.mark.parametrize("z", (2, -2, 1.23, -0.5))
    def test_matrix_against_shortcut_jax(self, z):
        """Test the matrix using a jax parameter."""
        from jax import numpy as jnp

        param = jnp.array(2.34)
        assert self.check_matrix(param, z)

    @pytest.mark.torch
    @pytest.mark.parametrize("z", (2, -2, 1.23, -0.5))
    def test_matrix_against_shortcut_torch(self, z):
        """Tests the matrix using a torch tensor parameter."""
        import torch

        param = torch.tensor(2.34)
        assert self.check_matrix(param, z)

    @pytest.mark.tf
    @pytest.mark.parametrize("z", (2, -2, 1.23, -0.5))
    def test_matrix_against_shortcut_tf(self, z):
        """Test the matrix using a tf variable parameter."""
        import tensorflow as tf

        param = tf.Variable(2.34)
        assert self.check_matrix(param, z)

    @pytest.mark.tf
    @pytest.mark.parametrize("z", [-3, -1, 0, 1, 3])
    def test_matrix_tf_int_z(self, z):
        """Test that matrix works with integer power."""
        import tensorflow as tf

        theta = tf.Variable(1.0)
        mat = qml.pow(qml.RX(theta, wires=0), z=z).matrix()
        assert qml.math.allclose(mat, qml.RX.compute_matrix(1.0 * z))

    def test_matrix_wire_order(self):
        """Test that the wire_order keyword rearranges ording."""

        param = 1.234
        z = 3
        base = qml.IsingXX(param, wires=(0, 1))
        op = Pow(base, z)

        compare_op = qml.IsingXX(param * z, wires=(0, 1))

        op_mat = op.matrix(wire_order=(1, 0))
        compare_mat = compare_op.matrix(wire_order=(1, 0))

        assert qml.math.allclose(op_mat, compare_mat)

    def test_pow_hamiltonian(self):
        """Test that a hamiltonian object can be exponentiated."""
        U = qml.Hamiltonian([1.0], [qml.PauliX(wires=0)])
        pow_op = Pow(base=U, z=2)
        mat = pow_op.matrix()

        true_mat = [[1, 0], [0, 1]]
        assert np.allclose(mat, true_mat)


class TestSparseMatrix:
    """Tests involving the sparse matrix method."""

    def test_sparse_matrix_exists_int_exponent(self):
        """Test the sparse matrix is correct when the base defines a
        sparse matrix and the exponennt is an int."""
        from scipy.sparse import csr_matrix

        H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
        H = csr_matrix(H)
        base = qml.SparseHamiltonian(H, wires=0)

        op = Pow(base, 3)

        H_cubed = H**3
        sparse_mat = op.sparse_matrix()
        assert isinstance(sparse_mat, csr_matrix)
        sparse_mat_array = sparse_mat.toarray()

        assert qml.math.allclose(sparse_mat_array, H_cubed.toarray())
        assert qml.math.allclose(sparse_mat_array, op.matrix())

    def test_sparse_matrix_float_exponent(self):
        """Test that even a sparse-matrix defining op raised to a float power
        raises a SparseMatrixUndefinedError error."""
        from scipy.sparse import csr_matrix

        H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
        H = csr_matrix(H)
        base = qml.SparseHamiltonian(H, wires=0)

        op = Pow(base, 0.5)

        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            op.sparse_matrix()

    def test_base_no_sparse_matrix(self):
        """Test that if the base doesn't define a sparse matrix, then the power won't either."""
        op: Pow = Pow(TempOperator(0.1), 2)

        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            op.sparse_matrix()

    def test_sparse_matrix_format(self):
        """Test the sparse matrix is correct when the base defines a
        sparse matrix and the exponennt is an int."""
        from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix

        H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
        H = csr_matrix(H)
        base = qml.SparseHamiltonian(H, wires=0)
        op = Pow(base, 3)

        assert isinstance(op.sparse_matrix(), csr_matrix)
        assert isinstance(op.sparse_matrix(format="csc"), csc_matrix)
        assert isinstance(op.sparse_matrix(format="lil"), lil_matrix)
        assert isinstance(op.sparse_matrix(format="coo"), coo_matrix)


class TestDecompositionExpand:
    """Test the Pow Operator decomposition and expand methods."""

    def test_shortcut_exists(self):
        """Test the decomposition method uses a shortcut if it is defined for that exponent and base."""
        base = qml.PauliX(0)
        op = Pow(base, 0.5)
        decomp = op.decomposition()

        assert len(decomp) == 1
        assert isinstance(decomp[0], qml.SX)
        assert decomp[0].wires == qml.wires.Wires(0)

    def test_shortcut_exists_expand(self):
        """Test that the expand method uses a shortcut if it is defined for that exponent and base."""
        base = qml.PauliX(0)
        op = Pow(base, 0.5)
        expansion_tape = qml.tape.QuantumScript(op.decomposition())

        assert len(expansion_tape) == 1
        assert isinstance(expansion_tape[0], qml.SX)

    def test_positive_integer_exponent(self):
        """Test that decomposition repeats base operator z times if z is a positive integer
        and a shortcut is not defined by op.pow."""
        base = qml.SX(0)
        op = Pow(base, 3)
        decomp = op.decomposition()

        assert len(decomp) == 3

        for op in decomp:
            assert isinstance(op, qml.SX)
            assert op.wires == qml.wires.Wires(0)

    def test_positive_integer_exponent_expand(self):
        """Test that expansion repeats base operator z times if z is a positive integer
        and a shortcut is not defined by op.pow."""

        base = qml.SX(0)
        op = Pow(base, 3)
        expansion_tape = qml.tape.QuantumScript(op.decomposition())

        assert len(expansion_tape) == 3

        for op in expansion_tape:
            assert isinstance(op, qml.SX)
            assert op.wires == qml.wires.Wires(0)

    def test_decomposition_float_power(self):
        """Test that the decomposition raises an error if no shortcut exists and the exponent is a float."""
        base = qml.PauliX(0)
        op = Pow(base, 0.11)

        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()

    def test_decomposition_in_recording_context_with_int_z(self):
        """Tests that decomposition applies ops to a surrounding context."""
        base_ops = [qml.PauliX(0), qml.PauliZ(1)]
        base = qml.prod(*base_ops)
        z = 2
        op = Pow(base, z)

        with qml.queuing.AnnotatedQueue() as q:
            op.decomposition()

        assert len(q.queue) == z
        for applied_op in q.queue:
            qml.assert_equal(applied_op, base)


@pytest.mark.parametrize("power_method", [Pow, pow_using_dunder_method, qml.pow])
class TestOperationProperties:
    """Test Operation specific properties."""

    def test_basis(self, power_method):
        """Test that the basis attribute is the same as the base op's basis attribute."""
        base = qml.RX(1.2, wires=0)
        op: Pow = power_method(base, 2.1)

        assert base.basis == op.basis

    def test_control_wires(self, power_method):
        """Test that the control wires of a Pow operator are the same as the control wires of the base op."""

        base = qml.Toffoli(wires=(0, 1, 2))
        op: Pow = power_method(base, 3.5)

        assert base.control_wires == op.control_wires


class TestIntegration:
    """Test the execution of power gates in a QNode."""

    @pytest.mark.parametrize(
        "diff_method", ("parameter-shift", "finite-diff", "adjoint", "backprop")
    )
    @pytest.mark.parametrize("z", (2, -2, 0.5))
    def test_gradient_pow_rx(self, diff_method, z):
        """Test execution and gradients for a decomposable power operator."""

        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method)
        def circuit(x, z):
            Pow(base=qml.RX(x, wires=0), z=z)
            return qml.expval(qml.PauliY(0))

        x = qml.numpy.array(1.234, requires_grad=True)

        expected = -np.sin(x * z)
        assert qml.math.allclose(circuit(x, z), expected)

        grad = qml.grad(circuit)(x, z)
        expected_grad = -z * np.cos(x * z)
        assert qml.math.allclose(grad, expected_grad)

    def test_batching_execution(self):
        """Test Pow execution with batched base gate parameters."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x):
            Pow(qml.RX(x, wires=0), 2.5)
            return qml.expval(qml.PauliY(0))

        x = qml.numpy.array([1.234, 2.34, 3.456])
        res = circuit(x)

        expected = -np.sin(x * 2.5)
        assert qml.math.allclose(res, expected)

    def test_non_decomposable_power(self):
        """Test execution of a pow operator that cannot be decomposed."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circ():
            Pow(qml.SX(0), -1)
            return qml.state()

        circ()

    @pytest.mark.tf
    @pytest.mark.parametrize("z", [-3, -1, 0, 1, -3])
    @pytest.mark.parametrize("diff_method", ["adjoint", "backprop", "best"])
    def test_ctrl_grad_int_z_tf(self, z, diff_method):
        """Test that controlling a Pow op is differentiable with integer exponents."""
        import tensorflow as tf

        dev = qml.device("default.qubit")

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.Hadamard(0)
            qml.ctrl(Pow(qml.RX(x, wires=1), z=z), control=0)
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(dev)
        def expected_circuit(x):
            qml.Hadamard(0)
            qml.CRX(x * z, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        x = tf.Variable(1.23)

        with tf.GradientTape() as res_tape:
            res = circuit(x)
        res_grad = res_tape.gradient(res, x)

        with tf.GradientTape() as expected_tape:
            expected = expected_circuit(x)
        expected_grad = expected_tape.gradient(expected, x)

        assert np.allclose(res, expected)
        assert np.allclose(res_grad, expected_grad)
