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
from pennylane.operation import DecompositionUndefinedError
from pennylane.ops.op_math.controlled_class import ControlledOp
from pennylane.ops.op_math.pow_class import Pow, PowOperation


class TempOperator(qml.operation.Operator):
    """Dummy operator"""

    num_wires = 1


def pow_using_dunder_method(base, z, do_queue=True, id=None):
    """Helper function which computes the base raised to the power invoking the __pow__ dunder
    method."""
    return base**z


@pytest.mark.parametrize("power_method", [Pow, pow_using_dunder_method])
class TestInheritanceMixins:
    """Test the inheritance structure and mixin addition through dynamic __new__ method."""

    def test_plain_operator(self, power_method):
        """Test when base directly inherits from Operator only inherits from Operator."""

        base = TempOperator(1.234, wires=0)
        op: Pow = power_method(base=base, z=1.2)

        assert isinstance(op, Pow)
        assert isinstance(op, qml.operation.Operator)
        assert not isinstance(op, qml.operation.Operation)
        assert not isinstance(op, qml.operation.Observable)
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
        assert not isinstance(op, qml.operation.Observable)
        assert isinstance(op, PowOperation)

        # check operation-specific properties made it into the mapping
        assert "grad_recipe" in dir(op)
        assert "control_wires" in dir(op)

    def test_observable(self, power_method):
        """Test that when the base is an Observable, Adjoint will also inherit from Observable."""

        class CustomObs(qml.operation.Observable):
            num_wires = 1
            num_params = 0

        base = CustomObs(wires=0)
        ob: Pow = power_method(base=base, z=-1.2)

        assert isinstance(ob, Pow)
        assert isinstance(ob, qml.operation.Operator)
        assert not isinstance(ob, qml.operation.Operation)
        assert isinstance(ob, qml.operation.Observable)
        assert not isinstance(ob, PowOperation)

        # Check some basic observable functionality
        assert ob.compare(ob)
        assert isinstance(1.0 * ob @ ob, qml.Hamiltonian)

        # check the dir
        assert "return_type" in dir(ob)
        assert "grad_recipe" not in dir(ob)


@pytest.mark.parametrize("power_method", [Pow, pow_using_dunder_method])
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
        assert op.data == []

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

    def test_template_base(self, power_method):
        """Test pow initialization for a template."""
        rng = np.random.default_rng(seed=42)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        params = rng.random(shape)

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

    def test_hamiltonian_base(self, power_method):
        """Test pow initialization for a hamiltonian."""
        base = 2.0 * qml.PauliX(0) @ qml.PauliY(0) + qml.PauliZ("b")

        op: Pow = power_method(base=base, z=3.4)

        assert op.base is base
        assert op.z == 3.4
        assert op.hyperparameters["base"] is base
        assert op.hyperparameters["z"] == 3.4
        assert op.name == "Hamiltonian**3.4"

        assert op.num_params == 2
        assert qml.math.allclose(op.parameters, [2.0, 1.0])
        assert qml.math.allclose(op.data, [2.0, 1.0])

        assert op.wires == qml.wires.Wires([0, "b"])
        assert op.num_wires == 2


@pytest.mark.parametrize("power_method", [Pow, pow_using_dunder_method])
class TestProperties:
    """Test Pow properties."""

    def test_data(self, power_method):
        """Test base data can be get and set through Pow class."""
        x = np.array(1.234)

        base = qml.RX(x, wires="a")
        op: Pow = power_method(base=base, z=3.21)

        assert op.data == [x]

        # update parameters through pow
        x_new = np.array(2.3456)
        op.data = [x_new]
        assert base.data == [x_new]
        assert op.data == [x_new]

        # update base data updates pow data
        x_new2 = np.array(3.456)
        base.data = [x_new2]
        assert op.data == [x_new2]

    def test_private_wires_getter_setter(self, power_method):
        """Test that we can get and set the private _wires."""
        wires0 = qml.wires.Wires("a")
        base = qml.PauliZ(wires0)
        op: Pow = power_method(base=base, z=-2.1)

        assert op._wires == base._wires == wires0

        wires1 = qml.wires.Wires(1)
        op._wires = wires1
        assert op._wires == base._wires == wires1

    def test_has_matrix_true(self, power_method):
        """Test `has_matrix` property carries over when base op defines matrix."""
        base = qml.PauliX(0)
        op: Pow = power_method(base=base, z=-1.1)

        assert op.has_matrix

    def test_has_matrix_false(self, power_method):
        """Test has_matrix property carries over when base op does not define a matrix."""

        op: Pow = power_method(base=TempOperator(wires=0), z=2.0)

        assert not op.has_matrix

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
        assert op._queue_category == "_ops"

    def test_queue_category_None(self, power_method):
        """Test that the queue category `None` for some observables carries over."""
        op: Pow = power_method(base=qml.PauliX(0) @ qml.PauliY(1), z=-1.1)
        assert op._queue_category is None

    def test_batching_properties(self, power_method):
        """Test that Pow batching behavior mirrors that of the base."""

        class DummyOp(qml.operation.Operator):
            ndim_params = (0, 2)
            num_wires = 1

        param1 = [0.3] * 3
        param2 = [[[0.3, 1.2]]] * 3

        base = DummyOp(param1, param2, wires=0)
        op: Pow = power_method(base=base, z=3)

        assert op.ndim_params == (0, 2)
        assert op.batch_size == 3


class TestSimplify:
    """Test Pow simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        pow_op = Pow(base=qml.ops.Adjoint(qml.PauliX(0)), z=2)
        assert pow_op.arithmetic_depth == 2

    def test_simplify_nested_pow_ops(self):
        """Test the simplify method with nested pow operations."""
        pow_op = Pow(base=Pow(base=qml.adjoint(Pow(base=qml.CNOT([1, 0]), z=1.2)), z=2), z=5)
        final_op = qml.prod(qml.Identity(1), qml.Identity(0))
        simplified_op = pow_op.simplify()

        assert isinstance(simplified_op, qml.ops.Prod)
        assert final_op.data == simplified_op.data
        assert final_op.wires == simplified_op.wires
        assert final_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_simplify_zero_power(self):
        """Test that simplifying a matrix raised to the power of 0 returns an Identity matrix."""
        assert qml.equal(Pow(base=qml.PauliX(0), z=0).simplify(), qml.Identity(0))

    def test_simplify_zero_power_multiple_wires(self):
        """Test that simplifying a multi-wire operator raised to the power of 0 returns a product
        of Identity matrices."""
        pow_op = Pow(base=qml.CNOT([0, 1]), z=0)
        final_op = qml.prod(qml.Identity(0), qml.Identity(1))
        simplified_op = pow_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, qml.ops.Prod)
        assert final_op.data == simplified_op.data
        assert final_op.wires == simplified_op.wires
        assert final_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        pow_op = Pow(qml.op_sum(qml.PauliX(0), qml.PauliX(0)) + qml.PauliX(0), 2)
        final_op = qml.s_prod(9, qml.PauliX(0))
        simplified_op = pow_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, qml.ops.SProd)
        assert final_op.data == simplified_op.data
        assert final_op.wires == simplified_op.wires
        assert final_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_simplify_method_with_controlled_operation(self):
        """Test simplify method with controlled operation."""
        pow_op = Pow(ControlledOp(base=qml.PauliX(0), control_wires=1, id=3), z=3)
        final_op = ControlledOp(base=qml.PauliX(0), control_wires=1, id=3)
        simplified_op = pow_op.simplify()

        assert isinstance(simplified_op, ControlledOp)
        assert final_op.data == simplified_op.data
        assert final_op.wires == simplified_op.wires
        assert final_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_simplify_with_adjoint_not_defined(self):
        """Test the simplify method with an operator that has not defined the op.pow method."""
        op = Pow(qml.U2(1, 1, 0), z=3)
        simplified_op = op.simplify()
        assert isinstance(simplified_op, Pow)
        assert op.data == simplified_op.data
        assert op.wires == simplified_op.wires
        assert op.arithmetic_depth == simplified_op.arithmetic_depth


class TestMiscMethods:
    """Test miscellaneous minor Pow methods."""

    def test_repr(self):
        op = Pow(qml.PauliX(0), 2.5)
        assert repr(op) == "PauliX(wires=[0])**2.5"

        base = qml.RX(1, 0) + qml.S(1)
        op = Pow(base, 2.5)
        assert repr(op) == "(RX(1, wires=[0]) + S(wires=[1]))**2.5"

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
        assert copied_op.data == [param1]

        copied_op.data = [6.54]
        assert op.data == [param1]

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
        assert op.label(decimals=2, cache=cache) == "U(M0)⁻¹⋅²"
        assert len(cache["matrices"]) == 1

    def test_eigvals(self):
        """Test that the eigenvalues are correct."""
        base = qml.RZ(2.34, wires=0)
        op = Pow(base, 2.5)

        mat_eigvals = qml.math.linalg.eigvals(op.matrix())

        assert qml.math.allclose(mat_eigvals, op.eigvals())

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

        with qml.tape.QuantumTape() as tape:
            base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
            op = Pow(base, 1.2)

        assert tape._queue[base]["owner"] is op
        assert tape._queue[op]["owns"] is base
        assert tape.operations == [op]

    def test_queueing_base_defined_outside(self):
        """Test that base is added to queue even if it's defined outside the recording context."""

        base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
        with qml.tape.QuantumTape() as tape:
            op = Pow(base, 3.4)

        assert len(tape.queue) == 1
        assert tape._queue[op]["owns"] is base
        assert tape.operations == [op]

    def test_do_queue_False(self):
        """Test that when `do_queue` is specified, the operation is not queued."""
        base = qml.PauliX(0)
        with qml.tape.QuantumTape() as tape:
            op = Pow(base, 4.5, do_queue=False)

        assert len(tape) == 0


class TestMatrix:
    """Test the matrix method for the power operator."""

    def check_matrix(self, param, z):
        """Interface-independent helper function that checks that the matrix of a power op
        of an IsingZZ is the same as the matrix for its decomposition."""
        base = qml.IsingZZ(param, wires=(0, 1))
        op = Pow(base, z)

        mat = qml.matrix(op)
        shortcut = base.pow(z)[0]
        shortcut_mat = qml.matrix(shortcut)

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
    def test_matrix_against_shortcut_jax(self, z):
        """Test the matrix using a tf variable parameter."""
        import tensorflow as tf

        param = tf.Variable(2.34)
        assert self.check_matrix(param, z)

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
        assert qml.math.allclose(sparse_mat_array, qml.matrix(op))

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
        expansion_tape = op.expand()

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
        expansion_tape = op.expand()

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


class TestInverse:
    """Test the interaction between in-place inversion and the power operator."""

    def test_base_already_inverted(self):
        """Test that if the base is already inverted, then initialization un-inverts
        it and applies a negative sign to the exponent."""
        with pytest.warns(UserWarning, match="In-place inversion with inverse is deprecated"):
            base = qml.S(0).inv()
        op = Pow(base, 2)

        assert base.inverse is False

        assert op.z == -2
        assert op.name == "S**-2"
        assert op.base_name == "S**-2"
        assert op.inverse is False

    def test_invert_pow_op(self):
        """Test that in-place inversion of a power operator only changes the sign
        of the power and does not change the `inverse` property."""
        base = qml.S(0)
        op = Pow(base, 2)

        with pytest.warns(UserWarning, match="In-place inversion with inv is deprecated"):
            op.inv()

        assert base.inverse is False

        assert op.z == -2
        assert op.name == "S**-2"
        assert op.base_name == "S**-2"
        assert op.inverse is False

    def test_inverse_setter(self):
        """Assert that the inverse can be set to False, but trying to set it to True raises a
        NotImplementedError."""
        op = Pow(qml.S(0), 2.1)

        op.inverse = False

        with pytest.raises(NotImplementedError):
            op.inverse = True


@pytest.mark.parametrize("power_method", [Pow, pow_using_dunder_method])
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
