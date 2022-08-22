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
"""Tests for the Adjoint operator wrapper."""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.op_math import Adjoint
from pennylane.ops.op_math.adjoint_class import AdjointOperation


class TestInheritanceMixins:
    """Test inheritance structure and mixin addition through dynamic __new__ method."""

    def test_plain_operator(self):
        """Test when base directly inherits from Operator, Adjoint only inherits
        from Adjoint and Operator."""

        class Tester(qml.operation.Operator):
            num_wires = 1

        base = Tester(1.234, wires=0)
        op = Adjoint(base)

        assert isinstance(op, Adjoint)
        assert isinstance(op, qml.operation.Operator)
        assert not isinstance(op, qml.operation.Operation)
        assert not isinstance(op, qml.operation.Observable)
        assert not isinstance(op, AdjointOperation)

        # checking we can call `dir` without problems
        assert "num_params" in dir(op)

    def test_operation(self):
        """When the operation inherits from `Operation`, the `AdjointOperation` mixin is
        added and the Adjoint has Operation functionality."""

        class CustomOp(qml.operation.Operation):
            num_wires = 1
            num_params = 1

        base = CustomOp(1.234, wires=0)
        op = Adjoint(base)

        assert isinstance(op, Adjoint)
        assert isinstance(op, qml.operation.Operator)
        assert isinstance(op, qml.operation.Operation)
        assert not isinstance(op, qml.operation.Observable)
        assert isinstance(op, AdjointOperation)

        # check operation-specific properties made it into the mapping
        assert "grad_recipe" in dir(op)
        assert "control_wires" in dir(op)

    def test_observable(self):
        """Test that when the base is an Observable, Adjoint will also inherit from Observable."""

        class CustomObs(qml.operation.Observable):
            num_wires = 1
            num_params = 0

        base = CustomObs(wires=0)
        ob = Adjoint(base)

        assert isinstance(ob, Adjoint)
        assert isinstance(ob, qml.operation.Operator)
        assert not isinstance(ob, qml.operation.Operation)
        assert isinstance(ob, qml.operation.Observable)
        assert not isinstance(ob, AdjointOperation)

        # Check some basic observable functionality
        assert ob.compare(ob)
        assert isinstance(1.0 * ob @ ob, qml.Hamiltonian)

        # check the dir
        assert "return_type" in dir(ob)
        assert "grad_recipe" not in dir(ob)


class TestInitialization:
    """Test the initialization process and standard properties."""

    def test_nonparametric_ops(self):
        """Test adjoint initialization for a non parameteric operation."""
        base = qml.PauliX("a")

        op = Adjoint(base, id="something")

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == "Adjoint(PauliX)"
        assert op.id == "something"

        assert op.num_params == 0
        assert op.parameters == []
        assert op.data == []

        assert op.wires == qml.wires.Wires("a")

    def test_parametric_ops(self):
        """Test adjoint initialization for a standard parametric operation."""
        params = [1.2345, 2.3456, 3.4567]
        base = qml.Rot(*params, wires="b")

        op = Adjoint(base, id="id")

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == "Adjoint(Rot)"
        assert op.id == "id"

        assert op.num_params == 3
        assert qml.math.allclose(params, op.parameters)
        assert qml.math.allclose(params, op.data)

        assert op.wires == qml.wires.Wires("b")

    def test_template_base(self):
        """Test adjoint initialization for a template."""
        rng = np.random.default_rng(seed=42)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        params = rng.random(shape)

        base = qml.StronglyEntanglingLayers(params, wires=[0, 1])
        op = Adjoint(base)

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == "Adjoint(StronglyEntanglingLayers)"

        assert op.num_params == 1
        assert qml.math.allclose(params, op.parameters[0])
        assert qml.math.allclose(params, op.data[0])

        assert op.wires == qml.wires.Wires((0, 1))

    def test_hamiltonian_base(self):
        """Test adjoint initialization for a hamiltonian."""
        base = 2.0 * qml.PauliX(0) @ qml.PauliY(0) + qml.PauliZ("b")

        op = Adjoint(base)

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == "Adjoint(Hamiltonian)"

        assert op.num_params == 2
        assert qml.math.allclose(op.parameters, [2.0, 1.0])
        assert qml.math.allclose(op.data, [2.0, 1.0])

        assert op.wires == qml.wires.Wires([0, "b"])


class TestProperties:
    """Test Adjoint properties."""

    def test_data(self):
        """Test base data can be get and set through Adjoint class."""
        x = np.array(1.234)

        base = qml.RX(x, wires="a")
        adj = Adjoint(base)

        assert adj.data == [x]

        # update parameters through adjoint
        x_new = np.array(2.3456)
        adj.data = [x_new]
        assert base.data == [x_new]
        assert adj.data == [x_new]

        # update base data updates Adjoint data
        x_new2 = np.array(3.456)
        base.data = [x_new2]
        assert adj.data == [x_new2]

    def test_has_matrix_true(self):
        """Test `has_matrix` property carries over when base op defines matrix."""
        base = qml.PauliX(0)
        op = Adjoint(base)

        assert op.has_matrix is True

    def test_has_matrix_false(self):
        """Test has_matrix property carries over when base op does not define a matrix."""
        base = qml.QubitStateVector([1, 0], wires=0)
        op = Adjoint(base)

        assert op.has_matrix is False

    def test_has_decomposition_true_via_base_adjoint(self):
        """Test `has_decomposition` property is activated because the base operation defines an
        `adjoint` method."""
        base = qml.PauliX(0)
        op = Adjoint(base)

        assert op.has_decomposition is True

    def test_has_decomposition_true_via_base_adjoint(self):
        """Test `has_decomposition` property is activated because the base operation defines an
        `adjoint` method."""

        class MyOp(qml.operation.Operation):
            num_wires = 1
            decomposition = lambda self: [qml.RX(0.2, self.wires)]

        base = MyOp(0)
        op = Adjoint(base)

        assert op.has_decomposition is True

    def test_has_decomposition_false(self):
        """Test `has_decomposition` property is not activated if the base neither
        `has_adjoint` nor `has_decomposition`."""
        class MyOp(qml.operation.Operation):
            num_wires = 1

        base = MyOp(0)
        op = Adjoint(base)

        assert op.has_decomposition is False

    def test_has_adjoint_true_always(self):
        """Test `has_adjoint` property to always be true, irrespective of the base."""
        class MyOp(qml.operation.Operation):
            """Operation that does not define `adjoint` and hence has `has_adjoint=False`."""
            num_wires = 1

        base = MyOp(0)
        op = Adjoint(base)

        assert op.has_adjoint is True
        assert op.base.has_adjoint is False

        base = qml.PauliX(0)
        op = Adjoint(base)

        assert op.has_adjoint is True
        assert op.base.has_adjoint is True

    def test_queue_category(self):
        """Test that the queue category `"_ops"` carries over."""
        op = Adjoint(qml.PauliX(0))
        assert op._queue_category == "_ops"

    def test_queue_category_None(self):
        """Test that the queue category `None` for some observables carries over."""
        op = Adjoint(qml.PauliX(0) @ qml.PauliY(1))
        assert op._queue_category is None

    def test_private_wires(self):
        """Test that we can get and set the wires via the private property `_wires`."""
        wire0 = qml.wires.Wires("a")
        base = qml.PauliZ(wire0)
        op = Adjoint(base)

        assert op._wires == base._wires == wire0

        wire1 = qml.wires.Wires(0)
        op._wires = wire1
        assert op._wires == base._wires == wire1

    @pytest.mark.parametrize("value", (True, False))
    def test_is_hermitian(self, value):
        """Test `is_hermitian` property mirrors that of the base."""

        class DummyOp(qml.operation.Operator):
            num_wires = 1
            is_hermitian = value

        op = Adjoint(DummyOp(0))
        assert op.is_hermitian == value


class TestSimplify:
    """Test Adjoint simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        adj_op = Adjoint(Adjoint(qml.RZ(1.32, wires=0)))
        assert adj_op.arithmetic_depth == 2

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        adj_op = Adjoint(Adjoint(Adjoint(qml.RZ(1.32, wires=0))))
        final_op = qml.RZ(-1.32, wires=0)
        simplified_op = adj_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, qml.RZ)
        assert final_op.data == simplified_op.data
        assert final_op.wires == simplified_op.wires
        assert final_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_simplify_adj_of_sums(self):
        """Test that the simplify methods converts an adjoint of sums to a sum of adjoints."""
        adj_op = Adjoint(qml.op_sum(qml.RX(1, 0), qml.RY(1, 0), qml.RZ(1, 0)))
        sum_op = qml.op_sum(qml.RX(-1, 0), qml.RY(-1, 0), qml.RZ(-1, 0))
        simplified_op = adj_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, qml.ops.Sum)
        assert sum_op.data == simplified_op.data
        assert sum_op.wires == simplified_op.wires
        assert sum_op.arithmetic_depth == simplified_op.arithmetic_depth

        for s1, s2 in zip(sum_op.summands, simplified_op.summands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_adj_of_prod(self):
        """Test that the simplify method converts an adjoint of products to a (reverse) product
        of adjoints."""
        adj_op = Adjoint(qml.prod(qml.RX(1, 0), qml.RY(1, 0), qml.RZ(1, 0)))
        final_op = qml.prod(qml.RZ(-1, 0), qml.RY(-1, 0), qml.RX(-1, 0))
        simplified_op = adj_op.simplify()

        assert isinstance(simplified_op, qml.ops.Prod)
        assert final_op.data == simplified_op.data
        assert final_op.wires == simplified_op.wires
        assert final_op.arithmetic_depth == simplified_op.arithmetic_depth

        for s1, s2 in zip(final_op.factors, simplified_op.factors):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_with_adjoint_not_defined(self):
        """Test the simplify method with an operator that has not defined the op.adjoint method."""
        op = Adjoint(qml.T(0))
        simplified_op = op.simplify()
        assert isinstance(simplified_op, Adjoint)
        assert op.data == simplified_op.data
        assert op.wires == simplified_op.wires
        assert op.arithmetic_depth == simplified_op.arithmetic_depth


class TestMiscMethods:
    """Test miscellaneous small methods on the Adjoint class."""

    def test_label(self):
        """Test that the label method for the adjoint class adds a † to the end."""
        base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
        op = Adjoint(base)
        assert op.label(decimals=2) == "Rot\n(1.23,\n2.35,\n3.46)†"

    def test_adjoint_of_adjoint(self):
        """Test that the adjoint of an adjoint is the original operation."""
        base = qml.PauliX(0)
        op = Adjoint(base)

        assert op.adjoint() is base

    def test_diagonalizing_gates(self):
        """Assert that the diagonalizing gates method gives the base's diagonalizing gates."""
        base = qml.Hadamard(0)
        diag_gate = Adjoint(base).diagonalizing_gates()[0]

        assert isinstance(diag_gate, qml.RY)
        assert qml.math.allclose(diag_gate.data[0], -np.pi / 4)


class TestAdjointOperation:
    """Test methods in the AdjointOperation mixin."""

    @pytest.mark.parametrize(
        "base, adjoint_base_name",
        ((qml.PauliX(0), "Adjoint(PauliX)"), (qml.RX(1.2, wires=0), "Adjoint(RX)")),
    )
    def test_base_name(self, base, adjoint_base_name):
        """Test the base_name property of AdjointOperation."""
        op = Adjoint(base)
        assert op.base_name == adjoint_base_name

    def test_generator(self):
        """Assert that the generator of an Adjoint is -1.0 times the base generator."""
        base = qml.RX(1.23, wires=0)
        op = Adjoint(base)

        assert base.generator().compare(-1.0 * op.generator())

    def test_no_generator(self):
        """Test that an adjointed non-Operation raises a GeneratorUndefinedError."""

        with pytest.raises(qml.operation.GeneratorUndefinedError):
            Adjoint(1.0 * qml.PauliX(0)).generator()

    def test_single_qubit_rot_angles(self):

        param = 1.234
        base = qml.RX(param, wires=0)
        op = Adjoint(base)

        base_angles = base.single_qubit_rot_angles()
        angles = op.single_qubit_rot_angles()

        for angle1, angle2 in zip(angles, reversed(base_angles)):
            assert angle1 == -angle2

    @pytest.mark.parametrize(
        "base, basis",
        (
            (qml.RX(1.234, wires=0), "X"),
            (qml.PauliY("a"), "Y"),
            (qml.PhaseShift(4.56, wires="b"), "Z"),
            (qml.SX(-1), "X"),
        ),
    )
    def test_basis_property(self, base, basis):
        op = Adjoint(base)
        assert op.basis == basis

    def test_control_wires(self):
        """Test the control_wires of an adjoint are the same as the base op."""
        op = Adjoint(qml.CNOT(wires=("a", "b")))
        assert op.control_wires == qml.wires.Wires("a")


class TestInverse:
    """Tests involving the inverse attribute."""

    def test_base_inverted(self):
        """Test when base is already inverted."""
        base = qml.S(0).inv()
        op = Adjoint(base)

        assert op.inverse is True
        assert base.inverse is True
        assert op.name == "Adjoint(S.inv)"

        assert qml.math.allclose(qml.matrix(op), qml.matrix(qml.S(0)))

        decomp_adj_inv = op.expand().circuit
        decomp = qml.S(0).expand().circuit

        for op1, op2 in zip(decomp, decomp_adj_inv):
            assert type(op1) == type(op2)
            assert op1.data == op2.data
            assert op1.wires == op2.wires

    def test_inv_method(self):
        """Test that calling inv on an Adjoint op defers to base op."""

        base = qml.T(0)
        op = Adjoint(base)
        op.inv()

        assert base.inverse is True
        assert op.inverse is True
        assert op.name == "Adjoint(T.inv)"

        assert qml.math.allclose(qml.matrix(op), qml.matrix(qml.T(0)))
        decomp_adj_inv = op.expand().circuit
        decomp = qml.T(0).expand().circuit

        for op1, op2 in zip(decomp, decomp_adj_inv):
            assert type(op1) == type(op2)
            assert op1.data == op2.data
            assert op1.wires == op2.wires

    def test_inverse_setter(self):
        """Test the inverse getting updated by property setter."""
        base = qml.T(0)
        op = Adjoint(base)

        assert base.inverse == op.inverse == False
        op.inverse = True

        assert base.inverse == op.inverse == True
        assert op.name == "Adjoint(T.inv)"


class TestAdjointOperationDiffInfo:
    """Test differention related properties and methods of AdjointOperation."""

    def test_grad_method_None(self):
        """Test grad_method copies base grad_method when it is None."""
        base = qml.PauliX(0)
        op = Adjoint(base)

        assert op.grad_method is None

    @pytest.mark.parametrize("op", (qml.RX(1.2, wires=0),))
    def test_grad_method_not_None(self, op):
        """Make sure the grad_method property of a Adjoint op is the same as the base op."""
        assert Adjoint(op).grad_method == op.grad_method

    @pytest.mark.parametrize(
        "base", (qml.PauliX(0), qml.RX(1.234, wires=0), qml.Rotation(1.234, wires=0))
    )
    def test_grad_recipe(self, base):
        """Test that the grad_recipe of the Adjoint is the same as the grad_recipe of the base."""
        assert Adjoint(base).grad_recipe == base.grad_recipe

    def test_get_parameter_shift(self):
        """Test `get_parameter_shift` for an operation where it still doesn't raise warnings and errors."""
        base = qml.Rotation(1.234, wires=0)
        with pytest.warns(UserWarning, match=r"get_parameter_shift is deprecated."):
            assert Adjoint(base).get_parameter_shift(0) == base.get_parameter_shift(0)

    @pytest.mark.parametrize(
        "base",
        (qml.RX(1.23, wires=0), qml.Rot(1.23, 2.345, 3.456, wires=0), qml.CRX(1.234, wires=(0, 1))),
    )
    def test_parameter_frequencies(self, base):
        """Test that the parameter frequencies of an Adjoint are the same as those of the base."""
        assert Adjoint(base).parameter_frequencies == base.parameter_frequencies


class TestQueueing:
    """Test that Adjoint operators queue and update base metadata"""

    def test_queueing(self):
        """Test queuing and metadata when both Adjoint and base defined inside a recording context."""

        with qml.tape.QuantumTape() as tape:
            base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
            op = Adjoint(base)

        assert tape._queue[base]["owner"] is op
        assert tape._queue[op]["owns"] is base
        assert tape.operations == [op]

    def test_queueing_base_defined_outside(self):
        """Test that base isn't added to queue if it's defined outside the recording context."""

        base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
        with qml.tape.QuantumTape() as tape:
            op = Adjoint(base)

        assert len(tape._queue) == 1
        assert tape._queue[op]["owns"] is base
        assert tape.operations == [op]

    def test_do_queue_False(self):
        """Test that when `do_queue` is specified, the operation is not queued."""
        base = qml.PauliX(0)
        with qml.tape.QuantumTape() as tape:
            op = Adjoint(base, do_queue=False)

        assert len(tape) == 0


class TestMatrix:
    """Test the matrix method for a variety of interfaces."""

    def test_batching_error(self):
        """Test that a MatrixUndefinedError is raised if the base is batched."""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        base = qml.RX(x, wires=0)
        op = Adjoint(base)

        with pytest.raises(qml.operation.MatrixUndefinedError):
            op.matrix()

    def check_matrix(self, x, interface):
        """Compares matrices in a interface independent manner."""
        base = qml.RX(x, wires=0)
        base_matrix = base.matrix()
        expected = qml.math.conj(qml.math.transpose(base_matrix))

        mat = Adjoint(base).matrix()

        assert qml.math.allclose(expected, mat)
        assert qml.math.get_interface(mat) == interface

    @pytest.mark.autograd
    def test_matrix_autograd(self):
        """Test the matrix of an Adjoint operator with an autograd parameter."""
        self.check_matrix(np.array(1.2345), "autograd")

    @pytest.mark.jax
    def test_matrix_jax(self):
        """Test the matrix of an adjoint operator with a jax parameter."""
        import jax.numpy as jnp

        self.check_matrix(jnp.array(1.2345), "jax")

    @pytest.mark.torch
    def test_matrix_torch(self):
        """Test the matrix of an adjoint oeprator with a torch parameter."""
        import torch

        self.check_matrix(torch.tensor(1.2345), "torch")

    @pytest.mark.tf
    def test_matrix_tf(self):
        """Test the matrix of an adjoint opreator with a tensorflow parameter."""
        import tensorflow as tf

        self.check_matrix(tf.Variable(1.2345), "tensorflow")

    def test_no_matrix_defined(self):
        """Test that if the base has no matrix defined, then Adjoint.matrix also raises a MatrixUndefinedError."""
        rng = np.random.default_rng(seed=42)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        params = rng.random(shape)

        base = qml.StronglyEntanglingLayers(params, wires=[0, 1])

        with pytest.raises(qml.operation.MatrixUndefinedError):
            Adjoint(base).matrix()

    def test_adj_hamiltonian(self):
        """Test that a we can take the adjoint of a hamiltonian."""
        U = qml.Hamiltonian([1.0], [qml.PauliX(wires=0) @ qml.PauliZ(wires=1)])
        adj_op = Adjoint(base=U)  # hamiltonian = hermitian = self-adjoint
        mat = adj_op.matrix()

        true_mat = qml.matrix(U)
        assert np.allclose(mat, true_mat)


def test_sparse_matrix():
    """Test that the spare_matrix method returns the adjoint of the base sparse matrix."""
    from scipy.sparse import csr_matrix

    H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
    H = csr_matrix(H)
    base = qml.SparseHamiltonian(H, wires=0)

    op = Adjoint(base)

    base_sparse_mat = base.sparse_matrix()
    base_conj_T = qml.numpy.conj(qml.numpy.transpose(base_sparse_mat))
    op_sparse_mat = op.sparse_matrix()

    assert isinstance(op_sparse_mat, csr_matrix)

    assert qml.math.allclose(base_conj_T.toarray(), op_sparse_mat.toarray())


class TestEigvals:
    """Test the Adjoint class adjoint methods."""

    @pytest.mark.parametrize(
        "base", (qml.PauliX(0), qml.Hermitian(np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]]), wires=0))
    )
    def test_hermitian_eigvals(self, base):
        """Test adjoint's eigvals are the same as base eigvals when op is Hermitian."""
        base_eigvals = base.eigvals()
        adj_eigvals = Adjoint(base).eigvals()
        assert qml.math.allclose(base_eigvals, adj_eigvals)

    def test_non_hermitian_eigvals(self):
        """Test that the Adjoint eigvals are the conjugate of the base's eigvals."""

        base = qml.SX(0)
        base_eigvals = base.eigvals()
        adj_eigvals = Adjoint(base).eigvals()

        assert qml.math.allclose(qml.math.conj(base_eigvals), adj_eigvals)

    def test_no_matrix_defined_eigvals(self):
        """Test that if the base does not define eigvals, The Adjoint raises the same error."""
        base = qml.QubitStateVector([1, 0], wires=0)

        with pytest.raises(qml.operation.EigvalsUndefinedError):
            Adjoint(base).eigvals()


class TestDecompositionExpand:
    """Test the decomposition and expand methods for the Adjoint class."""

    def test_decomp_custom_adjoint_defined(self):
        """Test decomposition method when a custom adjoint is defined."""
        decomp = Adjoint(qml.Hadamard(0)).decomposition()
        assert len(decomp) == 1
        assert isinstance(decomp[0], qml.Hadamard)

    def test_expand_custom_adjoint_defined(self):
        """Test expansion method when a custom adjoint is defined."""
        base = qml.Hadamard(0)
        tape = Adjoint(base).expand()

        assert len(tape) == 1
        assert isinstance(tape[0], qml.Hadamard)

    def test_decomp(self):
        """Test decomposition when base has decomposition but no custom adjoint."""
        base = qml.SX(0)
        base_decomp = base.decomposition()
        decomp = Adjoint(base).decomposition()

        for adj_op, base_op in zip(decomp, reversed(base_decomp)):
            assert isinstance(adj_op, Adjoint)
            assert adj_op.base.__class__ == base_op.__class__
            assert qml.math.allclose(adj_op.data, base_op.data)

    def test_expand(self):
        """Test expansion when base has decomposition but no custom adjoint."""

        base = qml.SX(0)
        base_tape = base.expand()
        tape = Adjoint(base).expand()

        for base_op, adj_op in zip(reversed(base_tape), tape):
            assert isinstance(adj_op, Adjoint)
            assert base_op.__class__ == adj_op.base.__class__
            assert qml.math.allclose(adj_op.data, base_op.data)

    def test_no_base_gate_decomposition(self):
        """Test that when the base gate doesn't have a decomposition, the Adjoint decomposition
        method raises the proper error."""
        nr_wires = 2
        rho = np.zeros((2**nr_wires, 2**nr_wires), dtype=np.complex128)
        rho[0, 0] = 1  # initialize the pure state density matrix for the |0><0| state
        base = qml.QubitDensityMatrix(rho, wires=(0, 1))

        with pytest.raises(qml.operation.DecompositionUndefinedError):
            Adjoint(base).decomposition()

    def test_adjoint_of_adjoint(self):
        """Test that the adjoint an adjoint returns the base operator through both decomposition and expand."""

        base = qml.PauliX(0)
        adj1 = Adjoint(base)
        adj2 = Adjoint(adj1)

        assert adj2.decomposition()[0] is base

        tape = adj2.expand()
        assert tape.circuit[0] is base


class TestIntegration:
    """Test the integration of the Adjoint class with qnodes and gradients."""

    @pytest.mark.parametrize(
        "diff_method", ("parameter-shift", "finite-diff", "adjoint", "backprop")
    )
    def test_gradient_adj_rx(self, diff_method):
        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method)
        def circuit(x):
            Adjoint(qml.RX(x, wires=0))
            return qml.expval(qml.PauliY(0))

        x = np.array(1.2345, requires_grad=True)

        res = circuit(x)
        expected = np.sin(x)
        assert qml.math.allclose(res, expected)

        grad = qml.grad(circuit)(x)
        expected_grad = np.cos(x)

        assert qml.math.allclose(grad, expected_grad)

    def test_adj_batching(self):
        """Test execution of the adjoint of an operation with batched parameters."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x):
            Adjoint(qml.RX(x, wires=0))
            return qml.expval(qml.PauliY(0))

        x = qml.numpy.array([1.234, 2.34, 3.456])
        res = circuit(x)

        expected = np.sin(x)
        assert qml.math.allclose(res, expected)
