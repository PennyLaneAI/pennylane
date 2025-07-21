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
"""Tests for the Adjoint operator wrapper and the adjoint constructor."""

import pickle

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.op_math.adjoint import Adjoint, AdjointOperation, adjoint


# pylint: disable=too-few-public-methods
class PlainOperator(qml.operation.Operator):
    """just an operator."""


@pytest.mark.parametrize("target", (qml.PauliZ(0), qml.Rot(1.2, 2.3, 3.4, wires=0)))
def test_basic_validity(target):
    """Run basic operator validity fucntions."""
    op = qml.adjoint(target)
    qml.ops.functions.assert_valid(op)


class TestInheritanceMixins:
    """Test inheritance structure and mixin addition through dynamic __new__ method."""

    def test_plain_operator(self):
        """Test when base directly inherits from Operator, Adjoint only inherits
        from Adjoint and Operator."""

        base = PlainOperator(1.234, wires=0)
        op = Adjoint(base)

        assert isinstance(op, Adjoint)
        assert isinstance(op, qml.operation.Operator)
        assert not isinstance(op, qml.operation.Operation)
        assert not isinstance(op, AdjointOperation)

        # checking we can call `dir` without problems
        assert "num_params" in dir(op)

    def test_operation(self):
        """When the operation inherits from `Operation`, the `AdjointOperation` mixin is
        added and the Adjoint has Operation functionality."""

        # pylint: disable=too-few-public-methods
        class CustomOp(qml.operation.Operation):
            num_wires = 1
            num_params = 1

        base = CustomOp(1.234, wires=0)
        op = Adjoint(base)

        assert isinstance(op, Adjoint)
        assert isinstance(op, qml.operation.Operator)
        assert isinstance(op, qml.operation.Operation)
        assert isinstance(op, AdjointOperation)

        # check operation-specific properties made it into the mapping
        assert "grad_recipe" in dir(op)
        assert "control_wires" in dir(op)

    @pytest.mark.parametrize(
        "op",
        (
            PlainOperator(1.2, wires=0),
            qml.RX(1.2, wires=0),
            qml.Hermitian([[1, 0], [0, 1]], wires=0),
            qml.PauliX(0),
        ),
    )
    def test_pickling(self, op):
        """Test that pickling works for all inheritance combinations."""
        adj_op = Adjoint(op)

        pickled_adj_op = pickle.dumps(adj_op)
        unpickled_op = pickle.loads(pickled_adj_op)

        assert type(adj_op) is type(unpickled_op)
        qml.assert_equal(adj_op, unpickled_op)


class TestInitialization:
    """Test the initialization process and standard properties."""

    # pylint: disable=use-implicit-booleaness-not-comparison
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
        assert op.data == ()

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

    def test_template_base(self, seed):
        """Test adjoint initialization for a template."""
        rng = np.random.default_rng(seed=seed)
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


class TestProperties:
    """Test Adjoint properties."""

    def test_data(self):
        """Test base data can be get and set through Adjoint class."""
        x = np.array(1.234)

        base = qml.RX(x, wires="a")
        adj = Adjoint(base)

        assert adj.data == (x,)

        # update parameters through adjoint
        x_new = np.array(2.3456)
        adj.data = (x_new,)
        assert base.data == (x_new,)
        assert adj.data == (x_new,)

        # update base data updates Adjoint data
        x_new2 = np.array(3.456)
        base.data = (x_new2,)
        assert adj.data == (x_new2,)

    def test_has_matrix_true(self):
        """Test `has_matrix` property carries over when base op defines matrix."""
        base = qml.PauliX(0)
        op = Adjoint(base)

        assert op.has_matrix is True

    def test_has_matrix_false(self):
        """Test has_matrix property carries over when base op does not define a matrix."""
        base = qml.StatePrep([1, 0], wires=0)
        op = Adjoint(base)

        assert op.has_matrix is False

    def test_has_decomposition_true_via_base_adjoint(self):
        """Test `has_decomposition` property is activated because the base operation defines an
        `adjoint` method."""
        base = qml.PauliX(0)
        op = Adjoint(base)

        assert op.has_decomposition is True

    def test_has_decomposition_true_via_base_decomposition(self):
        """Test `has_decomposition` property is activated because the base operation defines a
        `decomposition` method."""

        # pylint: disable=too-few-public-methods
        class MyOp(qml.operation.Operation):
            num_wires = 1

            def decomposition(self):
                return [qml.RX(0.2, self.wires)]

        base = MyOp(0)
        op = Adjoint(base)

        assert op.has_decomposition is True

    def test_has_decomposition_false(self):
        """Test `has_decomposition` property is not activated if the base neither
        `has_adjoint` nor `has_decomposition`."""

        # pylint: disable=too-few-public-methods
        class MyOp(qml.operation.Operation):
            num_wires = 1

        base = MyOp(0)
        op = Adjoint(base)

        assert op.has_decomposition is False

    def test_has_adjoint_true_always(self):
        """Test `has_adjoint` property to always be true, irrespective of the base."""

        # pylint: disable=too-few-public-methods
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

    def test_has_diagonalizing_gates_true_via_base_diagonalizing_gates(self):
        """Test `has_diagonalizing_gates` property is activated because the
        base operation defines a `diagonalizing_gates` method."""

        op = Adjoint(qml.PauliX(0))

        assert op.has_diagonalizing_gates is True

    def test_has_diagonalizing_gates_false(self):
        """Test `has_diagonalizing_gates` property is not activated if the base neither
        `has_adjoint` nor `has_diagonalizing_gates`."""

        # pylint: disable=too-few-public-methods
        class MyOp(qml.operation.Operation):
            num_wires = 1
            has_diagonalizing_gates = False

        op = Adjoint(MyOp(0))

        assert op.has_diagonalizing_gates is False

    def test_queue_category(self):
        """Test that the queue category `"_ops"` carries over."""
        op = Adjoint(qml.PauliX(0))
        assert op._queue_category == "_ops"  # pylint: disable=protected-access

    @pytest.mark.parametrize("value", (True, False))
    def test_is_hermitian(self, value):
        """Test `is_hermitian` property mirrors that of the base."""

        # pylint: disable=too-few-public-methods
        class DummyOp(qml.operation.Operator):
            num_wires = 1
            is_hermitian = value

        op = Adjoint(DummyOp(0))
        assert op.is_hermitian == value

    def test_batching_properties(self):
        """Test the batching properties and methods."""

        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = Adjoint(base)
        assert op.batch_size == 3
        assert op.ndim_params == (0,)

    def test_pauli_rep(self):
        """Test pauli_rep works after adjoint operation."""
        coeffs = [1 - 0.5j, 0.2, -3j]
        paulis = [qml.X(0), qml.Y(0), qml.Z(0)]
        op = qml.dot(coeffs, paulis)
        adjoint_ps = qml.adjoint(op).pauli_rep
        assert (list(adjoint_ps.values()) == qml.math.conjugate(coeffs)).all()
        assert (qml.adjoint(adjoint_ps.operation()).matrix() == op.matrix()).all()


class TestSimplify:
    """Test Adjoint simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        adj_op = Adjoint(Adjoint(qml.RZ(1.32, wires=0)))
        assert adj_op.arithmetic_depth == 2

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        adj_op = Adjoint(Adjoint(Adjoint(qml.RZ(1.32, wires=0))))
        final_op = qml.RZ(4 * np.pi - 1.32, wires=0)
        simplified_op = adj_op.simplify()
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_adj_of_sums(self):
        """Test that the simplify methods converts an adjoint of sums to a sum of adjoints."""
        adj_op = Adjoint(qml.sum(qml.RX(1, 0), qml.RY(1, 0), qml.RZ(1, 0)))
        sum_op = qml.sum(
            qml.RX(4 * np.pi - 1, 0), qml.RY(4 * np.pi - 1, 0), qml.RZ(4 * np.pi - 1, 0)
        )
        simplified_op = adj_op.simplify()
        qml.assert_equal(simplified_op, sum_op)

    def test_simplify_adj_of_prod(self):
        """Test that the simplify method converts an adjoint of products to a (reverse) product
        of adjoints."""
        adj_op = Adjoint(qml.prod(qml.RX(1, 0), qml.RY(1, 0), qml.RZ(1, 0)))
        final_op = qml.prod(
            qml.RZ(4 * np.pi - 1, 0), qml.RY(4 * np.pi - 1, 0), qml.RX(4 * np.pi - 1, 0)
        )
        simplified_op = adj_op.simplify()
        qml.assert_equal(simplified_op, final_op)

    def test_simplify_with_adjoint_not_defined(self):
        """Test the simplify method with an operator that has not defined the op.adjoint method."""
        op = Adjoint(qml.T(0))
        simplified_op = op.simplify()
        qml.assert_equal(simplified_op, op)


class TestMiscMethods:
    """Test miscellaneous small methods on the Adjoint class."""

    def test_repr(self):
        """Test __repr__ method."""
        assert repr(Adjoint(qml.S(0))) == "Adjoint(S(0))"

        base = qml.S(0) + qml.T(0)
        op = Adjoint(base)
        assert repr(op) == "Adjoint(S(0) + T(0))"

    def test_label(self):
        """Test that the label method for the adjoint class adds a † to the end."""
        base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
        op = Adjoint(base)
        assert op.label(decimals=2) == "Rot\n(1.23,\n2.35,\n3.46)†"

        base = qml.S(0) + qml.T(0)
        op = Adjoint(base)
        assert op.label() == "𝓗†"

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

    # pylint: disable=protected-access
    def test_flatten_unflatten(self):
        """Test the flatten and unflatten methods."""

        # pylint: disable=too-few-public-methods
        class CustomOp(qml.operation.Operator):
            pass

        op = CustomOp(1.2, 2.3, wires=0)
        adj_op = Adjoint(op)
        data, metadata = adj_op._flatten()
        assert len(data) == 1
        assert data[0] is op

        assert metadata == tuple()

        new_op = type(adj_op)._unflatten(*adj_op._flatten())
        qml.assert_equal(adj_op, new_op)


class TestAdjointOperation:
    """Test methods in the AdjointOperation mixin."""

    def test_has_generator_true(self):
        """Test `has_generator` property carries over when base op defines generator."""
        base = qml.RX(0.5, 0)
        op = Adjoint(base)

        assert op.has_generator is True

    def test_has_generator_false(self):
        """Test `has_generator` property carries over when base op does not define a generator."""
        base = qml.PauliX(0)
        op = Adjoint(base)

        assert op.has_generator is False

    def test_generator(self):
        """Assert that the generator of an Adjoint is -1.0 times the base generator."""
        base = qml.RX(1.23, wires=0)
        op = Adjoint(base)

        qml.assert_equal(base.generator(), -1.0 * op.generator())

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

        with qml.queuing.AnnotatedQueue() as q:
            base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
            _ = Adjoint(base)

        assert base not in q
        assert len(q) == 1

    def test_queuing_base_defined_outside(self):
        """Test that base isn't added to queue if it's defined outside the recording context."""

        base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
        with qml.queuing.AnnotatedQueue() as q:
            op = Adjoint(base)

        assert len(q) == 1
        assert q.queue[0] is op


class TestMatrix:
    """Test the matrix method for a variety of interfaces."""

    def test_batching_support(self):
        """Test that adjoint matrix has batching support."""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        base = qml.RX(x, wires=0)
        op = Adjoint(base)
        mat = op.matrix()
        compare = qml.RX(-x, wires=0)

        assert qml.math.allclose(mat, compare.matrix())
        assert mat.shape == (3, 2, 2)

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

    def test_no_matrix_defined(self, seed):
        """Test that if the base has no matrix defined, then Adjoint.matrix also raises a MatrixUndefinedError."""
        rng = np.random.default_rng(seed=seed)
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
    from scipy.sparse import coo_matrix, csr_matrix

    H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
    H = csr_matrix(H)
    base = qml.SparseHamiltonian(H, wires=0)

    op = Adjoint(base)

    base_sparse_mat = base.sparse_matrix()
    base_conj_T = qml.numpy.conj(qml.numpy.transpose(base_sparse_mat))
    op_sparse_mat = op.sparse_matrix()

    assert isinstance(op_sparse_mat, csr_matrix)
    assert isinstance(op.sparse_matrix(format="coo"), coo_matrix)

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

    def test_batching_eigvals(self):
        """Test that eigenvalues work with batched parameters."""
        x = np.array([1.2, 2.3, 3.4])
        base = qml.RX(x, 0)
        adj = Adjoint(base)
        compare = qml.RX(-x, 0)

        # eigvals might have different orders
        assert qml.math.allclose(adj.eigvals()[:, 0], compare.eigvals()[:, 1])
        assert qml.math.allclose(adj.eigvals()[:, 1], compare.eigvals()[:, 0])

    def test_no_matrix_defined_eigvals(self):
        """Test that if the base does not define eigvals, The Adjoint raises the same error."""
        base = qml.StatePrep([1, 0], wires=0)

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
        tape = qml.tape.QuantumScript(Adjoint(base).decomposition())

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
        base_tape = qml.tape.QuantumScript(base.decomposition())
        tape = qml.tape.QuantumScript(Adjoint(base).decomposition())

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

        tape = qml.tape.QuantumScript(adj2.decomposition())
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


##### TESTS FOR THE ADJOINT CONSTRUCTOR ######

noncallable_objects = [
    [qml.Hadamard(1), qml.RX(-0.2, wires=1)],
    qml.tape.QuantumScript(),
]


@pytest.mark.parametrize("obj", noncallable_objects)
def test_error_adjoint_on_noncallable(obj):
    """Test that an error is raised if qml.adjoint is applied to an object that
    is not callable, as it silently does not have any effect on those."""
    with pytest.raises(ValueError, match=f"{type(obj)} is not callable."):
        adjoint(obj)


class TestAdjointConstructorPreconstructedOp:
    """Test providing an already initialized operator to the transform."""

    @pytest.mark.parametrize(
        "base", (qml.IsingXX(1.23, wires=("c", "d")), qml.QFT(wires=(0, 1, 2)))
    )
    def test_single_op(self, base):
        """Test passing a single preconstructed op in a queuing context."""
        with qml.queuing.AnnotatedQueue() as q:
            base.queue()
            out = adjoint(base)

        assert len(q) == 1
        assert q.queue[0] is out

    def test_single_op_defined_outside_queue_eager(self):
        """Test if base is defined outside context and the function eagerly simplifies
        the adjoint, the base is not added to queue."""
        base = qml.RX(1.2, wires=0)
        with qml.queuing.AnnotatedQueue() as q:
            out = adjoint(base, lazy=False)

        assert len(q) == 1
        assert q.queue[0] is out

    def test_correct_queued_operators(self):
        """Test that args and kwargs do not add operators to the queue."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.adjoint(qml.QSVT)(qml.X(1), [qml.Z(1)])
            qml.adjoint(qml.QSVT(qml.X(1), [qml.Z(1)]))

        for op in q.queue:
            assert op.name == "Adjoint(QSVT)"

        assert len(q.queue) == 2


class TestAdjointConstructorDifferentCallableTypes:
    """Test the adjoint transform on a variety of possible inputs."""

    def test_adjoint_single_op_function(self):
        """Test the adjoint transform on a single operation."""

        with qml.queuing.AnnotatedQueue() as q:
            out = adjoint(qml.RX)(1.234, wires="a")

        tape = qml.tape.QuantumScript.from_queue(q)
        assert out is tape[0]
        assert isinstance(out, Adjoint)
        qml.assert_equal(out.base, qml.RX(1.234, "a"))

    def test_adjoint_template(self):
        """Test the adjoint transform on a template."""

        with qml.queuing.AnnotatedQueue() as q:
            out = adjoint(qml.QFT)(wires=(0, 1, 2))

        tape = qml.tape.QuantumScript.from_queue(q)
        assert len(tape) == 1
        assert out is tape[0]
        assert isinstance(out, Adjoint)
        assert out.base.__class__ is qml.QFT
        assert out.wires == qml.wires.Wires((0, 1, 2))

    def test_adjoint_on_function(self):
        """Test adjoint transform on a function"""

        def func(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.RZ(z, wires=0)

        x = 1.23
        y = 2.34
        z = 3.45
        with qml.queuing.AnnotatedQueue() as q:
            out = adjoint(func)(x, y, z)

        tape = qml.tape.QuantumScript.from_queue(q)
        assert out == tape.circuit

        for op in tape:
            assert isinstance(op, Adjoint)

        # check order reversed
        assert tape[0].base.__class__ is qml.RZ
        assert tape[1].base.__class__ is qml.RY
        assert tape[2].base.__class__ is qml.RX

        # check parameters assigned correctly
        assert tape[0].data == (z,)
        assert tape[1].data == (y,)
        assert tape[2].data == (x,)

    def test_nested_adjoint(self):
        """Test the adjoint transform on an adjoint transform."""
        x = 4.321
        with qml.queuing.AnnotatedQueue() as q:
            out = adjoint(adjoint(qml.RX))(x, wires="b")

        tape = qml.tape.QuantumScript.from_queue(q)
        assert out is tape[0]
        assert isinstance(out, Adjoint)
        assert isinstance(out.base, Adjoint)
        assert out.base.base.__class__ is qml.RX
        assert out.data == (x,)
        assert out.wires == qml.wires.Wires("b")


class TestAdjointConstructorNonLazyExecution:
    """Test the lazy=False keyword."""

    def test_single_decomposeable_op(self):
        """Test lazy=False for a single op that gets decomposed."""

        x = 1.23
        with qml.queuing.AnnotatedQueue() as q:
            base = qml.RX(x, wires="b")
            out = adjoint(base, lazy=False)

        assert len(q) == 1
        assert q.queue[0] is out

        assert isinstance(out, qml.RX)
        assert out.data == (-1.23,)

    def test_single_nondecomposable_op(self):
        """Test lazy=false for a single op that can't be decomposed."""
        with qml.queuing.AnnotatedQueue() as q:
            base = qml.S(0)
            out = adjoint(base, lazy=False)

        assert len(q) == 1
        assert q.queue[0] is out

        assert isinstance(out, Adjoint)
        assert isinstance(out.base, qml.S)

    def test_single_decomposable_op_function(self):
        """Test lazy=False for a single op callable that gets decomposed."""
        x = 1.23
        with qml.queuing.AnnotatedQueue() as q:
            out = adjoint(qml.RX, lazy=False)(x, wires="b")

        tape = qml.tape.QuantumScript.from_queue(q)
        assert out is tape[0]
        assert not isinstance(out, Adjoint)
        assert isinstance(out, qml.RX)
        assert out.data == (-x,)

    def test_single_nondecomposable_op_function(self):
        """Test lazy=False for a single op function that can't be decomposed."""
        with qml.queuing.AnnotatedQueue() as q:
            out = adjoint(qml.S, lazy=False)(0)

        tape = qml.tape.QuantumScript.from_queue(q)
        assert out is tape[0]
        assert isinstance(out, Adjoint)
        assert isinstance(out.base, qml.S)

    def test_mixed_function(self):
        """Test lazy=False with a function that applies operations of both types."""
        x = 1.23

        def qfunc(x):
            qml.RZ(x, wires="b")
            qml.T("b")

        with qml.queuing.AnnotatedQueue() as q:
            out = adjoint(qfunc, lazy=False)(x)

        tape = qml.tape.QuantumScript.from_queue(q)
        assert len(tape) == len(out) == 2
        assert isinstance(tape[0], Adjoint)
        assert isinstance(tape[0].base, qml.T)

        assert isinstance(tape[1], qml.RZ)
        assert tape[1].data[0] == -x


class TestAdjointConstructorOutsideofQueuing:
    """Test the behaviour of the adjoint transform when not called in a queueing context."""

    def test_single_op(self):
        """Test providing a single op outside of a queuing context."""

        x = 1.234
        out = adjoint(qml.RZ(x, wires=0))

        assert isinstance(out, Adjoint)
        assert out.base.__class__ is qml.RZ
        assert out.data == (1.234,)
        assert out.wires == qml.wires.Wires(0)

    def test_single_op_eager(self):
        """Test a single op that can be decomposed in eager mode outside of a queuing context."""

        x = 1.234
        base = qml.RX(x, wires=0)
        out = adjoint(base, lazy=False)

        assert isinstance(out, qml.RX)
        assert out.data == (-x,)

    def test_single_op_function(self):
        """Test the transform on a single op as a callable outside of a queuing context."""
        x = 1.234
        out = adjoint(qml.IsingXX)(x, wires=(0, 1))

        assert isinstance(out, Adjoint)
        assert out.base.__class__ is qml.IsingXX
        assert out.data == (1.234,)
        assert out.wires == qml.wires.Wires((0, 1))

    def test_function(self):
        """Test the transform on a function outside of a queuing context."""

        def func(wire):
            qml.S(wire)
            qml.SX(wire)

        wire = 1.234
        out = adjoint(func)(wire)

        assert len(out) == 2
        assert all(isinstance(op, Adjoint) for op in out)
        assert all(op.wires == qml.wires.Wires(wire) for op in out)

    def test_nonlazy_op_function(self):
        """Test non-lazy mode on a simplifiable op outside of a queuing context."""

        out = adjoint(qml.PauliX, lazy=False)(0)

        assert not isinstance(out, Adjoint)
        assert isinstance(out, qml.PauliX)


class TestAdjointConstructorIntegration:
    """Test circuit execution and gradients with the adjoint transform."""

    def test_single_op(self):
        """Test the adjoint of a single op against analytically expected results."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circ():
            qml.PauliX(0)
            adjoint(qml.S)(0)
            return qml.state()

        res = circ()
        expected = np.array([0, -1j])

        assert np.allclose(res, expected)

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ("backprop", "adjoint", "parameter-shift"))
    def test_gradient_autograd(self, diff_method):
        """Test gradients through the adjoint transform with autograd."""
        import autograd

        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method)
        def circ(x):
            adjoint(qml.RX)(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = autograd.numpy.array(0.234)
        expected_res = np.sin(x)
        expected_grad = np.cos(x)
        assert qml.math.allclose(circ(x), expected_res)
        assert qml.math.allclose(
            autograd.grad(circ)(x), expected_grad  # pylint: disable=no-value-for-parameter
        )

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ("backprop", "adjoint", "parameter-shift"))
    def test_gradient_jax(self, diff_method):
        """Test gradients through the adjoint transform with jax."""
        import jax

        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method)
        def circ(x):
            adjoint(qml.RX)(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = jax.numpy.array(0.234)
        expected_res = jax.numpy.sin(x)
        expected_grad = jax.numpy.cos(x)
        assert qml.math.allclose(circ(x), expected_res)
        assert qml.math.allclose(jax.grad(circ)(x), expected_grad)

    @pytest.mark.torch
    @pytest.mark.parametrize("diff_method", ("backprop", "adjoint", "parameter-shift"))
    def test_gradient_torch(self, diff_method):
        """Test gradients through the adjoint transform with torch."""
        import torch

        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method)
        def circ(x):
            adjoint(qml.RX)(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = torch.tensor(0.234, requires_grad=True)
        y = circ(x)
        y.backward()

        assert qml.math.allclose(y, torch.sin(x))
        assert qml.math.allclose(x.grad, torch.cos(x))

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ("backprop", "adjoint", "parameter-shift"))
    def test_gradient_tf(self, diff_method):
        """Test gradients through the adjoint transform with tensorflow."""

        import tensorflow as tf

        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method)
        def circ(x):
            adjoint(qml.RX)(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = tf.Variable(0.234, dtype=tf.float64)
        with tf.GradientTape() as tape:
            y = circ(x)

        grad = tape.gradient(y, x)

        assert qml.math.allclose(y, tf.sin(x))
        assert qml.math.allclose(grad, tf.cos(x))
