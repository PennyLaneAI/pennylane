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
"""Unit tests for the ``Exp`` class"""
import copy

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import (
    DecompositionUndefinedError,
    GeneratorUndefinedError,
    ParameterFrequenciesUndefinedError,
)
from pennylane.ops.op_math import Evolution, Exp


@pytest.mark.parametrize("constructor", (qml.exp, Exp))
class TestInitialization:
    """Test the initialization process and standard properties."""

    def test_pauli_base(self, constructor):
        """Test initialization with no coeff and a simple base."""
        base = qml.PauliX("a")

        op = constructor(base, id="something")

        assert op.base is base
        assert op.coeff == 1
        assert op.name == "Exp"
        assert op.id == "something"

        assert op.num_params == 1
        assert op.parameters == [[1], []]
        assert op.data == [[1], []]

        assert op.wires == qml.wires.Wires("a")

        assert op.control_wires == qml.wires.Wires([])

    def test_provided_coeff(self, constructor):
        """Test initialization with a provided coefficient and a Tensor base."""
        base = qml.PauliZ("b") @ qml.PauliZ("c")
        coeff = np.array(1.234)

        op = constructor(base, coeff)

        assert op.base is base
        assert op.coeff is coeff
        assert op.name == "Exp"

        assert op.num_params == 1
        assert op.parameters == [[coeff], []]
        assert op.data == [[coeff], []]

        assert op.wires == qml.wires.Wires(("b", "c"))

    def test_parametric_base(self, constructor):
        """Test initialization with a coefficient and a parametric operation base."""

        base_coeff = 1.23
        base = qml.RX(base_coeff, wires=5)
        coeff = np.array(-2.0)

        op = constructor(base, coeff)

        assert op.base is base
        assert op.coeff is coeff
        assert op.name == "Exp"

        assert op.num_params == 2
        assert op.data == [coeff, [base_coeff]]

        assert op.wires == qml.wires.Wires(5)

    @pytest.mark.parametrize("value", (True, False))
    def test_has_diagonalizing_gates(self, value, constructor):
        """Test that Exp defers has_diagonalizing_gates to base operator."""

        # pylint: disable=too-few-public-methods
        class DummyOp(qml.operation.Operator):
            num_wires = 1
            has_diagonalizing_gates = value

        op = constructor(DummyOp(1), 2.312)
        assert op.has_diagonalizing_gates is value


class TestProperties:
    """Test of the properties of the Exp class."""

    def test_data(self):
        """Test intializaing and accessing the data property."""

        phi = np.array(1.234)
        coeff = np.array(2.345)

        base = qml.RX(phi, wires=0)
        op = Exp(base, coeff)

        assert op.data == [[coeff], [phi]]

    # pylint: disable=protected-access
    def test_queue_category_ops(self):
        """Test the _queue_category property."""
        assert Exp(qml.PauliX(0), -1.234j)._queue_category == "_ops"

        assert Exp(qml.PauliX(0), 1 + 2j)._queue_category is None

        assert Exp(qml.RX(1.2, 0), -1.2j)._queue_category is None

    def test_is_hermitian(self):
        """Test that the op is hermitian if the base is hermitian and the coeff is real."""
        assert Exp(qml.PauliX(0), -1.0).is_hermitian

        assert not Exp(qml.PauliX(0), 1.0 + 2j).is_hermitian

        assert not Exp(qml.RX(1.2, wires=0)).is_hermitian

    def test_batching_properties(self):
        """Test the batching properties and methods."""

        # base is batched
        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = Exp(base)
        assert op.batch_size == 3

        # coeff is batched
        base = qml.RX(1, 0)
        op = Exp(base, coeff=np.array([1.2, 2.3, 3.4]))
        assert op.batch_size == 3

        # both are batched
        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = Exp(base, coeff=np.array([1.2, 2.3, 3.4]))
        assert op.batch_size == 3

    def test_different_batch_sizes_raises_error(self):
        """Test that using different batch sizes for base and scalar raises an error."""
        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        with pytest.raises(
            ValueError, match="Broadcasting was attempted but the broadcasted dimensions"
        ):
            _ = Exp(base, np.array([0.1, 1.2, 2.3, 3.4]))


class TestMatrix:
    """Test the matrix method."""

    def test_base_batching_support(self):
        """Test that Exp matrix has base batching support."""
        x = np.array([-1, -2, -3])
        op = Exp(qml.RX(x, 0), 3)
        mat = op.matrix()
        true_mat = qml.math.stack([Exp(qml.RX(i, 0), 3).matrix() for i in x])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    def test_coeff_batching_support(self):
        """Test that Exp matrix has coeff batching support."""
        x = np.array([-1, -2, -3])
        op = Exp(qml.PauliX(0), x)
        mat = op.matrix()
        true_mat = qml.math.stack([Exp(qml.PauliX(0), i).matrix() for i in x])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    def test_base_and_coeff_batching_support(self):
        """Test that Exp matrix has base and coeff batching support."""
        x = np.array([-1, -2, -3])
        y = np.array([1, 2, 3])
        op = Exp(qml.RX(x, 0), y)
        mat = op.matrix()
        true_mat = qml.math.stack([Exp(qml.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    @pytest.mark.jax
    def test_batching_jax(self):
        """Test that Exp matrix has batching support with the jax interface."""
        import jax.numpy as jnp

        x = jnp.array([-1, -2, -3])
        y = jnp.array([1, 2, 3])
        op = Exp(qml.RX(x, 0), y)
        mat = op.matrix()
        true_mat = qml.math.stack([Exp(qml.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)
        assert isinstance(mat, jnp.ndarray)

    @pytest.mark.torch
    def test_batching_torch(self):
        """Test that Exp matrix has batching support with the torch interface."""
        import torch

        x = torch.tensor([-1, -2, -3])
        y = torch.tensor([1, 2, 3])
        op = Exp(qml.RX(x, 0), y)
        mat = op.matrix()
        true_mat = qml.math.stack([Exp(qml.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)
        assert isinstance(mat, torch.Tensor)

    @pytest.mark.tf
    def test_batching_tf(self):
        """Test that Exp matrix has batching support with the tensorflow interface."""
        import tensorflow as tf

        x = tf.constant([-1.0, -2.0, -3.0])
        y = tf.constant([1.0, 2.0, 3.0])
        op = Exp(qml.RX(x, 0), y)
        mat = op.matrix()
        true_mat = qml.math.stack([Exp(qml.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qml.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)
        assert isinstance(mat, tf.Tensor)

    def test_tensor_base_isingxx(self):
        """Test that isingxx can be created with a tensor base."""
        phi = -0.46
        base = qml.PauliX(0) @ qml.PauliX(1)
        op = Exp(base, -0.5j * phi)
        isingxx = qml.IsingXX(phi, wires=(0, 1))

        assert qml.math.allclose(op.matrix(), isingxx.matrix())

    def test_prod_base_isingyy(self):
        """Test that IsingYY can be created with a `Prod` base."""
        phi = -0.46
        base = qml.prod(qml.PauliY(0), qml.PauliY(1))
        op = Exp(base, -0.5j * phi)
        isingxx = qml.IsingYY(phi, wires=(0, 1))

        assert qml.math.allclose(op.matrix(), isingxx.matrix())

    @pytest.mark.autograd
    @pytest.mark.parametrize("requires_grad", (True, False))
    def test_matrix_autograd_rx(self, requires_grad):
        """Test the matrix comparing to the rx gate."""
        phi = np.array(1.234, requires_grad=requires_grad)
        exp_rx = Exp(qml.PauliX(0), -0.5j * phi)
        rx = qml.RX(phi, 0)

        assert qml.math.allclose(exp_rx.matrix(), rx.matrix())

    @pytest.mark.autograd
    @pytest.mark.parametrize("requires_grad", (True, False))
    def test_matrix_autograd_rz(self, requires_grad):
        """Test the matrix comparing to the rz gate. This is a gate with an
        autograd coefficient but empty diagonalizing gates."""
        phi = np.array(1.234, requires_grad=requires_grad)
        exp_rz = Exp(qml.PauliZ(0), -0.5j * phi)
        rz = qml.RZ(phi, 0)

        assert qml.math.allclose(exp_rz.matrix(), rz.matrix())

    @pytest.mark.autograd
    @pytest.mark.parametrize("requires_grad", (True, False))
    def test_tensor_with_pauliz_autograd(self, requires_grad):
        """Test the matrix for the case when the coefficient is autograd and
        the diagonalizing gates don't act on every wire for the matrix."""
        phi = qml.numpy.array(-0.345, requires_grad=requires_grad)
        base = qml.PauliZ(0) @ qml.PauliY(1)
        autograd_op = Exp(base, phi)
        mat = qml.math.expm(phi * qml.matrix(base))

        assert qml.math.allclose(autograd_op.matrix(), mat)

    @pytest.mark.autograd
    def test_base_no_diagonalizing_gates_autograd_coeff(self):
        """Test the matrix when the base matrix doesn't define the diagonalizing gates."""
        coeff = np.array(0.4)
        base = qml.RX(2.0, wires=0)
        op = Exp(base, coeff)

        with pytest.warns(UserWarning, match="The autograd matrix for "):
            mat = op.matrix()
        expected = qml.math.expm(coeff * base.matrix())
        assert qml.math.allclose(mat, expected)

    @pytest.mark.torch
    def test_torch_matrix_rx(self):
        """Test the matrix with torch."""
        import torch

        phi = torch.tensor(0.4, dtype=torch.complex128)

        base = qml.PauliX(0)
        op = Exp(base, -0.5j * phi)
        compare = qml.RX(phi, 0)

        assert qml.math.allclose(op.matrix(), compare.matrix())

    @pytest.mark.tf
    def test_tf_matrix_rx(self):
        """Test the matrix with tensorflow."""

        import tensorflow as tf

        phi = tf.Variable(0.4, dtype=tf.complex128)
        base = qml.PauliX(0)
        op = Exp(base, -0.5j * phi)
        compare = qml.RX(phi, wires=0)
        assert qml.math.allclose(op.matrix(), compare.matrix())

    @pytest.mark.jax
    def test_jax_matrix_rx(self):
        """Test the matrix with jax."""
        import jax

        phi = jax.numpy.array(0.4 + 0j)

        base = qml.PauliX(0)
        op = Exp(base, -0.5j * phi)
        compare = qml.RX(phi, 0)

        assert qml.math.allclose(op.matrix(), compare.matrix())

        def exp_mat(x):
            return qml.exp(base, -0.5j * x).matrix()

        def rx_mat(x):
            return qml.RX(x, wires=0).matrix()

        exp_mat_grad = jax.jacobian(exp_mat, holomorphic=True)(phi)
        rx_mat_grad = jax.jacobian(rx_mat, holomorphic=True)(phi)

        assert qml.math.allclose(exp_mat_grad, rx_mat_grad)

    def test_sparse_matrix(self):
        """Test the sparse matrix function."""
        from scipy.sparse import csr_matrix

        H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
        H = csr_matrix(H)
        base = qml.SparseHamiltonian(H, wires=0)

        op = Exp(base, 3)

        sp_format = "lil"
        sparse_mat = op.sparse_matrix(format=sp_format)
        assert sparse_mat.format == sp_format

        dense_mat = qml.matrix(op)

        assert qml.math.allclose(sparse_mat.toarray(), dense_mat)

    def test_sparse_matrix_wire_order_error(self):
        """Test that sparse_matrix raises an error if wire_order provided."""
        from scipy.sparse import csr_matrix

        H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
        H = csr_matrix(H)
        base = qml.SparseHamiltonian(H, wires=0)

        op = Exp(base, 3)

        with pytest.raises(NotImplementedError):
            op.sparse_matrix(wire_order=[0, 1])


class TestDecomposition:
    @pytest.mark.parametrize("coeff", (1, 1 + 0.5j))
    def test_non_imag_no_decomposition(self, coeff):
        """Tests that the decomposition doesn't exist if the coefficient has a real component."""
        op = Exp(qml.PauliX(0), coeff)
        assert not op.has_decomposition
        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()

    def test_non_pauli_word_base_no_decomposition(self):
        """Tests that the decomposition doesn't exist if the base is not a pauli word."""
        op = Exp(qml.S(0), -0.5j)
        assert not op.has_decomposition
        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()

    def test_nontensor_tensor_raises_error(self):
        """Checks that accessing the decomposition throws an error if the base is a Tensor
        object that is not a mathematical tensor"""
        base_op = qml.PauliX(0) @ qml.PauliZ(0)
        op = Exp(base_op, 1j)

        with pytest.raises(NotImplementedError):
            op.has_decomposition()

        with pytest.raises(NotImplementedError):
            op.decomposition()

    @pytest.mark.parametrize(
        "base, base_string",
        (
            (qml.PauliX(0), "X"),
            (qml.PauliZ(0) @ qml.PauliY(1), "ZY"),
            (qml.PauliY(0) @ qml.Identity(1) @ qml.PauliZ(2), "YIZ"),
        ),
    )
    def test_pauli_word_decompositions(self, base, base_string):
        """Check that Exp decomposes into PauliRot if possible."""
        theta = 3.21
        op = Exp(base, -0.5j * theta)

        assert op.has_decomposition
        pr = op.decomposition()[0]
        assert qml.equal(pr, qml.PauliRot(3.21, base_string, base.wires))


class TestMiscMethods:
    """Test other representation methods."""

    def test_repr_paulix(self):
        """Test the __repr__ method when the base is a simple observable."""
        op = Exp(qml.PauliX(0), 3)
        assert repr(op) == "Exp(3 PauliX)"

    def test_repr_tensor(self):
        """Test the __repr__ method when the base is a tensor."""
        t = qml.PauliX(0) @ qml.PauliX(1)
        isingxx = Exp(t, 0.25j)

        assert repr(isingxx) == "Exp(0.25j PauliX(wires=[0]) @ PauliX(wires=[1]))"

    def test_repr_deep_operator(self):
        """Test the __repr__ method when the base is any operator with arithmetic depth > 0."""
        base = qml.S(0) @ qml.PauliX(0)
        op = qml.ops.Exp(base, 3)

        assert repr(op) == "Exp(3 S(wires=[0]) @ PauliX(wires=[0]))"

    def test_diagonalizing_gates(self):
        """Test that the diagonalizing gates are the same as the base diagonalizing gates."""
        base = qml.PauliX(0)
        op = Exp(base, 1 + 2j)
        for op1, op2 in zip(base.diagonalizing_gates(), op.diagonalizing_gates()):
            assert qml.equal(op1, op2)

    def test_pow(self):
        """Test the pow decomposition method."""
        base = qml.PauliX(0)
        coeff = 2j
        z = 0.3

        op = Exp(base, coeff)
        pow_op = op.pow(z)

        assert isinstance(pow_op, Exp)
        assert pow_op.base is base
        assert pow_op.coeff == coeff * z

    @pytest.mark.parametrize(
        "op,decimals,expected",
        [
            (Exp(qml.PauliZ(0), 2 + 3j), None, "Exp((2+3j) Z)"),
            (Exp(qml.PauliZ(0), 2 + 3j), 2, "Exp(2.00+3.00j Z)"),
            (Exp(qml.prod(qml.PauliZ(0), qml.PauliY(1)), 2 + 3j), None, "Exp((2+3j) Z@Y)"),
            (Exp(qml.prod(qml.PauliZ(0), qml.PauliY(1)), 2 + 3j), 2, "Exp(2.00+3.00j Z@Y)"),
            (Exp(qml.RZ(1.234, wires=[0]), 5.678), None, "Exp(5.678 RZ)"),
            (Exp(qml.RZ(1.234, wires=[0]), 5.678), 2, "Exp(5.68 RZ\n(1.23))"),
        ],
    )
    def test_label(self, op, decimals, expected):
        """Test that the label is informative and uses decimals."""
        assert op.label(decimals=decimals) == expected

    def test_simplify_sprod(self):
        """Test that simplify merges SProd into the coefficent."""
        base = qml.adjoint(qml.PauliX(0))
        s_op = qml.s_prod(2.0, base)

        op = Exp(s_op, 3j)
        new_op = op.simplify()
        assert qml.equal(new_op.base, qml.PauliX(0))
        assert new_op.coeff == 6.0j

    def test_simplify(self):
        """Test that the simplify method simplifies the base."""
        orig_base = qml.adjoint(qml.adjoint(qml.PauliX(0)))

        op = Exp(orig_base, coeff=0.2)
        new_op = op.simplify()
        assert qml.equal(new_op.base, qml.PauliX(0))
        assert new_op.coeff == 0.2

    def test_simplify_s_prod(self):
        """Tests that when simplification of the base results in an SProd,
        the scalar is included in the coeff rather than the base"""
        base = qml.s_prod(2, qml.sum(qml.PauliX(0), qml.PauliX(0)))
        op = Exp(base, 3)
        new_op = op.simplify()

        assert qml.equal(new_op.base, qml.PauliX(0))
        assert new_op.coeff == 12
        assert new_op is not op

    def test_copy(self):
        """Tests making a copy."""
        op = Exp(qml.CNOT([0, 1]), 2)
        copied_op = copy.copy(op)

        assert qml.equal(op.base, copied_op.base)
        assert op.data == copied_op.data
        assert op.hyperparameters.keys() == copied_op.hyperparameters.keys()

        for attr, value in vars(copied_op).items():
            if (
                attr != "_hyperparameters"
            ):  # hyperparameters contains base, which can't be compared via ==
                assert vars(op)[attr] == value

        assert len(vars(op).items()) == len(vars(copied_op).items())


class TestIntegration:
    """Test Exp with gradients in qnodes."""

    @pytest.mark.jax
    def test_jax_qnode(self):
        """Test the execution and gradient of a jax qnode."""

        import jax
        from jax import numpy as jnp

        phi = jnp.array(1.234)

        @qml.qnode(qml.device("default.qubit.jax", wires=1), interface="jax")
        def circ(phi):
            Exp(qml.PauliX(0), -0.5j * phi)
            return qml.expval(qml.PauliZ(0))

        res = circ(phi)
        assert qml.math.allclose(res, jnp.cos(phi))
        grad = jax.grad(circ)(phi)
        assert qml.math.allclose(grad, -jnp.sin(phi))

    @pytest.mark.tf
    def test_tensorflow_qnode(self):
        """test the execution of a tensorflow qnode."""
        import tensorflow as tf

        phi = tf.Variable(1.2, dtype=tf.complex128)

        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tensorflow")
        def circ(phi):
            Exp(qml.PauliX(0), -0.5j * phi)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circ(phi)

        phi_grad = tape.gradient(res, phi)

        assert qml.math.allclose(res, tf.cos(phi))
        assert qml.math.allclose(
            phi_grad, -tf.sin(phi)  # pylint: disable=invalid-unary-operand-type
        )

    @pytest.mark.torch
    def test_torch_qnode(self):
        """Test execution with torch."""
        import torch

        phi = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="torch")
        def circuit(phi):
            Exp(qml.PauliX(0), -0.5j * phi)
            return qml.expval(qml.PauliZ(0))

        res = circuit(phi)
        assert qml.math.allclose(res, torch.cos(phi))

        res.backward()
        assert qml.math.allclose(phi.grad, -torch.sin(phi))

    @pytest.mark.autograd
    def test_autograd_qnode(self):
        """Test execution and gradient with pennylane numpy array."""
        phi = qml.numpy.array(1.2)

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(phi):
            Exp(qml.PauliX(0), -0.5j * phi)
            return qml.expval(qml.PauliZ(0))

        res = circuit(phi)
        assert qml.math.allclose(res, qml.numpy.cos(phi))

        grad = qml.grad(circuit)(phi)
        assert qml.math.allclose(grad, -qml.numpy.sin(phi))

    @pytest.mark.autograd
    def test_autograd_param_shift_qnode(self):
        """Test execution and gradient with pennylane numpy array."""
        phi = qml.numpy.array(1.2)

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, gradient_fn=qml.gradients.param_shift)
        def circuit(phi):
            Exp(qml.PauliX(0), -0.5j * phi)
            return qml.expval(qml.PauliZ(0))

        res = circuit(phi)
        assert qml.math.allclose(res, qml.numpy.cos(phi))

        grad = qml.grad(circuit)(phi)
        assert qml.math.allclose(grad, -qml.numpy.sin(phi))

    @pytest.mark.autograd
    def test_autograd_measurement(self):
        """Test exp in a measurement with gradient and autograd."""

        x = qml.numpy.array(2)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit(x):
            qml.Hadamard(0)
            return qml.expval(Exp(qml.PauliZ(0), x))

        res = circuit(x)
        expected = 0.5 * (np.exp(x) + np.exp(-x))
        assert qml.math.allclose(res, expected)

        grad = qml.grad(circuit)(x)
        expected_grad = 0.5 * (np.exp(x) - np.exp(-x))
        assert qml.math.allclose(grad, expected_grad)

    @pytest.mark.torch
    def test_torch_measurement(self):
        """Test Exp in a measurement with gradient and torch."""

        import torch

        x = torch.tensor(2.0, requires_grad=True, dtype=float)

        @qml.qnode(qml.device("default.qubit", wires=1), interface="torch")
        def circuit(x):
            qml.Hadamard(0)
            return qml.expval(Exp(qml.PauliZ(0), x))

        res = circuit(x)
        expected = 0.5 * (torch.exp(x) + torch.exp(-x))
        assert qml.math.allclose(res, expected)

        res.backward()
        expected_grad = 0.5 * (torch.exp(x) - torch.exp(-x))
        assert qml.math.allclose(x.grad, expected_grad)

    @pytest.mark.jax
    def test_jax_measurement(self):
        """Test Exp in a measurement with gradient and jax."""

        import jax
        from jax import numpy as jnp

        x = jnp.array(2.0)

        @qml.qnode(qml.device("default.qubit", wires=1), interface="jax")
        def circuit(x):
            qml.Hadamard(0)
            return qml.expval(Exp(qml.PauliZ(0), x))

        res = circuit(x)
        expected = 0.5 * (jnp.exp(x) + jnp.exp(-x))
        assert qml.math.allclose(res, expected)

        grad = jax.grad(circuit)(x)
        expected_grad = 0.5 * (jnp.exp(x) - jnp.exp(-x))
        assert qml.math.allclose(grad, expected_grad)

    @pytest.mark.tf
    def test_tf_measurement(self):
        """Test Exp in a measurement with gradient and tensorflow."""
        import tensorflow as tf

        x = tf.Variable(2.0)

        @qml.qnode(qml.device("default.qubit", wires=1), interface="tensorflow")
        def circuit(x):
            qml.Hadamard(0)
            return qml.expval(Exp(qml.PauliZ(0), x))

        with tf.GradientTape() as tape:
            res = circuit(x)

        expected = 0.5 * (tf.exp(x) + tf.exp(-x))
        assert qml.math.allclose(res, expected)

        x_grad = tape.gradient(res, x)
        expected_grad = 0.5 * (tf.exp(x) - tf.exp(-x))
        assert qml.math.allclose(x_grad, expected_grad)

    def test_draw_integration(self):
        """Test that Exp integrates with drawing."""

        phi = qml.numpy.array(1.2)

        with qml.queuing.AnnotatedQueue() as q:
            Exp(qml.PauliX(0), -0.5j * phi)

        tape = qml.tape.QuantumScript.from_queue(q)

        assert qml.drawer.tape_text(tape) == "0: ──Exp(-0.6j X)─┤  "

    def test_exp_batching(self):
        """Test execution of a batched Exp operator."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x):
            Exp(qml.PauliX(0), 0.5j * x)
            return qml.expval(qml.PauliY(0))

        x = qml.numpy.array([1.234, 2.34, 3.456])
        res = circuit(x)

        expected = np.sin(x)
        assert qml.math.allclose(res, expected)


class TestDifferentiation:
    """Test generator and parameter_frequency for differentiation"""

    def test_base_not_hermitian_generator_undefined(self):
        """That that imaginary coefficient but non-Hermitian base operator raises GeneratorUndefinedError"""
        op = Exp(qml.RX(1.23, 0), 1j)
        with pytest.raises(GeneratorUndefinedError):
            op.generator()

    def test_real_component_coefficient_generator_undefined(self):
        """Test that Hermitian base operator but real coefficient raises GeneratorUndefinedError"""
        op = Exp(qml.PauliX(0), 1)
        with pytest.raises(GeneratorUndefinedError):
            op.generator()

    def test_generator_is_base_operator(self):
        """Test that generator is base operator"""
        base_op = qml.PauliX(0)
        op = Exp(base_op, 1j)
        assert op.base == op.generator()

    def test_parameter_frequencies(self):
        """Test parameter_frequencies property"""
        op = Exp(qml.PauliZ(1), 1j)
        assert op.parameter_frequencies == [(2,)]

    def test_parameter_frequencies_raises_error(self):
        """Test that parameter_frequencies raises an error if the op.generator() is undefined"""
        op = Exp(qml.PauliX(0), 1)
        with pytest.raises(GeneratorUndefinedError):
            _ = op.generator()
        with pytest.raises(ParameterFrequenciesUndefinedError):
            _ = op.parameter_frequencies

    def test_parameter_frequency_with_parameters_in_base_operator(self):
        """Test that parameter_frequency raises an error for the Exp class, but not the
        Evolution class, if there are additional parameters in the base operator"""

        base_op = 2 * qml.PauliX(0)
        op1 = Exp(base_op, 1j)
        op2 = Evolution(base_op, 1)

        with pytest.raises(ParameterFrequenciesUndefinedError):
            op1.parameter_frequencies()

        assert op2.parameter_frequencies == [(4.0,)]
