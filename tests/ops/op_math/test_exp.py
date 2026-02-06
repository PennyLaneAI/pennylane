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
import re

import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.exceptions import (
    DecompositionUndefinedError,
    GeneratorUndefinedError,
    ParameterFrequenciesUndefinedError,
)
from pennylane.ops.op_math import Evolution, Exp


@pytest.mark.parametrize("constructor", (qp.exp, Exp))
class TestInitialization:
    """Test the initialization process and standard properties."""

    def test_pauli_base(self, constructor):
        """Test initialization with no coeff and a simple base."""
        base = qp.PauliX("a")

        op = constructor(base, id="something")

        assert op.base is base
        assert op.coeff == 1
        assert op.name == "Exp"
        assert op.id == "something"

        assert op.num_params == 1
        assert op.parameters == [1]
        assert op.data == (1,)

        assert op.wires == qp.wires.Wires("a")

        assert op.control_wires == qp.wires.Wires([])

    def test_provided_coeff(self, constructor):
        """Test initialization with a provided coefficient and a Tensor base."""
        base = qp.PauliZ("b") @ qp.PauliZ("c")
        coeff = np.array(1.234)

        op = constructor(base, coeff)

        assert op.base is base
        assert op.coeff is coeff
        assert op.name == "Exp"

        assert op.num_params == 1
        assert op.parameters == [coeff]
        assert op.data == (coeff,)

        assert op.wires == qp.wires.Wires(("b", "c"))

    def test_parametric_base(self, constructor):
        """Test initialization with a coefficient and a parametric operation base."""

        base_coeff = 1.23
        base = qp.RX(base_coeff, wires=5)
        coeff = np.array(-2.0)

        op = constructor(base, coeff)

        assert op.base is base
        assert op.coeff is coeff
        assert op.name == "Exp"

        assert op.num_params == 2
        assert op.data == (coeff, base_coeff)

        assert op.wires == qp.wires.Wires(5)

    @pytest.mark.parametrize("value", (True, False))
    def test_has_diagonalizing_gates(self, value, constructor):
        """Test that Exp defers has_diagonalizing_gates to base operator."""

        # pylint: disable=too-few-public-methods
        class DummyOp(qp.operation.Operator):
            num_wires = 1
            has_diagonalizing_gates = value

        op = constructor(DummyOp(1), 2.312)
        assert op.has_diagonalizing_gates is value

    def test_base_is_not_operator_error(self, constructor):
        """Test that Exp raises an error if a base is provided that is not an Operator"""

        with pytest.raises(TypeError, match="base is expected to be of type Operator"):
            constructor(2, qp.PauliX(0))


class TestProperties:
    """Test of the properties of the Exp class."""

    def test_data(self):
        """Test intializaing and accessing the data property."""

        phi = np.array(1.234)
        coeff = np.array(2.345)

        base = qp.RX(phi, wires=0)
        op = Exp(base, coeff)

        assert op.data == (coeff, phi)

        new_phi = np.array(0.1234)
        new_coeff = np.array(3.456)
        op.data = (new_coeff, new_phi)

        assert op.data == (new_coeff, new_phi)
        assert op.base.data == (new_phi,)
        assert op.scalar == new_coeff

    # pylint: disable=protected-access
    def test_queue_category_ops(self):
        """Test the _queue_category property."""
        assert Exp(qp.PauliX(0), -1.234j)._queue_category == "_ops"

        assert Exp(qp.PauliX(0), 1 + 2j)._queue_category == "_ops"

        assert Exp(qp.RX(1.2, 0), -1.2j)._queue_category == "_ops"

    def test_is_verified_hermitian(self):
        """Test that the op is hermitian if the base is hermitian and the coeff is real."""
        assert Exp(qp.PauliX(0), -1.0).is_verified_hermitian

        assert not Exp(qp.PauliX(0), 1.0 + 2j).is_verified_hermitian

        assert not Exp(qp.RX(1.2, wires=0)).is_verified_hermitian

    def test_batching_properties(self):
        """Test the batching properties and methods."""

        # base is batched
        base = qp.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = Exp(base)
        assert op.batch_size == 3

        # coeff is batched
        base = qp.RX(1, 0)
        op = Exp(base, coeff=np.array([1.2, 2.3, 3.4]))
        assert op.batch_size == 3

        # both are batched
        base = qp.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = Exp(base, coeff=np.array([1.2, 2.3, 3.4]))
        assert op.batch_size == 3

    def test_different_batch_sizes_raises_error(self):
        """Test that using different batch sizes for base and scalar raises an error."""
        base = qp.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = Exp(base, np.array([0.1, 1.2, 2.3, 3.4]))
        with pytest.raises(
            ValueError, match="Broadcasting was attempted but the broadcasted dimensions"
        ):
            _ = op.batch_size


class TestMatrix:
    """Test the matrix method."""

    def test_base_batching_support(self):
        """Test that Exp matrix has base batching support."""
        x = np.array([-1, -2, -3])
        op = Exp(qp.RX(x, 0), 3)
        mat = op.matrix()
        true_mat = qp.math.stack([Exp(qp.RX(i, 0), 3).matrix() for i in x])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    def test_coeff_batching_support(self):
        """Test that Exp matrix has coeff batching support."""
        x = np.array([-1, -2, -3])
        op = Exp(qp.PauliX(0), x)
        mat = op.matrix()
        true_mat = qp.math.stack([Exp(qp.PauliX(0), i).matrix() for i in x])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    def test_base_and_coeff_batching_support(self):
        """Test that Exp matrix has base and coeff batching support."""
        x = np.array([-1, -2, -3])
        y = np.array([1, 2, 3])
        op = Exp(qp.RX(x, 0), y)
        mat = op.matrix()
        true_mat = qp.math.stack([Exp(qp.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)

    @pytest.mark.jax
    def test_batching_jax(self):
        """Test that Exp matrix has batching support with the jax interface."""
        import jax.numpy as jnp

        x = jnp.array([-1, -2, -3])
        y = jnp.array([1, 2, 3])
        op = Exp(qp.RX(x, 0), y)
        mat = op.matrix()
        true_mat = qp.math.stack([Exp(qp.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)
        assert isinstance(mat, jnp.ndarray)

    @pytest.mark.torch
    def test_batching_torch(self):
        """Test that Exp matrix has batching support with the torch interface."""
        import torch

        x = torch.tensor([-1, -2, -3])
        y = torch.tensor([1, 2, 3])
        op = Exp(qp.RX(x, 0), y)
        mat = op.matrix()
        true_mat = qp.math.stack([Exp(qp.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)
        assert isinstance(mat, torch.Tensor)

    @pytest.mark.tf
    def test_batching_tf(self):
        """Test that Exp matrix has batching support with the tensorflow interface."""
        import tensorflow as tf

        x = tf.constant([-1.0, -2.0, -3.0])
        y = tf.constant([1.0, 2.0, 3.0])
        op = Exp(qp.RX(x, 0), y)
        mat = op.matrix()
        true_mat = qp.math.stack([Exp(qp.RX(i, 0), j).matrix() for i, j in zip(x, y)])
        assert qp.math.allclose(mat, true_mat)
        assert mat.shape == (3, 2, 2)
        assert isinstance(mat, tf.Tensor)

    def test_tensor_base_isingxx(self):
        """Test that isingxx can be created with a tensor base."""
        phi = -0.46
        base = qp.PauliX(0) @ qp.PauliX(1)
        op = Exp(base, -0.5j * phi)
        isingxx = qp.IsingXX(phi, wires=(0, 1))

        assert qp.math.allclose(op.matrix(), isingxx.matrix())

    def test_prod_base_isingyy(self):
        """Test that IsingYY can be created with a `Prod` base."""
        phi = -0.46
        base = qp.prod(qp.PauliY(0), qp.PauliY(1))
        op = Exp(base, -0.5j * phi)
        isingxx = qp.IsingYY(phi, wires=(0, 1))

        assert qp.math.allclose(op.matrix(), isingxx.matrix())

    @pytest.mark.autograd
    @pytest.mark.parametrize("requires_grad", (True, False))
    def test_matrix_autograd_rx(self, requires_grad):
        """Test the matrix comparing to the rx gate."""
        phi = np.array(1.234, requires_grad=requires_grad)
        exp_rx = Exp(qp.PauliX(0), -0.5j * phi)
        rx = qp.RX(phi, 0)

        assert qp.math.allclose(exp_rx.matrix(), rx.matrix())

    @pytest.mark.autograd
    @pytest.mark.parametrize("requires_grad", (True, False))
    def test_matrix_autograd_rz(self, requires_grad):
        """Test the matrix comparing to the rz gate. This is a gate with an
        autograd coefficient but empty diagonalizing gates."""
        phi = np.array(1.234, requires_grad=requires_grad)
        exp_rz = Exp(qp.PauliZ(0), -0.5j * phi)
        rz = qp.RZ(phi, 0)

        assert qp.math.allclose(exp_rz.matrix(), rz.matrix())

    @pytest.mark.autograd
    @pytest.mark.parametrize("requires_grad", (True, False))
    def test_tensor_with_pauliz_autograd(self, requires_grad):
        """Test the matrix for the case when the coefficient is autograd and
        the diagonalizing gates don't act on every wire for the matrix."""
        phi = qp.numpy.array(-0.345, requires_grad=requires_grad)
        base = qp.PauliZ(0) @ qp.PauliY(1)
        autograd_op = Exp(base, phi)
        mat = qp.math.expm(phi * qp.matrix(base))

        assert qp.math.allclose(autograd_op.matrix(), mat)

    @pytest.mark.autograd
    def test_base_no_diagonalizing_gates_autograd_coeff(self):
        """Test the matrix when the base matrix doesn't define the diagonalizing gates."""
        coeff = np.array(0.4)
        base = qp.RX(2.0, wires=0)
        op = Exp(base, coeff)

        with pytest.warns(UserWarning, match="The autograd matrix for "):
            mat = op.matrix()
        expected = qp.math.expm(coeff * base.matrix())
        assert qp.math.allclose(mat, expected)

    @pytest.mark.torch
    def test_torch_matrix_rx(self):
        """Test the matrix with torch."""
        import torch

        phi = torch.tensor(0.4, dtype=torch.complex128)

        base = qp.PauliX(0)
        op = Exp(base, -0.5j * phi)
        compare = qp.RX(phi, 0)

        assert qp.math.allclose(op.matrix(), compare.matrix())

    @pytest.mark.tf
    def test_tf_matrix_rx(self):
        """Test the matrix with tensorflow."""

        import tensorflow as tf

        phi = tf.Variable(0.4, dtype=tf.complex128)
        base = qp.PauliX(0)
        op = Exp(base, -0.5j * phi)
        compare = qp.RX(phi, wires=0)
        assert qp.math.allclose(op.matrix(), compare.matrix())

    @pytest.mark.jax
    def test_jax_matrix_rx(self):
        """Test the matrix with jax."""
        import jax

        phi = jax.numpy.array(0.4 + 0j)

        base = qp.PauliX(0)
        op = Exp(base, -0.5j * phi)
        compare = qp.RX(phi, 0)

        assert qp.math.allclose(op.matrix(), compare.matrix())

        def exp_mat(x):
            return qp.exp(base, -0.5j * x).matrix()

        def rx_mat(x):
            return qp.RX(x, wires=0).matrix()

        exp_mat_grad = jax.jacobian(exp_mat, holomorphic=True)(phi)
        rx_mat_grad = jax.jacobian(rx_mat, holomorphic=True)(phi)

        assert qp.math.allclose(exp_mat_grad, rx_mat_grad)

    def test_sparse_matrix(self):
        """Test the sparse matrix function."""
        from scipy.sparse import csr_matrix

        H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
        H = csr_matrix(H)
        base = qp.SparseHamiltonian(H, wires=0)

        op = Exp(base, 3)

        sp_format = "lil"
        sparse_mat = op.sparse_matrix(format=sp_format)
        assert sparse_mat.format == sp_format

        dense_mat = qp.matrix(op)

        assert qp.math.allclose(sparse_mat.toarray(), dense_mat)

    def test_sparse_matrix_wire_order_error(self):
        """Test that sparse_matrix raises an error if wire_order provided."""
        from scipy.sparse import csr_matrix

        H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
        H = csr_matrix(H)
        base = qp.SparseHamiltonian(H, wires=0)

        op = Exp(base, 3)

        with pytest.raises(NotImplementedError):
            op.sparse_matrix(wire_order=[0, 1])


class TestDecomposition:
    """Test the decomposition of the `Exp` gate."""

    # Order of `qp.ops.qubit.__all__` is not reliable, so
    # must sort for consistent order in testing with multiple
    # workers
    all_qubit_operators = sorted(qp.ops.qubit.__all__)  # pylint: disable=no-member

    def test_sprod_decomposition(self):
        """Test that the exp of an SProd has a decomposition."""
        op = Exp(qp.s_prod(3, qp.PauliX(0)), 1j)
        assert op.has_decomposition
        [decomp] = op.decomposition()
        qp.assert_equal(decomp, qp.RX(-6.0, 0))

    def test_imaginary_coeff_in_operator(self):
        """Test that the operator can be decomposed if the imaginary coeff is in the operator instead."""

        op = Exp(1j * qp.Z(0) @ qp.Z(1))
        assert op.has_decomposition
        [decomp] = op.decomposition()
        qp.assert_equal(decomp, qp.IsingZZ(-2.0, wires=(0, 1)))

    @pytest.mark.parametrize("coeff", (1, 1 + 0.5j))
    def test_non_imag_no_decomposition(self, coeff):
        """Tests that the decomposition doesn't exist if the coefficient has a real component."""
        op = Exp(qp.PauliX(0), coeff)
        assert not op.has_decomposition
        with pytest.raises(
            DecompositionUndefinedError,
            match="Decomposition is not defined for real coefficients of hermitian operators.",
        ):
            op.decomposition()

    def test_non_pauli_word_base_no_decomposition(self):
        """Tests that the decomposition doesn't exist if the base is not a pauli word."""
        op = Exp(qp.S(0), -0.5j)
        assert not op.has_decomposition
        with pytest.raises(
            DecompositionUndefinedError,
            match=re.escape(f"The decomposition of the {op} operator is not defined."),
        ):
            op.decomposition()

        op = Exp(2 * qp.S(0) + qp.PauliZ(1), -0.5j)
        assert not op.has_decomposition
        with pytest.raises(
            DecompositionUndefinedError,
            match=re.escape(f"The decomposition of the {op} operator is not defined."),
        ):
            op.decomposition()

    @pytest.mark.parametrize(
        "base, base_string, wires",
        (
            (qp.prod(qp.PauliZ(0), qp.PauliY(1)), "ZY", (0, 1)),
            (qp.prod(qp.PauliY(0), qp.Identity(1), qp.PauliZ(2)), "YZ", (0, 2)),
        ),
    )
    def test_decomposition_into_pauli_rot(self, base, base_string, wires):
        """Check that Exp decomposes into PauliRot if base is a pauli word with more than one term."""
        theta = 3.21
        op = Exp(base, -0.5j * theta)

        assert op.has_decomposition
        pr = op.decomposition()[0]
        qp.assert_equal(pr, qp.PauliRot(3.21, base_string, wires))

    @pytest.mark.parametrize("op_name", all_qubit_operators)
    @pytest.mark.parametrize("str_wires", (True, False))
    def test_generator_decomposition(self, op_name, str_wires):
        """Check that Exp decomposes into a specific operator if ``base`` corresponds to the
        generator of that operator."""

        op_class = getattr(qp.ops.qubit, op_name)  # pylint:disable=no-member

        if not op_class.has_generator:
            pytest.skip("Operator does not have a generator.")

        if op_class in {qp.DoubleExcitationMinus, qp.DoubleExcitationPlus}:
            pytest.skip("qp.equal doesn't work for `SparseHamiltonian` generators.")

        if op_class is qp.PCPhase:
            pytest.skip(
                "`PCPhase` decompositions not currently possible due to different signature."
            )

        phi = 1.23

        wires = [0, 1, 2] if op_class.num_wires is None else list(range(op_class.num_wires))
        if str_wires:
            alphabet = ("a", "b", "c", "d", "e", "f", "g")
            wires = [alphabet[w] for w in wires]

        # PauliRot and PCPhase each have an extra required arg
        if op_class is qp.PauliRot:
            op = op_class(phi, pauli_word="XYZ", wires=wires)
        else:
            op = op_class(phi, wires=wires)

        exp = qp.evolve(op.generator(), coeff=-phi)
        dec = exp.decomposition()
        assert len(dec) == 1
        if op_class in {qp.PhaseShift, qp.U1}:
            # These operators have the same generator so when reconstructing from
            # the generator, cannot predict which will be returned
            assert (
                isinstance(dec[0], (qp.PhaseShift, qp.U1))
                and qp.math.isclose(dec[0].data[0], phi)
                and dec[0].wires == op.wires
            )
        elif op_class is qp.GlobalPhase:
            # exp(qp.GlobalPhase.generator(), phi) decomposes to PauliRot
            # cannot compare GlobalPhase and PauliRot with qp.equal
            assert np.allclose(op.matrix(wire_order=op.wires), dec[0].matrix(wire_order=op.wires))
        elif op_class is qp.FermionicSWAP:
            expected = op.map_wires(dict(zip(op.wires, reversed(op.wires))))
            # simplifying the generator changes the wire order
            qp.assert_equal(expected, dec[0])
        else:
            qp.assert_equal(op, dec[0])

    def test_real_coeff_error(self):
        """Test that the decomposition raises an error if the coefficient has non-zero real part"""
        op = qp.exp(qp.sum(qp.PauliX(0), qp.PauliY(1)), 1.23 + 0.5j)
        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()

    @pytest.mark.integration
    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize(
        "coeff, hamiltonian",
        [
            (0.3j, qp.Z(0) @ qp.Y(1)),
            (0.5, 0.1j * qp.Y(0) @ qp.I(1) @ qp.Z(2)),
            (0.3j, qp.Z(0)),
        ],
    )
    def test_pauli_decomposition_integration_graph(self, coeff, hamiltonian):
        """Tests that the pauli decomposition works in the new graph-based system."""

        op = qp.exp(hamiltonian, coeff)
        tape = qp.tape.QuantumScript([op])

        [decomp_tape], _ = qp.transforms.decompose(tape, gate_set={"PauliRot"})
        assert len(decomp_tape) == 1
        actual_matrix = qp.matrix(decomp_tape, wire_order=op.wires)
        expected_matrix = qp.matrix(op, wire_order=op.wires)
        assert qp.math.allclose(actual_matrix, expected_matrix)


class TestMiscMethods:
    """Test other representation methods."""

    def test_repr_paulix(self):
        """Test the __repr__ method when the base is a simple observable."""
        op = Exp(qp.PauliX(0), 3)
        assert repr(op) == "Exp(3 PauliX)"

    # pylint: disable=protected-access
    @pytest.mark.parametrize("op_class", [Exp, Evolution])
    def test_flatten_unflatten(self, op_class):
        """Tests the _unflatten and _flatten methods for the Exp and Evolution operators."""
        base = qp.RX(1.2, wires=0)
        op = op_class(base, 2.5)

        data, _ = op._flatten()
        assert data[0] is base
        assert data[1] == 2.5

        new_op = type(op)._unflatten(*op._flatten())
        qp.assert_equal(new_op, op)

    def test_repr_tensor(self):
        """Test the __repr__ method when the base is a tensor."""
        t = qp.PauliX(0) @ qp.PauliX(1)
        isingxx = Exp(t, 0.25j)

        assert repr(isingxx) == "Exp(0.25j X(0) @ X(1))"

    def test_repr_deep_operator(self):
        """Test the __repr__ method when the base is any operator with arithmetic depth > 0."""
        base = qp.S(0) @ qp.PauliX(0)
        op = qp.ops.Exp(base, 3)  # pylint:disable=no-member

        assert repr(op) == "Exp(3 S(0) @ X(0))"

    def test_diagonalizing_gates(self):
        """Test that the diagonalizing gates are the same as the base diagonalizing gates."""
        base = qp.PauliX(0)
        op = Exp(base, 1 + 2j)
        for op1, op2 in zip(base.diagonalizing_gates(), op.diagonalizing_gates()):
            qp.assert_equal(op1, op2)

    def test_pow(self):
        """Test the pow decomposition method."""
        base = qp.PauliX(0)
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
            (Exp(qp.PauliZ(0), 2 + 3j), None, "Exp((2+3j) Z)"),
            (Exp(qp.PauliZ(0), 2 + 3j), 2, "Exp(2.00+3.00j Z)"),
            (Exp(qp.prod(qp.PauliZ(0), qp.PauliY(1)), 2 + 3j), None, "Exp((2+3j) Z@Y)"),
            (Exp(qp.prod(qp.PauliZ(0), qp.PauliY(1)), 2 + 3j), 2, "Exp(2.00+3.00j Z@Y)"),
            (Exp(qp.RZ(1.234, wires=[0]), 5.678), None, "Exp(5.678 RZ)"),
            (Exp(qp.RZ(1.234, wires=[0]), 5.678), 2, "Exp(5.68 RZ\n(1.23))"),
        ],
    )
    def test_label(self, op, decimals, expected):
        """Test that the label is informative and uses decimals."""
        assert op.label(decimals=decimals) == expected

    def test_simplify_sprod(self):
        """Test that simplify merges SProd into the coefficent."""
        base = qp.adjoint(qp.PauliX(0))
        s_op = qp.s_prod(2.0, base)

        op = Exp(s_op, 3j)
        new_op = op.simplify()
        qp.assert_equal(new_op.base, qp.PauliX(0))
        assert new_op.coeff == 6.0j

    def test_simplify(self):
        """Test that the simplify method simplifies the base."""
        orig_base = qp.adjoint(qp.adjoint(qp.PauliX(0)))

        op = Exp(orig_base, coeff=0.2)
        new_op = op.simplify()
        qp.assert_equal(new_op.base, qp.PauliX(0))
        assert new_op.coeff == 0.2

    def test_simplify_s_prod(self):
        """Tests that when simplification of the base results in an SProd,
        the scalar is included in the coeff rather than the base"""
        base = qp.s_prod(2, qp.sum(qp.PauliX(0), qp.PauliX(0)))
        op = Exp(base, 3)
        new_op = op.simplify()

        qp.assert_equal(new_op.base, qp.PauliX(0))
        assert new_op.coeff == 12
        assert new_op is not op

    def test_copy(self):
        """Tests making a copy."""
        op = Exp(qp.CNOT([0, 1]), 2)
        copied_op = copy.copy(op)

        qp.assert_equal(op.base, copied_op.base)
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

        @qp.qnode(qp.device("default.qubit", wires=1))
        def circ(phi):
            Exp(qp.PauliX(0), -0.5j * phi)
            return qp.expval(qp.PauliZ(0))

        res = circ(phi)
        assert qp.math.allclose(res, jnp.cos(phi))
        grad = jax.grad(circ)(phi)
        assert qp.math.allclose(grad, -jnp.sin(phi))

    @pytest.mark.catalyst
    @pytest.mark.external
    def test_catalyst_qnode(self):
        """Test with Catalyst interface"""

        pytest.importorskip("catalyst")

        phi = 0.345

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def func(params):
            qp.exp(qp.X(0), -0.5j * params)
            return qp.expval(qp.Z(0))

        res = func(phi)
        assert qp.math.allclose(res, np.cos(phi))
        grad = qp.grad(func)(phi)
        assert qp.math.allclose(grad, -np.sin(phi))

    @pytest.mark.xfail(
        reason="change in lightning broke this test. Temporary patch to unblock CI", strict=False
    )
    @pytest.mark.jax
    def test_jax_jit_qnode(self):
        """Tests with jax.jit"""

        import jax
        from jax import numpy as jnp

        phi = jnp.array(0.345)

        @jax.jit
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def func(params):
            qp.exp(qp.X(0), -0.5j * params)
            return qp.expval(qp.Z(0))

        res = func(phi)
        assert qp.math.allclose(res, jnp.cos(phi))
        grad = jax.grad(func)(phi)
        assert qp.math.allclose(grad, -jnp.sin(phi))

    @pytest.mark.tf
    def test_tensorflow_qnode(self):
        """test the execution of a tensorflow qnode."""
        import tensorflow as tf

        phi = tf.Variable(1.2, dtype=tf.complex128)

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circ(phi):
            Exp(qp.PauliX(0), -0.5j * phi)
            return qp.expval(qp.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circ(phi)

        phi_grad = tape.gradient(res, phi)
        phi_real = qp.math.cast(phi, tf.float64)

        assert qp.math.allclose(res, tf.cos(phi_real))
        # pylint: disable=invalid-unary-operand-type
        assert qp.math.allclose(phi_grad, -tf.sin(phi))

    @pytest.mark.torch
    def test_torch_qnode(self):
        """Test execution with torch."""
        import torch

        phi = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(phi):
            Exp(qp.PauliX(0), -0.5j * phi)
            return qp.expval(qp.PauliZ(0))

        res = circuit(phi)
        assert qp.math.allclose(res, torch.cos(phi))

        res.backward()  # pylint:disable=no-member
        assert qp.math.allclose(phi.grad, -torch.sin(phi))

    @pytest.mark.autograd
    def test_autograd_qnode(self):
        """Test execution and gradient with pennylane numpy array."""
        phi = qp.numpy.array(1.2)

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(phi):
            Exp(qp.PauliX(0), -0.5j * phi)
            return qp.expval(qp.PauliZ(0))

        res = circuit(phi)
        assert qp.math.allclose(res, qp.numpy.cos(phi))

        grad = qp.grad(circuit)(phi)
        assert qp.math.allclose(grad, -qp.numpy.sin(phi))

    @pytest.mark.xfail  # related to #6333
    @pytest.mark.autograd
    def test_autograd_param_shift_qnode(self):
        """Test execution and gradient with pennylane numpy array."""

        phi = qp.numpy.array(1.2)

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev, diff_method=qp.gradients.param_shift)
        def circuit(phi):
            Exp(qp.PauliX(0), -0.5j * phi)
            return qp.expval(qp.PauliZ(0))

        res = circuit(phi)
        assert qp.math.allclose(res, qp.numpy.cos(phi))

        grad = qp.grad(circuit)(phi)
        assert qp.math.allclose(grad, -qp.numpy.sin(phi))

    @pytest.mark.autograd
    def test_autograd_measurement(self):
        """Test exp in a measurement with gradient and autograd."""

        x = qp.numpy.array(2.0)

        @qp.qnode(qp.device("default.qubit", wires=1))
        def circuit(x):
            qp.Hadamard(0)
            return qp.expval(Exp(qp.PauliZ(0), x))

        res = circuit(x)
        expected = 0.5 * (np.exp(x) + np.exp(-x))
        assert qp.math.allclose(res, expected)

        grad = qp.grad(circuit)(x)
        expected_grad = 0.5 * (np.exp(x) - np.exp(-x))
        assert qp.math.allclose(grad, expected_grad)

    @pytest.mark.torch
    def test_torch_measurement(self):
        """Test Exp in a measurement with gradient and torch."""

        import torch

        x = torch.tensor(2.0, requires_grad=True, dtype=float)

        @qp.qnode(qp.device("default.qubit", wires=1))
        def circuit(x):
            qp.Hadamard(0)
            return qp.expval(Exp(qp.PauliZ(0), x))

        res = circuit(x)
        expected = 0.5 * (torch.exp(x) + torch.exp(-x))
        assert qp.math.allclose(res, expected)

        res.backward()  # pylint:disable=no-member
        expected_grad = 0.5 * (torch.exp(x) - torch.exp(-x))
        assert qp.math.allclose(x.grad, expected_grad)

    @pytest.mark.jax
    def test_jax_measurement(self):
        """Test Exp in a measurement with gradient and jax."""

        import jax
        from jax import numpy as jnp

        x = jnp.array(2.0)

        @qp.qnode(qp.device("default.qubit", wires=1))
        def circuit(x):
            qp.Hadamard(0)
            return qp.expval(Exp(qp.PauliZ(0), x))

        res = circuit(x)
        expected = 0.5 * (jnp.exp(x) + jnp.exp(-x))
        assert qp.math.allclose(res, expected)

        grad = jax.grad(circuit)(x)
        expected_grad = 0.5 * (jnp.exp(x) - jnp.exp(-x))
        assert qp.math.allclose(grad, expected_grad)

    @pytest.mark.tf
    def test_tf_measurement(self):
        """Test Exp in a measurement with gradient and tensorflow."""
        # pylint:disable=invalid-unary-operand-type
        import tensorflow as tf

        x = tf.Variable(2.0, dtype=tf.float64)

        @qp.qnode(qp.device("default.qubit", wires=1))
        def circuit(x):
            qp.Hadamard(0)
            return qp.expval(Exp(qp.PauliZ(0), x))

        with tf.GradientTape() as tape:
            res = circuit(x)

        expected = 0.5 * (tf.exp(x) + tf.exp(-x))
        assert qp.math.allclose(res, expected)

        x_grad = tape.gradient(res, x)
        expected_grad = 0.5 * (tf.exp(x) - tf.exp(-x))
        assert qp.math.allclose(x_grad, expected_grad)

    def test_draw_integration(self):
        """Test that Exp integrates with drawing."""

        phi = qp.numpy.array(1.2)

        with qp.queuing.AnnotatedQueue() as q:
            Exp(qp.PauliX(0), -0.5j * phi)

        tape = qp.tape.QuantumScript.from_queue(q)

        assert qp.drawer.tape_text(tape) == "0: ──Exp(-0.6j X)─┤  "

    def test_exp_batching(self):
        """Test execution of a batched Exp operator."""
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(x):
            Exp(qp.PauliX(0), 0.5j * x)
            return qp.expval(qp.PauliY(0))

        x = qp.numpy.array([1.234, 2.34, 3.456])
        res = circuit(x)

        expected = np.sin(x)
        assert qp.math.allclose(res, expected)


class TestDifferentiation:
    """Test generator and parameter_frequency for differentiation"""

    def test_base_not_hermitian_generator_undefined(self):
        """That that imaginary coefficient but non-Hermitian base operator raises GeneratorUndefinedError"""
        op = Exp(qp.RX(1.23, 0), 1j)
        with pytest.raises(GeneratorUndefinedError):
            op.generator()

    def test_has_generator_true(self):
        """Test that has_generator returns True if the coefficient is
        purely imaginary and the base is Hermitian."""
        op = Exp(qp.PauliX(0), 1j)
        assert op.has_generator is True

    def test_base_not_hermitian_has_generator_false(self):
        """Test that has_generator returns False if the coefficient is
        purely imaginary but the base is not Hermitian."""
        op = Exp(qp.RX(1.23, 0), 1j)
        assert op.has_generator is False

    def test_real_component_has_generator_false(self):
        """Test that has_generator returns False if the coefficient is not purely imaginary."""
        op = Exp(qp.PauliX(0), 3)
        assert op.has_generator is False

        op = Exp(qp.PauliX(0), 0.01 + 2j)
        assert op.has_generator is False

    def test_real_component_coefficient_generator_undefined(self):
        """Test that Hermitian base operator but real coefficient raises GeneratorUndefinedError"""
        op = Exp(qp.PauliX(0), 1)
        with pytest.raises(GeneratorUndefinedError):
            op.generator()

    def test_generator_is_base_operator(self):
        """Test that generator is base operator"""
        base_op = qp.PauliX(0)
        op = Exp(base_op, 1j)
        assert op.base == op.generator()

    def test_parameter_frequencies(self):
        """Test parameter_frequencies property"""
        op = Exp(qp.PauliZ(1), 1j)
        assert op.parameter_frequencies == [(2,)]

    def test_parameter_frequencies_raises_error(self):
        """Test that parameter_frequencies raises an error if the op.generator() is undefined"""
        op = Exp(qp.PauliX(0), 1)
        with pytest.raises(GeneratorUndefinedError):
            _ = op.generator()
        with pytest.raises(ParameterFrequenciesUndefinedError):
            _ = op.parameter_frequencies

    def test_parameter_frequency_with_parameters_in_base_operator(self):
        """Test that parameter_frequency raises an error for the Exp class, but not the
        Evolution class, if there are additional parameters in the base operator"""

        base_op = 2 * qp.PauliX(0)
        op1 = Exp(base_op, 1j)
        op2 = Evolution(base_op, 1)

        with pytest.raises(ParameterFrequenciesUndefinedError):
            _ = op1.parameter_frequencies

        assert op2.parameter_frequencies == [(4.0,)]

    def test_params_can_be_considered_trainable(self):
        """Tests that the parameters of an Exp are considered trainable."""
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(x, coeff):
            Exp(qp.RX(x, 0), coeff)
            return qp.expval(qp.PauliZ(0))

        with pytest.warns(UserWarning):
            circuit(np.array(2.0), np.array(0.5))

        tape = qp.workflow.construct_tape(circuit)(np.array(2.0), np.array(0.5))
        assert tape.trainable_params == [0, 1]
