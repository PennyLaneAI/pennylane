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
from scipy.linalg import expm

import pennylane as qp
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.ops.op_math import Evolution, Exp


@pytest.mark.jax
def test_basic_validity():
    """Assert the basic validity of an evolution op."""
    base = qp.prod(qp.PauliX(0), qp.PauliY(1))
    op = Evolution(base, 5.2)
    qp.ops.functions.assert_valid(op)


class TestEvolution:  # pylint: disable=too-many-public-methods
    """Test Evolution(Exp) class that takes a parameter x and a generator G and defines an evolution exp(ixG)"""

    def test_initialization(self):
        """Test initialization with a provided coefficient and a Tensor base."""
        base = qp.PauliZ("b") @ qp.PauliZ("c")
        param = 1.23

        op = Evolution(base, param)

        assert op.base is base
        assert op.coeff == -1j * param
        assert op.name == "Evolution"
        assert isinstance(op, Exp)

        assert op.num_params == 1
        assert op.parameters == [param]
        assert op.data == (param,)

        assert op.wires == qp.wires.Wires(("b", "c"))

    def test_evolution_matches_corresponding_exp(self):
        base_op = 2 * qp.PauliX(0)
        op1 = Exp(base_op, 1j)
        op2 = Evolution(base_op, -1)

        assert np.all(op1.matrix() == op2.matrix())

    def test_has_generator_true(self):
        """Test that has_generator returns True if the coefficient is purely imaginary."""
        U = Evolution(qp.PauliX(0), 3)
        assert U.has_generator is True

    def test_has_generator_false(self):
        """Test that has_generator returns False if the coefficient is not purely imaginary."""
        U = Evolution(qp.PauliX(0), 3j)
        assert U.has_generator is False

        U = Evolution(qp.PauliX(0), 0.01 + 2j)
        assert U.has_generator is False

    def test_generator(self):
        U = Evolution(qp.PauliX(0), 3)
        assert U.generator() == -1 * U.base

    def test_num_params_for_parametric_base(self):
        base_op = 0.5 * qp.PauliY(0) + qp.PauliZ(0) @ qp.PauliX(1)
        op = Evolution(base_op, 1.23)

        assert base_op.num_params == 1
        assert op.num_params == 1

    def test_data(self):
        """Test initializing and accessing the data property."""

        param = np.array(1.234)

        base = qp.PauliX(0)
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
        op = Evolution(qp.PauliX(0), 3)
        # Python 3.13: "Evolution(-3j PauliX)"
        # Python 3.14+: "Evolution((-0-3j) PauliX)"
        assert repr(op) in ["Evolution(-3j PauliX)", "Evolution((-0-3j) PauliX)"]

    def test_repr_tensor(self):
        """Test the __repr__ method when the base is a tensor."""
        t = qp.PauliX(0) @ qp.PauliX(1)
        isingxx = Evolution(t, 0.25)
        assert repr(isingxx) in [
            "Evolution(-0.25j X(0) @ X(1))",
            "Evolution((-0-0.25j) X(0) @ X(1))",
        ]

    def test_repr_deep_operator(self):
        """Test the __repr__ method when the base is any operator with arithmetic depth > 0."""
        base = qp.S(0) @ qp.X(0)
        op = Evolution(base, 3)
        assert repr(op) in ["Evolution(-3j S(0) @ X(0))", "Evolution((-0-3j) S(0) @ X(0))"]

    @pytest.mark.parametrize(
        "op,decimals,expected",
        [
            (Evolution(qp.PauliZ(0), 2), None, "Exp(-2j Z)"),
            (Evolution(qp.PauliZ(0), 2), 2, "Exp(-2.00j Z)"),
            (Evolution(qp.prod(qp.PauliZ(0), qp.PauliY(1)), 2), None, "Exp(-2j Z@Y)"),
            (Evolution(qp.prod(qp.PauliZ(0), qp.PauliY(1)), 2), 2, "Exp(-2.00j Z@Y)"),
            (Evolution(qp.RZ(1.234, wires=[0]), 5.678), None, "Exp(-5.678j RZ)"),
            (Evolution(qp.RZ(1.234, wires=[0]), 5.678), 2, "Exp(-5.68j RZ\n(1.23))"),
        ],
    )
    def test_label(self, op, decimals, expected):
        """Test that the label is informative and uses decimals."""
        assert op.label(decimals=decimals) == expected

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
        op = Evolution(base, 3)
        new_op = op.simplify()

        qp.assert_equal(new_op.base, qp.PauliX(0))
        assert new_op.coeff == -12j

    @pytest.mark.autograd
    def test_sum_generator_default_gradient(self):
        """Backprop gradient of evolve over a composite (Sum) generator with
        scalar coefficients must match a dense matrix-exponential reference.
        Previously this path silently produced an incorrect matrix and gradient."""

        base = 0.5 * qp.X(0) + 0.7 * qp.Z(0)
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(t):
            Evolution(base, t)
            return qp.expval(qp.Z(0))

        h_mat = qp.matrix(base, wire_order=[0])
        z_mat = qp.matrix(qp.Z(0), wire_order=[0])
        psi0 = np.array([1.0, 0.0], dtype=complex)

        def ref(tv):
            u = expm(-1j * tv * h_mat)
            psi = u @ psi0
            return np.real(np.conj(psi) @ z_mat @ psi)

        t = np.array(0.123, requires_grad=True)
        eps = 1e-6
        ref_grad = (ref(0.123 + eps) - ref(0.123 - eps)) / (2 * eps)

        assert qp.math.allclose(circuit(t), ref(0.123))
        assert qp.math.allclose(qp.grad(circuit)(t), ref_grad, atol=1e-5)

    @pytest.mark.autograd
    def test_qaoa_evolve_matches_approx_time_evolution(self):
        """Reproduces the originally reported QAOA workflow at [sc-119491]: a multi-layer QAOA circuit
        built from ``qp.evolve`` of ``Sum`` cost/mixer Hamiltonians silently gave
        wrong forward values and gradients under the default differentiation path,
        diverging during optimization. ``ApproxTimeEvolution(H, t, n=1)`` (the 
        workaround at [sc-119491]) is exact here because the terms within each cost/mixer block
        commute, so it is used as the reference for both value and gradient."""
        edges = [[0, 1], [1, 2], [0, 2], [2, 3]]
        wires = sorted({w for edge in edges for w in edge})
        num_wires = len(wires)

        cost_terms = []
        for i, j in edges:
            cost_terms += [0.75 * (qp.Z(i) @ qp.Z(j)), 0.75 * qp.Z(i), 0.75 * qp.Z(j)]
        cost_terms += [-1.0 * qp.Z(w) for w in wires]
        cost_ham = qp.sum(*cost_terms)
        mixer_ham = qp.sum(*(qp.X(w) for w in wires))

        dev = qp.device("default.qubit", wires=num_wires)

        @qp.qnode(dev)
        def circuit_evolve(params):
            for w in range(num_wires):
                qp.Hadamard(wires=w)
            for gamma, beta in params:
                qp.evolve(cost_ham, gamma)
                qp.evolve(mixer_ham, beta)
            return qp.expval(cost_ham)

        @qp.qnode(dev)
        def circuit_ate(params):
            for w in range(num_wires):
                qp.Hadamard(wires=w)
            for gamma, beta in params:
                qp.ApproxTimeEvolution(cost_ham, gamma, n=1)
                qp.ApproxTimeEvolution(mixer_ham, beta, n=1)
            return qp.expval(cost_ham)

        params = np.array([[0.5, 0.5] for _ in range(4)], requires_grad=True)

        assert qp.math.allclose(circuit_evolve(params), circuit_ate(params))
        assert qp.math.allclose(qp.grad(circuit_evolve)(params), qp.grad(circuit_ate)(params))

    @pytest.mark.jax
    def test_parameter_shift_gradient_matches_jax(self):
        import jax

        dev = qp.device("default.qubit", wires=2)
        base = qp.PauliX(0)
        x = np.array(1.234)

        @qp.qnode(dev, diff_method=qp.gradients.param_shift)
        def circ_param_shift(x):
            Evolution(base, -0.5 * x)
            return qp.expval(qp.PauliZ(0))

        @qp.qnode(qp.device("default.qubit"), interface="jax")
        def circ(x):
            Evolution(qp.PauliX(0), -0.5 * x)
            return qp.expval(qp.PauliZ(0))

        grad_param_shift = qp.grad(circ_param_shift)(x)
        grad = jax.grad(circ)(x)

        assert qp.math.allclose(grad, grad_param_shift)

    def test_generator_warns_if_not_hermitian(self):
        base = qp.s_prod(1j, qp.Identity(0))
        op = Evolution(base, 2)
        with pytest.warns(UserWarning, match="may not be hermitian"):
            op.generator()

    def test_simplifying_Evolution_operator(self):
        base = qp.PauliX(0) + qp.PauliX(1) + qp.PauliX(0)
        op = Evolution(base, 2)

        qp.assert_equal(op.simplify(), Evolution(base.simplify(), 2))

    @pytest.mark.parametrize(
        "base",
        [
            qp.pow(qp.PauliX(0) + qp.PauliY(1)),
            qp.adjoint(qp.PauliZ(2)),
            qp.s_prod(0.5, qp.PauliX(0)),
        ],
    )
    def test_generator_not_observable_class(self, base):
        """Test that qp.generator will return generator if it is_hermitian, but is not a subclass of Observable"""
        op = Evolution(base, 1)
        gen, c = qp.generator(op)
        qp.assert_equal(gen if c == 1 else qp.s_prod(c, gen), qp.s_prod(-1, base))

    def test_generator_error_if_not_hermitian(self):
        """Tests that an error is raised if the generator is not hermitian."""
        op = Evolution(qp.RX(np.pi / 3, 0), 1)

        with pytest.raises(QuantumFunctionError, match="of operation Evolution is not hermitian"):
            qp.generator(op)

    def test_generator_undefined_error(self):
        """Tests that an error is raised if the generator of an Evolution operator is requested
        with a non-zero complex term in the operator parameter."""
        param = 1 + 2.5j
        op = Evolution(qp.PauliZ(0), param)

        with pytest.raises(
            qp.operation.GeneratorUndefinedError,
            match="is not imaginary; the expected format is exp",
        ):
            _ = op.generator()

    def test_pow_is_evolution(self):
        """Test that Evolution raised to a pow is another Evolution."""

        op = Evolution(qp.Z(0), -0.5)

        pow_op = op.pow(2.5)
        qp.assert_equal(pow_op, Evolution(qp.Z(0), -0.5 * 2.5))
        assert type(pow_op) == Evolution  # pylint: disable=unidiomatic-typecheck


@pytest.mark.integration
@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize(
    "coeff, hamiltonian",
    [
        (0.3, qp.Z(0) @ qp.Y(1)),
        (0.5, 0.1 * qp.Y(0) @ qp.I(1) @ qp.Z(2)),
        (0.3, qp.Z(0)),
    ],
)
def test_pauli_decomposition_integration_graph(coeff, hamiltonian):
    """Tests that the pauli decomposition works in the new graph-based system."""

    op = qp.evolve(hamiltonian, coeff)
    tape = qp.tape.QuantumScript([op])

    [decomp_tape], _ = qp.transforms.decompose(tape, gate_set={"PauliRot"})
    assert len(decomp_tape) == 1
    assert not qp.math.iscomplex(decomp_tape[0].data[0])
    actual_matrix = qp.matrix(decomp_tape, wire_order=op.wires)
    expected_matrix = qp.matrix(op, wire_order=op.wires)
    assert qp.math.allclose(actual_matrix, expected_matrix)
