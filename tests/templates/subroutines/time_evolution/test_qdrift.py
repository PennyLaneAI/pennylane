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
"""
Tests for the QDrift template.
"""
import copy
from functools import reduce

import pytest

import pennylane as qp
from pennylane import numpy as qnp
from pennylane.exceptions import QuantumFunctionError
from pennylane.math import allclose, isclose
from pennylane.templates.subroutines.time_evolution.qdrift import _sample_decomposition

test_hamiltonians = (
    (
        [1, 1, 1],
        [qp.PauliX(0), qp.PauliY(0), qp.PauliZ(1)],
    ),
    (
        [1.23, -0.45],
        [
            qp.s_prod(0.1, qp.PauliX(0)),
            qp.prod(qp.PauliZ(0), qp.PauliX(1)),
        ],  #  Here we chose such hamiltonian to have non-commutability
    ),  # op arith
    (
        [1, -0.5, 0.5],
        [qp.Identity(wires=[0, 1]), qp.PauliZ(0), qp.PauliZ(1)],
    ),
)


class TestInitialization:
    """Test that the class is intialized correctly."""

    def test_queuing(self):
        """Test that QDrift de-queues the input hamiltonian."""

        with qp.queuing.AnnotatedQueue() as q:
            H = qp.X(0) + qp.Y(1)
            op = qp.QDrift(H, 0.1, n=20)

        assert len(q.queue) == 1
        assert q.queue[0] is op

    @pytest.mark.jax
    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_init_correctly(self, coeffs, ops, time, n, seed):  # pylint: disable=too-many-arguments
        """Test that all of the attributes are initialized correctly."""
        h = qp.dot(coeffs, ops)
        op = qp.QDrift(h, time, n=n, seed=seed)

        if seed is not None:
            # For seed = None, decomposition and compute_decomposition do not match because
            # compute_decomposition is stochastic
            qp.ops.functions.assert_valid(op, skip_differentiation=True)

        assert op.wires == h.wires
        assert op.parameters == [*h.data, time]
        assert op.data == (*h.data, time)

        assert op.hyperparameters["n"] == n
        assert op.hyperparameters["seed"] == seed
        assert op.hyperparameters["base"] == h

    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_copy(self, coeffs, ops, time, n, seed):  # pylint: disable=too-many-arguments
        """Test that we can make copies of QDrift correctly."""
        h = qp.dot(coeffs, ops)
        op = qp.QDrift(h, time, n=n, seed=seed)
        new_op = copy.copy(op)

        assert op.wires == new_op.wires
        assert op.parameters == new_op.parameters
        assert op.data == new_op.data
        assert op.hyperparameters == new_op.hyperparameters
        assert op is not new_op

    @pytest.mark.parametrize(
        "hamiltonian, raise_error",
        (
            (qp.PauliX(0), True),
            (qp.prod(qp.PauliX(0), qp.PauliZ(1)), True),
            (qp.Hamiltonian([1.23, 3.45], [qp.PauliX(0), qp.PauliZ(1)]), False),
            (qp.dot([1.23, 3.45], [qp.PauliX(0), qp.PauliZ(1)]), False),
        ),
    )
    def test_error_type(self, hamiltonian, raise_error):
        """Test an error is raised of an incorrect type is passed"""
        if raise_error:
            with pytest.raises(TypeError, match="The given operator must be a PennyLane ~.Sum"):
                qp.QDrift(hamiltonian, time=1.23)
        else:
            try:
                qp.QDrift(hamiltonian, time=1.23)
            except TypeError:
                assert False  # test should fail if an error was raised when we expect it not to

    def test_error_hamiltonian(self):
        """Test that a hamiltonian must have at least 2 terms to be supported."""
        msg = "There should be at least 2 terms in the Hamiltonian."
        with pytest.raises(ValueError, match=msg):
            h = qp.Hamiltonian([1.0], [qp.PauliX(0)])
            qp.QDrift(h, 1.23, n=2, seed=None)


class TestDecomposition:
    """Test decompositions are generated correctly."""

    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_private_sample(self, coeffs, ops, time, seed, n):  # pylint: disable=too-many-arguments
        """Test the private function which samples the decomposition"""
        ops_to_coeffs = dict(zip(ops, coeffs))
        normalization = qnp.sum(qnp.abs(coeffs))

        with qp.tape.QuantumTape() as tape:
            decomp = _sample_decomposition(coeffs, ops, time, n, seed)

        assert len(decomp) == n
        assert len(tape.operations) == 0  # no queuing
        for term in decomp:
            coeff = ops_to_coeffs[term.base]
            s = coeff / qp.math.abs(coeff)

            assert term.base in ops  # sample from ops
            assert term.coeff == (s * normalization * time * 1j / n)  # with this exponent

    @pytest.mark.parametrize("coeffs", ([0.999, 0.001], [0.5 + 0.499j, -0.001j]))
    def test_private_sample_statistics(self, coeffs, seed):
        """Test the private function samples from the right distribution"""
        ops = [qp.PauliX(0), qp.PauliZ(1)]
        decomp = _sample_decomposition(coeffs, ops, 1.23, n=10, seed=seed)

        # High probability we only sample PauliX!
        assert all(isinstance(op.base, qp.PauliX) for op in decomp)

    def test_compute_decomposition(self, seed):
        """Test that the decomposition is computed and queues correctly."""
        coeffs = [1, -0.5, 0.5]
        ops = [qp.Identity(wires=[0, 1]), qp.PauliZ(0), qp.PauliZ(1)]

        h = qp.dot(coeffs, ops)
        op = qp.QDrift(h, time=1.23, n=10, seed=seed)

        expected_decomp = _sample_decomposition(coeffs, ops, 1.23, 10, seed=seed)

        with qp.tape.QuantumTape() as tape:
            decomp = op.compute_decomposition(*op.parameters, **op.hyperparameters)

        assert decomp == tape.operations  # queue matches decomp with circuit ordering
        assert decomp == list(expected_decomp)  # sample the same ops

        # Decompositions of an instance are maintained across calls to `compute_decomposition`
        with qp.tape.QuantumTape() as second_tape:
            second_decomp = op.compute_decomposition(*op.parameters, **op.hyperparameters)

        assert second_tape.operations == tape.operations
        assert second_decomp == decomp


class TestIntegration:
    """Test that the QDrift template integrates well with the rest of PennyLane"""

    @pytest.mark.local_salt(3)
    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_execution(self, coeffs, ops, time, n, seed):  # pylint: disable=too-many-arguments
        """Test that the circuit executes as expected"""
        hamiltonian = qp.dot(coeffs, ops)
        wires = hamiltonian.wires
        dev = qp.device("reference.qubit", wires=wires)

        @qp.qnode(dev)
        def circ():
            qp.QDrift(hamiltonian, time, n=n, seed=seed)
            return qp.state()

        expected_decomp = _sample_decomposition(coeffs, ops, time, n=n, seed=seed)

        initial_state = qnp.zeros(2 ** (len(wires)))
        initial_state[0] = 1

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qp.matrix(op, wire_order=wires) for op in expected_decomp[::-1]],
            )
            @ initial_state
        )
        state = circ()

        assert allclose(state, expected_state)

    @pytest.mark.autograd
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_execution_autograd(self, coeffs, ops, seed):
        """Test that the circuit executes as expected using autograd"""

        time = qnp.array(0.5)
        coeffs = qnp.array(coeffs, requires_grad=False)

        dev = qp.device("reference.qubit", wires=[0, 1])

        @qp.qnode(dev)
        def circ(time):
            hamiltonian = qp.dot(coeffs, ops)
            qp.QDrift(hamiltonian, time, n=2, seed=seed)
            return qp.state()

        expected_decomp = _sample_decomposition(coeffs, ops, time, n=2, seed=seed)

        initial_state = qnp.array([1.0, 0.0, 0.0, 0.0])

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qp.matrix(op, wire_order=[0, 1]) for op in expected_decomp[::-1]],
            )
            @ initial_state
        )
        state = circ(time)

        assert allclose(expected_state, state)

    @pytest.mark.torch
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_execution_torch(self, coeffs, ops, seed):
        """Test that the circuit executes as expected using torch"""
        import torch

        time = torch.tensor(0.5, dtype=torch.complex128, requires_grad=True)
        dev = qp.device("default.qubit", wires=[0, 1])

        @qp.qnode(dev)
        def circ(time):
            hamiltonian = qp.dot(coeffs, ops)
            qp.QDrift(hamiltonian, time, n=2, seed=seed)
            return qp.state()

        expected_decomp = _sample_decomposition(coeffs, ops, time, n=2, seed=seed)

        initial_state = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qp.matrix(op, wire_order=[0, 1]) for op in expected_decomp[::-1]],
            )
            @ initial_state
        )
        state = circ(time)

        assert allclose(expected_state, state)

    @pytest.mark.tf
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_execution_tf(self, coeffs, ops, seed):
        """Test that the circuit executes as expected using tensorflow"""
        import tensorflow as tf

        time = tf.Variable(0.5, dtype=tf.complex128)
        dev = qp.device("default.qubit", wires=[0, 1])

        @qp.qnode(dev)
        def circ(time):
            hamiltonian = qp.dot(coeffs, ops)
            qp.QDrift(hamiltonian, time, n=2, seed=seed)
            return qp.state()

        expected_decomp = _sample_decomposition(coeffs, ops, time, n=2, seed=seed)

        initial_state = tf.Variable([1.0, 0.0, 0.0, 0.0], dtype=tf.complex128)

        expected_state = tf.linalg.matvec(
            reduce(
                lambda x, y: x @ y,
                [qp.matrix(op, wire_order=[0, 1]) for op in expected_decomp[::-1]],
            ),
            initial_state,
        )
        state = circ(time)

        assert allclose(expected_state, state)

    @pytest.mark.jax
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_execution_jax(self, coeffs, ops, seed):
        """Test that the circuit executes as expected using jax"""
        from jax import numpy as jnp

        time = jnp.array(0.5)
        dev = qp.device("reference.qubit", wires=[0, 1])

        @qp.qnode(dev)
        def circ(time):
            hamiltonian = qp.dot(coeffs, ops)
            qp.QDrift(hamiltonian, time, n=2, seed=seed)
            return qp.state()

        expected_decomp = _sample_decomposition(coeffs, ops, time, n=2, seed=seed)

        initial_state = jnp.array([1.0, 0.0, 0.0, 0.0])

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qp.matrix(op, wire_order=[0, 1]) for op in expected_decomp[::-1]],
            )
            @ initial_state
        )
        state = circ(time)

        assert allclose(expected_state, state)

    @pytest.mark.jax
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_execution_jaxjit(self, coeffs, ops, seed):
        """Test that the circuit executes as expected using jax jit"""
        import jax
        from jax import numpy as jnp

        time = jnp.array(0.5)
        dev = qp.device("reference.qubit", wires=[0, 1])

        @jax.jit
        @qp.qnode(dev, interface="jax")
        def circ(time):
            hamiltonian = qp.sum(*(qp.s_prod(coeff, op) for coeff, op in zip(coeffs, ops)))
            qp.QDrift(hamiltonian, time, n=2, seed=seed)
            return qp.state()

        expected_decomp = _sample_decomposition(coeffs, ops, time, n=2, seed=seed)

        initial_state = jnp.array([1.0, 0.0, 0.0, 0.0])

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qp.matrix(op, wire_order=[0, 1]) for op in expected_decomp[::-1]],
            )
            @ initial_state
        )
        state = circ(time)

        assert allclose(expected_state, state)

    @pytest.mark.autograd
    def test_error_gradient_workflow_autograd(self):
        """Test that an error is raised if we require a gradient of QDrift with respect to hamiltonian coefficients."""
        time = qnp.array(1.5)
        coeffs = qnp.array([1.23, -0.45], requires_grad=True)

        terms = [qp.PauliX(0), qp.PauliZ(0)]
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circ(time, coeffs):
            h = qp.dot(coeffs, terms)
            qp.QDrift(h, time, n=3)
            return qp.expval(qp.Hadamard(0))

        msg = "The QDrift template currently doesn't support differentiation through the coefficients of the input Hamiltonian."
        with pytest.raises(QuantumFunctionError, match=msg):
            qp.grad(circ)(time, coeffs)

    @pytest.mark.torch
    def test_error_gradient_workflow_torch(self):
        """Test that an error is raised if we require a gradient of QDrift with respect to hamiltonian coefficients."""
        import torch

        time = torch.tensor(1.5, dtype=torch.complex128, requires_grad=True)
        coeffs = torch.tensor([1.23, -0.45], dtype=torch.complex128, requires_grad=True)

        terms = [qp.PauliX(0), qp.PauliZ(0)]
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circ(time, coeffs):
            h = qp.dot(coeffs, terms)
            qp.QDrift(h, time, n=3)
            return qp.expval(qp.Hadamard(0))

        msg = "The QDrift template currently doesn't support differentiation through the coefficients of the input Hamiltonian."
        with pytest.raises(QuantumFunctionError, match=msg):
            res_circ = circ(time, coeffs)
            res_circ.backward()

    @pytest.mark.tf
    def test_error_gradient_workflow_tf(self):
        """Test that an error is raised if we require a gradient of QDrift with respect to hamiltonian coefficients."""
        import tensorflow as tf

        time = tf.Variable(1.5, dtype=tf.complex128)
        coeffs = tf.Variable([1.23, -0.45], dtype=tf.complex128)

        terms = [qp.PauliX(0), qp.PauliZ(0)]
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circ(time, coeffs):
            h = qp.sum(
                qp.s_prod(coeffs[0], terms[0]),
                qp.s_prod(coeffs[1], terms[1]),
            )
            qp.QDrift(h, time, n=3)
            return qp.expval(qp.Hadamard(0))

        msg = "The QDrift template currently doesn't support differentiation through the coefficients of the input Hamiltonian."
        with pytest.raises(QuantumFunctionError, match=msg):
            with tf.GradientTape() as tape:
                result = circ(time, coeffs)
            tape.gradient(result, coeffs)

    @pytest.mark.jax
    def test_error_gradient_workflow_jax(self):
        """Test that an error is raised if we require a gradient of QDrift with respect to hamiltonian coefficients."""
        import jax
        from jax import numpy as jnp

        time = jnp.array(1.5)
        coeffs = jnp.array([1.23, -0.45])

        terms = [qp.PauliX(0), qp.PauliZ(0)]
        dev = qp.device("reference.qubit", wires=1)

        @qp.qnode(dev)
        def circ(time, coeffs):
            h = qp.dot(coeffs, terms)
            qp.QDrift(h, time, n=3)
            return qp.expval(qp.Hadamard(0))

        msg = "The QDrift template currently doesn't support differentiation through the coefficients of the input Hamiltonian."
        with pytest.raises(QuantumFunctionError, match=msg):
            jax.grad(circ, argnums=[1])(time, coeffs)

    @pytest.mark.autograd
    @pytest.mark.parametrize("n", (1, 5, 10))
    def test_autograd_gradient(self, n, seed):
        """Test that the gradient is computed correctly"""
        time = qnp.array(1.5)
        coeffs = qnp.array([1.23, -0.45], requires_grad=False)
        terms = [qp.PauliX(0), qp.PauliZ(0)]

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circ(time, coeffs):
            h = qp.dot(coeffs, terms)
            qp.QDrift(h, time, n=n, seed=seed)
            return qp.expval(qp.Hadamard(0))

        @qp.qnode(dev)
        def reference_circ(time, coeffs):
            with qp.QueuingManager.stop_recording():
                decomp = _sample_decomposition(coeffs, terms, time, n, seed)

            for op in decomp:
                qp.apply(op)

            return qp.expval(qp.Hadamard(0))

        measured_grad = qp.grad(circ)(time, coeffs)
        reference_grad = qp.grad(reference_circ)(time, coeffs)
        assert allclose(measured_grad, reference_grad)

    @pytest.mark.torch
    @pytest.mark.parametrize("n", (1, 5, 10))
    def test_torch_gradient(self, n, seed):
        """Test that the gradient is computed correctly using torch"""
        import torch

        time = torch.tensor(1.5, dtype=torch.complex128, requires_grad=True)
        coeffs = torch.tensor([1.23, -0.45], dtype=torch.complex128, requires_grad=False)
        ref_time = torch.tensor(1.5, dtype=torch.complex128, requires_grad=True)
        ref_coeffs = torch.tensor([1.23, -0.45], dtype=torch.complex128, requires_grad=False)
        terms = [qp.PauliX(0), qp.PauliZ(0)]

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circ(time, coeffs):
            h = qp.dot(coeffs, terms)
            qp.QDrift(h, time, n=n, seed=seed)
            return qp.expval(qp.Hadamard(0))

        @qp.qnode(dev)
        def reference_circ(time, coeffs):
            with qp.QueuingManager.stop_recording():
                decomp = _sample_decomposition(coeffs, terms, time, n, seed)

            for op in decomp:
                qp.apply(op)

            return qp.expval(qp.Hadamard(0))

        res_circ = circ(time, coeffs)
        res_circ.backward()
        measured_grad = time.grad

        ref_circ = reference_circ(ref_time, ref_coeffs)
        ref_circ.backward()
        reference_grad = ref_time.grad

        assert allclose(measured_grad, reference_grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("n", (1, 5, 10))
    def test_tf_gradient(self, n, seed):
        """Test that the gradient is computed correctly using tensorflow"""
        import tensorflow as tf

        time = tf.Variable(1.5, dtype=tf.complex128)
        coeffs = [1.23, -0.45]
        terms = [qp.PauliX(0), qp.PauliZ(0)]

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circ(time, coeffs):
            h = qp.dot(coeffs, terms)
            qp.QDrift(h, time, n=n, seed=seed)
            return qp.expval(qp.Hadamard(0))

        @qp.qnode(dev)
        def reference_circ(time, coeffs):
            with qp.QueuingManager.stop_recording():
                decomp = _sample_decomposition(coeffs, terms, time, n, seed)

            for op in decomp:
                qp.apply(op)

            return qp.expval(qp.Hadamard(0))

        with tf.GradientTape() as tape:
            result = circ(time, coeffs)
        measured_grad = tape.gradient(result, time)

        with tf.GradientTape() as tape:
            result = reference_circ(time, coeffs)
        reference_grad = tape.gradient(result, time)

        assert allclose(measured_grad, reference_grad)

    @pytest.mark.jax
    @pytest.mark.parametrize("n", (1, 5, 10))
    def test_jax_gradient(self, n, seed):
        """Test that the gradient is computed correctly using jax"""
        import jax
        from jax import numpy as jnp

        time = jnp.array(1.5)
        coeffs = jnp.array([1.23, -0.45])
        terms = [qp.PauliX(0), qp.PauliZ(0)]

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circ(time, coeffs):
            h = qp.dot(coeffs, terms)
            qp.QDrift(h, time, n=n, seed=seed)
            return qp.expval(qp.Hadamard(0))

        @qp.qnode(dev)
        def reference_circ(time, coeffs):
            with qp.QueuingManager.stop_recording():
                decomp = _sample_decomposition(coeffs, terms, time, n, seed)

            for op in decomp:
                qp.apply(op)

            return qp.expval(qp.Hadamard(0))

        measured_grad = jax.grad(circ, argnums=[0])(time, coeffs)
        reference_grad = jax.grad(reference_circ, argnums=[0])(time, coeffs)
        assert allclose(measured_grad, reference_grad)


test_error_data = (  # Computed by hand
    (qp.dot([1.23, -0.45j], [qp.PauliX(0), qp.PauliZ(1)]), 0.5, 5, 0.3949494464),
    (qp.Hamiltonian([1.23, -0.45], [qp.PauliX(0), qp.PauliZ(1)]), 0.5, 5, 0.3949494464),
    (
        qp.dot([1, -0.5, 0.5j], [qp.Identity(wires=[0, 1]), qp.PauliZ(0), qp.Hadamard(1)]),
        3,
        100,
        0.81179773314,
    ),
)


@pytest.mark.parametrize("h, time, n, expected_error", test_error_data)
def test_error_func(h, time, n, expected_error):
    """Test that the error function computes the expected precision correctly"""
    computed_error = qp.QDrift.error(h, time, n)
    assert isclose(computed_error, expected_error)


def test_error_func_type_error():
    """Test that an error is raised if the wrong type is passed for hamiltonian"""
    msg = "The given operator must be a PennyLane ~.Sum"
    with pytest.raises(TypeError, match=msg):
        qp.QDrift.error(qp.PauliX(0), time=1.23, n=10)
