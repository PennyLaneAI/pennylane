# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for default qubit 2."""
import pytest

import numpy as np

import pennylane as qml
from pennylane.devices.experimental import DefaultQubit2, ExecutionConfig
from pennylane.devices.qubit.preprocess import validate_and_expand_adjoint


def test_name():
    """Tests the name of DefaultQubit2."""
    assert DefaultQubit2().name == "default.qubit.2"


def test_no_jvp_functionality():
    """Test that jvp is not supported on DefaultQubit2."""
    dev = DefaultQubit2()

    assert not dev.supports_jvp(ExecutionConfig())

    with pytest.raises(NotImplementedError):
        dev.compute_jvp(qml.tape.QuantumScript(), (10, 10))

    with pytest.raises(NotImplementedError):
        dev.execute_and_compute_jvp(qml.tape.QuantumScript(), (10, 10))


def test_no_vjp_functionality():
    """Test that vjp is not supported on DefaultQubit2."""
    dev = DefaultQubit2()

    assert not dev.supports_vjp(ExecutionConfig())

    with pytest.raises(NotImplementedError):
        dev.compute_vjp(qml.tape.QuantumScript(), (10.0, 10.0))

    with pytest.raises(NotImplementedError):
        dev.execute_and_compute_vjp(qml.tape.QuantumScript(), (10.0, 10.0))


def test_no_device_derivatives():
    """Test that DefaultQubit2 currently doesn't support device derivatives."""
    dev = DefaultQubit2()

    with pytest.raises(NotImplementedError):
        dev.compute_derivatives(qml.tape.QuantumScript())

    with pytest.raises(NotImplementedError):
        dev.execute_and_compute_derivatives(qml.tape.QuantumScript())


class TestTracking:
    """Testing the tracking capabilities of DefaultQubit2."""

    def test_tracker_set_upon_initialization(self):
        """Test that a new tracker is intialized with each device."""
        assert DefaultQubit2.tracker is not DefaultQubit2().tracker

    def test_tracker_not_updated_if_not_active(self):
        """Test that the tracker is not updated if not active."""
        dev = DefaultQubit2()
        assert len(dev.tracker.totals) == 0

        dev.execute(qml.tape.QuantumScript())
        assert len(dev.tracker.totals) == 0
        assert len(dev.tracker.history) == 0

    def test_tracking_batch(self):
        """Test that the experimental default qubit integrates with the tracker."""

        qs = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])

        dev = DefaultQubit2()
        with qml.Tracker(dev) as tracker:
            dev.execute(qs)
            dev.execute([qs, qs])  # and a second time

        assert tracker.history == {"batches": [1, 1], "executions": [1, 2]}
        assert tracker.totals == {"batches": 2, "executions": 3}
        assert tracker.latest == {"batches": 1, "executions": 2}


class TestSupportsDerivatives:
    """Test that DefaultQubit2 states what kind of derivatives it supports."""

    def test_supports_backprop(self):
        """Test that DefaultQubit2 says that it supports backpropagation."""
        dev = DefaultQubit2()
        config = ExecutionConfig(gradient_method="backprop")
        assert dev.supports_derivatives(config) is True

        qs = qml.tape.QuantumScript([], [qml.state()])
        assert dev.supports_derivatives(config, qs)

    def test_supports_adjoint(self):
        """Test that DefaultQubit2 says that it supports adjoint differentiation."""
        dev = DefaultQubit2()
        config = ExecutionConfig(gradient_method="adjoint")
        assert dev.supports_derivatives(config) is True

        qs = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])
        assert dev.supports_derivatives(config, qs) is True

    def test_doesnt_support_adjoint_with_invalid_tape(self):
        """Tests that DefaultQubit2 does not support adjoint differentiation with invalid circuits."""
        dev = DefaultQubit2()
        config = ExecutionConfig(gradient_method="adjoint")
        circuit = qml.tape.QuantumScript([], [qml.probs()])
        assert dev.supports_derivatives(config, circuit=circuit) is False

    @pytest.mark.parametrize("gradient_method", ["parameter-shift", "finite-diff", "device"])
    def test_doesnt_support_other_gradient_methods(self, gradient_method):
        """Test that DefaultQubit2 currently does not support other gradient methods natively."""
        dev = DefaultQubit2()
        config = ExecutionConfig(gradient_method=gradient_method)
        assert not dev.supports_derivatives(config)


class TestBasicCircuit:
    """Tests a basic circuit with one RX gate and two simple expectation values."""

    def test_basic_circuit_numpy(self):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )

        dev = DefaultQubit2()
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

    @pytest.mark.autograd
    def test_autograd_results_and_backprop(self):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        dev = DefaultQubit2()

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return qml.numpy.array(dev.execute(qs))

        result = f(phi)
        expected = np.array([-np.sin(phi), np.cos(phi)])
        assert qml.math.allclose(result, expected)

        g = qml.jacobian(f)(phi)
        expected = np.array([-np.cos(phi), -np.sin(phi)])
        assert qml.math.allclose(g, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_results_and_backprop(self, use_jit):
        """Tests execution and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        dev = DefaultQubit2()

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return dev.execute(qs)

        if use_jit:
            f = jax.jit(f)

        result = f(phi)
        assert qml.math.allclose(result[0], -np.sin(phi))
        assert qml.math.allclose(result[1], np.cos(phi))

        g = jax.jacobian(f)(phi)
        assert qml.math.allclose(g[0], -np.cos(phi))
        assert qml.math.allclose(g[1], -np.sin(phi))

    @pytest.mark.torch
    def test_torch_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with torch."""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        dev = DefaultQubit2()

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return dev.execute(qs)

        result = f(phi)
        assert qml.math.allclose(result[0], -torch.sin(phi))
        assert qml.math.allclose(result[1], torch.cos(phi))

        g = torch.autograd.functional.jacobian(f, phi + 0j)
        assert qml.math.allclose(g[0], -torch.cos(phi))
        assert qml.math.allclose(g[1], -torch.sin(phi))

    # pylint: disable=invalid-unary-operand-type
    @pytest.mark.tf
    def test_tf_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)

        dev = DefaultQubit2()

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = qml.tape.QuantumScript(
                [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            result = dev.execute(qs)

        assert qml.math.allclose(result[0], -tf.sin(phi))
        assert qml.math.allclose(result[1], tf.cos(phi))

        grad0 = grad_tape.jacobian(result[0], [phi])
        grad1 = grad_tape.jacobian(result[1], [phi])

        assert qml.math.allclose(grad0[0], -tf.cos(phi))
        assert qml.math.allclose(grad1[0], -tf.sin(phi))


class TestExecutingBatches:
    @staticmethod
    def f(phi):
        """A function that executes a batch of scripts on DefaultQubit2 without preprocessing."""
        ops = [
            qml.PauliX("a"),
            qml.PauliX("b"),
            qml.ctrl(qml.RX(phi, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
        ]

        qs1 = qml.tape.QuantumScript(
            ops,
            [
                qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
            ],
        )

        ops = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        qs2 = qml.tape.QuantumScript(ops, [qml.probs(wires=(0, 1))])
        return DefaultQubit2().execute((qs1, qs2))

    @staticmethod
    def expected(phi):
        out1 = (-qml.math.sin(phi) - 1, 3 * qml.math.cos(phi))

        x1 = qml.math.cos(phi / 2) ** 2 / 2
        x2 = qml.math.sin(phi / 2) ** 2 / 2
        out2 = x1 * np.array([1, 0, 1, 0]) + x2 * np.array([0, 1, 0, 1])
        return (out1, out2)

    @staticmethod
    def nested_compare(x1, x2):
        assert len(x1) == len(x2)
        assert len(x1[0]) == len(x2[0])
        assert qml.math.allclose(x1[0][0], x2[0][0])
        assert qml.math.allclose(x1[0][1], x2[0][1])
        assert qml.math.allclose(x1[1], x2[1])

    def test_numpy(self):
        """Test that results are expected when the parameter does not have a parameter."""
        phi = 0.892
        results = self.f(phi)
        expected = self.expected(phi)

        self.nested_compare(results, expected)

    @pytest.mark.autograd
    def test_autograd(self):
        """Test batches can be executed and have backprop derivatives in autograd."""

        phi = qml.numpy.array(-0.629)
        results = self.f(phi)
        expected = self.expected(phi)

        self.nested_compare(results, expected)

        g0 = qml.jacobian(lambda x: qml.numpy.array(self.f(x)[0]))(phi)
        g0_expected = qml.jacobian(lambda x: qml.numpy.array(self.expected(x)[0]))(phi)
        assert qml.math.allclose(g0, g0_expected)

        g1 = qml.jacobian(lambda x: qml.numpy.array(self.expected(x)[1]))(phi)
        g1_expected = qml.jacobian(lambda x: qml.numpy.array(self.expected(x)[1]))(phi)
        assert qml.math.allclose(g1, g1_expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax(self, use_jit):
        """Test batches can be executed and have backprop derivatives in jax."""
        import jax

        phi = jax.numpy.array(0.123)

        f = jax.jit(self.f) if use_jit else self.f
        results = f(phi)
        expected = self.expected(phi)

        self.nested_compare(results, expected)

        g = jax.jacobian(f)(phi)
        g_expected = jax.jacobian(self.expected)(phi)

        self.nested_compare(g, g_expected)

    @pytest.mark.torch
    def test_torch(self):
        """Test batches can be executed and have backprop derivatives in torch."""
        import torch

        x = torch.tensor(9.6243)

        results = self.f(x)
        expected = self.expected(x)

        self.nested_compare(results, expected)

        g1 = torch.autograd.functional.jacobian(lambda y: self.f(y)[0], x)
        assert qml.math.allclose(g1[0], -qml.math.cos(x))
        assert qml.math.allclose(g1[1], -3 * qml.math.sin(x))

        g1 = torch.autograd.functional.jacobian(lambda y: self.f(y)[1], x)
        temp = -0.5 * qml.math.cos(x / 2) * qml.math.sin(x / 2)
        g3 = torch.tensor([temp, -temp, temp, -temp])
        assert qml.math.allclose(g1, g3)

    @pytest.mark.tf
    def test_tf(self):
        """Test batches can be executed and have backprop derivatives in tf."""

        import tensorflow as tf

        x = tf.Variable(5.2281)
        with tf.GradientTape(persistent=True) as tape:
            results = self.f(x)

        expected = self.expected(x)
        self.nested_compare(results, expected)

        g00 = tape.gradient(results[0][0], x)
        assert qml.math.allclose(g00, -qml.math.cos(x))
        g01 = tape.gradient(results[0][1], x)
        assert qml.math.allclose(g01, -3 * qml.math.sin(x))

        g1 = tape.jacobian(results[1], x)
        temp = -0.5 * qml.math.cos(x / 2) * qml.math.sin(x / 2)
        g3 = tf.Variable([temp, -temp, temp, -temp])
        assert qml.math.allclose(g1, g3)


class TestSumOfTermsDifferentiability:
    """Basically a copy of the `qubit.simulate` test but using the device instead."""

    @staticmethod
    def f(scale, n_wires=10, offset=0.1, convert_to_hamiltonian=False):
        ops = [qml.RX(offset + scale * i, wires=i) for i in range(n_wires)]

        t1 = 2.5 * qml.prod(*(qml.PauliZ(i) for i in range(n_wires)))
        t2 = 6.2 * qml.prod(*(qml.PauliY(i) for i in range(n_wires)))
        H = t1 + t2
        if convert_to_hamiltonian:
            H = H._pauli_rep.hamiltonian()  # pylint: disable=protected-access
        qs = qml.tape.QuantumScript(ops, [qml.expval(H)])
        return DefaultQubit2().execute(qs)

    @staticmethod
    def expected(scale, n_wires=10, offset=0.1, like="numpy"):
        phase = offset + scale * qml.math.asarray(range(n_wires), like=like)
        cosines = qml.math.cos(phase)
        sines = qml.math.sin(phase)
        return 2.5 * qml.math.prod(cosines) + 6.2 * qml.math.prod(sines)

    @pytest.mark.autograd
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_autograd_backprop(self, convert_to_hamiltonian):
        """Test that backpropagation derivatives work in autograd with hamiltonians and large sums."""
        x = qml.numpy.array(0.52)
        out = self.f(x, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_out = self.expected(x)
        assert qml.math.allclose(out, expected_out)

        g = qml.grad(self.f)(x, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_g = qml.grad(self.expected)(x)
        assert qml.math.allclose(g, expected_g)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_jax_backprop(self, convert_to_hamiltonian, use_jit):
        """Test that backpropagation derivatives work with jax with hamiltonians and large sums."""
        import jax
        from jax.config import config

        config.update("jax_enable_x64", True)  # otherwise output is too noisy

        x = jax.numpy.array(0.52, dtype=jax.numpy.float64)
        f = jax.jit(self.f, static_argnums=(1, 2, 3)) if use_jit else self.f

        out = f(x, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_out = self.expected(x)
        assert qml.math.allclose(out, expected_out, atol=1e-6)

        g = jax.grad(f)(x, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_g = jax.grad(self.expected)(x)
        assert qml.math.allclose(g, expected_g)

    @pytest.mark.torch
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_torch_backprop(self, convert_to_hamiltonian):
        """Test that backpropagation derivatives work with torch with hamiltonians and large sums."""
        import torch

        x = torch.tensor(-0.289, requires_grad=True)
        x2 = torch.tensor(-0.289, requires_grad=True)
        out = self.f(x, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_out = self.expected(x2, like="torch")
        assert qml.math.allclose(out, expected_out)

        out.backward()
        expected_out.backward()
        assert qml.math.allclose(x.grad, x2.grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_tf_backprop(self, convert_to_hamiltonian):
        """Test that backpropagation derivatives work with tensorflow with hamiltonians and large sums."""
        import tensorflow as tf

        x = tf.Variable(0.5)

        with tf.GradientTape() as tape1:
            out = self.f(x, convert_to_hamiltonian=convert_to_hamiltonian)

        with tf.GradientTape() as tape2:
            expected_out = self.expected(x)

        assert qml.math.allclose(out, expected_out)
        g1 = tape1.gradient(out, x)
        g2 = tape2.gradient(expected_out, x)
        assert qml.math.allclose(g1, g2)


class TestAdjointDifferentiation:
    """Tests adjoint differentiation integration with DefaultQubit2."""

    ec = ExecutionConfig(gradient_method="adjoint")

    def test_single_circuit(self):
        """Tests a basic example with a single circuit."""
        dev = DefaultQubit2()
        x = np.array(np.pi / 7)
        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        qs = validate_and_expand_adjoint(qs)
        expected_grad = -qml.math.sin(x)
        actual_grad = dev.compute_derivatives(qs, self.ec)
        assert isinstance(actual_grad, np.ndarray)
        assert actual_grad.shape == ()
        assert np.isclose(actual_grad, expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_derivatives(qs, self.ec)
        assert np.isclose(actual_val, expected_val)
        assert np.isclose(actual_grad, expected_grad)

    def test_list_with_single_circuit(self):
        """Tests a basic example with a batch containing a single circuit."""
        dev = DefaultQubit2()
        x = np.array(np.pi / 7)
        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        qs = validate_and_expand_adjoint(qs)
        expected_grad = -qml.math.sin(x)
        actual_grad = dev.compute_derivatives([qs], self.ec)
        assert isinstance(actual_grad, tuple)
        assert isinstance(actual_grad[0], np.ndarray)
        assert np.isclose(actual_grad[0], expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_derivatives([qs], self.ec)
        assert np.isclose(expected_val, actual_val[0])
        assert np.isclose(expected_grad, actual_grad[0])

    def test_many_tapes_many_results(self):
        """Tests a basic example with a batch of circuits of varying return shapes."""
        dev = DefaultQubit2()
        x = np.array(np.pi / 7)
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        expected_grad = (-qml.math.sin(x), (qml.math.cos(x), -qml.math.sin(x)))
        actual_grad = dev.compute_derivatives([single_meas, multi_meas], self.ec)
        assert np.isclose(actual_grad[0], expected_grad[0])
        assert isinstance(actual_grad[1], tuple)
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

    def test_integration(self):
        """Tests the expected workflow done by a calling method."""
        dev = DefaultQubit2()
        x = np.array(np.pi / 7)
        expected_grad = (-qml.math.sin(x), (qml.math.cos(x), -qml.math.sin(x)))
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )

        circuits, fn = dev.preprocess([single_meas, multi_meas], self.ec)
        actual_grad = fn(dev.compute_derivatives(circuits, self.ec))

        assert np.isclose(actual_grad[0], expected_grad[0])
        assert isinstance(actual_grad[1], tuple)
        assert qml.math.allclose(actual_grad[1], expected_grad[1])


class TestPreprocessingIntegration:
    def test_preprocess_single_circuit(self):
        """Test integration between preprocessing and execution with numpy parameters."""

        # pylint: disable=too-few-public-methods
        class MyTemplate(qml.operation.Operation):
            num_wires = 2

            def decomposition(self):
                return [
                    qml.RX(self.data[0], self.wires[0]),
                    qml.RY(self.data[1], self.wires[1]),
                    qml.CNOT(self.wires),
                ]

        x = 0.928
        y = -0.792
        qscript = qml.tape.QuantumScript(
            [MyTemplate(x, y, ("a", "b"))],
            [qml.expval(qml.PauliY("a")), qml.expval(qml.PauliZ("a")), qml.expval(qml.PauliX("b"))],
        )

        dev = DefaultQubit2()

        batch, post_procesing_fn = dev.preprocess(qscript)

        assert len(batch) == 1
        execute_circuit = batch[0]
        assert qml.equal(execute_circuit[0], qml.RX(x, "a"))
        assert qml.equal(execute_circuit[1], qml.RY(y, "b"))
        assert qml.equal(execute_circuit[2], qml.CNOT(("a", "b")))
        assert qml.equal(execute_circuit[3], qml.expval(qml.PauliY("a")))
        assert qml.equal(execute_circuit[4], qml.expval(qml.PauliZ("a")))
        assert qml.equal(execute_circuit[5], qml.expval(qml.PauliX("b")))

        results = dev.execute(batch)
        assert len(results) == 1
        assert len(results[0]) == 3

        processed_results = post_procesing_fn(results)
        assert len(processed_results) == 3
        assert qml.math.allclose(processed_results[0], -np.sin(x) * np.sin(y))
        assert qml.math.allclose(processed_results[1], np.cos(x))
        assert qml.math.allclose(processed_results[2], np.sin(y))

    def test_preprocess_batch_circuit(self):
        """Test preprocess integrates with default qubit when we start with a batch of circuits."""

        # pylint: disable=too-few-public-methods
        class CustomIsingXX(qml.operation.Operation):
            num_wires = 2

            def decomposition(self):
                return [qml.IsingXX(self.data[0], self.wires)]

        x = 0.692

        measurements1 = [qml.density_matrix("a"), qml.vn_entropy("a")]
        qs1 = qml.tape.QuantumScript([CustomIsingXX(x, ("a", "b"))], measurements1)

        y = -0.923

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliX(wires=1)
            m_0 = qml.measure(1)
            qml.cond(m_0, qml.RY)(y, wires=0)
            qml.expval(qml.PauliZ(0))

        qs2 = qml.tape.QuantumScript.from_queue(q)

        initial_batch = [qs1, qs2]

        dev = DefaultQubit2()
        batch, post_processing_fn = dev.preprocess(initial_batch)

        results = dev.execute(batch)
        processed_results = post_processing_fn(results)

        assert len(processed_results) == 2
        assert len(processed_results[0]) == 2

        expected_density_mat = np.array([[np.cos(x / 2) ** 2, 0], [0, np.sin(x / 2) ** 2]])
        assert qml.math.allclose(processed_results[0][0], expected_density_mat)

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(x / 2) ** 2 * np.sin(x / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(x / 2) ** 2 * np.sin(x / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = -np.sum(eigs * np.log(eigs))
        assert qml.math.allclose(processed_results[0][1], expected_entropy)

        expected_expval = np.cos(y)
        assert qml.math.allclose(expected_expval, processed_results[1])


def test_broadcasted_parameter():
    """Test that DefaultQubit2 handles broadcasted parameters as expected."""
    dev = DefaultQubit2()
    x = np.array([0.536, 0.894])
    qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
    batch, post_processing_fn = dev.preprocess(qs)
    assert len(batch) == 2
    results = dev.execute(batch)
    processed_results = post_processing_fn(results)
    assert qml.math.allclose(processed_results, np.cos(x))
