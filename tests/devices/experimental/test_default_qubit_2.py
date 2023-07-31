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
# pylint: disable=import-outside-toplevel, no-member

import pytest

import numpy as np

import pennylane as qml
from pennylane.resource import Resources
from pennylane.devices.experimental import DefaultQubit2, ExecutionConfig
from pennylane.devices.qubit.preprocess import validate_and_expand_adjoint


def test_name():
    """Tests the name of DefaultQubit2."""
    assert DefaultQubit2().name == "default.qubit.2"


def test_shots():
    """Test the shots property of DefaultQubit2."""
    assert DefaultQubit2().shots == qml.measurements.Shots(None)
    assert DefaultQubit2(shots=100).shots == qml.measurements.Shots(100)

    with pytest.raises(AttributeError):
        DefaultQubit2().shots = 10


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


def test_debugger_attribute():
    """Test that DefaultQubit2 has a debugger attribute and that it is `None`"""
    # pylint: disable=protected-access
    dev = DefaultQubit2()

    assert hasattr(dev, "_debugger")
    assert dev._debugger is None


def test_snapshot_multiprocessing_execute():
    """DefaultQubit2 cannot execute tapes with Snapshot if `max_workers` is not `None`"""
    dev = DefaultQubit2(max_workers=2)

    tape = qml.tape.QuantumScript(
        [
            qml.Snapshot(),
            qml.Hadamard(wires=0),
            qml.Snapshot("very_important_state"),
            qml.CNOT(wires=[0, 1]),
            qml.Snapshot(),
        ],
        [qml.expval(qml.PauliX(0))],
    )
    with pytest.raises(RuntimeError, match="ProcessPoolExecutor cannot execute a QuantumScript"):
        dev.execute(tape)


def test_snapshot_multiprocessing_qnode():
    """DefaultQubit2 cannot execute tapes with Snapshot if `max_workers` is not `None`"""
    dev = DefaultQubit2(max_workers=2)

    @qml.qnode(dev)
    def circuit():
        qml.Snapshot("tag")
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Snapshot()
        return qml.expval(qml.PauliX(0) + qml.PauliY(0))

    with pytest.raises(
        qml.DeviceError, match="Debugging with ``Snapshots`` is not available with multiprocessing."
    ):
        qml.snapshots(circuit)()


class TestTracking:
    """Testing the tracking capabilities of DefaultQubit2."""

    def test_tracker_set_upon_initialization(self):
        """Test that a new tracker is intialized with each device."""
        assert DefaultQubit2().tracker is not DefaultQubit2().tracker

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
        config = ExecutionConfig(gradient_method="adjoint")
        with qml.Tracker(dev) as tracker:
            dev.execute(qs)
            dev.compute_derivatives(qs, config)
            dev.execute([qs, qs])  # and a second time

        assert tracker.history == {
            "batches": [1, 1],
            "executions": [1, 2],
            "resources": [Resources(num_wires=1), Resources(num_wires=1), Resources(num_wires=1)],
            "derivative_batches": [1],
            "derivatives": [1],
        }
        assert tracker.totals == {
            "batches": 2,
            "executions": 3,
            "derivative_batches": 1,
            "derivatives": 1,
        }
        assert tracker.latest == {"batches": 1, "executions": 2}

    def test_tracking_execute_and_derivatives(self):
        """Test that the execute_and_compute_* calls are being tracked for the
        experimental default qubit device"""

        qs = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])
        dev = DefaultQubit2()
        config = ExecutionConfig(gradient_method="adjoint")

        with qml.Tracker(dev) as tracker:
            dev.compute_derivatives(qs, config)
            dev.execute_and_compute_derivatives([qs] * 2, config)
            dev.compute_jvp([qs] * 3, [(0,)] * 3, config)
            dev.execute_and_compute_jvp([qs] * 4, [(0,)] * 4, config)
            dev.compute_vjp([qs] * 5, [(0,)] * 5, config)
            dev.execute_and_compute_vjp([qs] * 6, [(0,)] * 6, config)

        assert tracker.history == {
            "executions": [2, 4, 6],
            "derivatives": [1, 2],
            "derivative_batches": [1],
            "execute_and_derivative_batches": [1],
            "jvps": [3, 4],
            "jvp_batches": [1],
            "execute_and_jvp_batches": [1],
            "vjps": [5, 6],
            "vjp_batches": [1],
            "execute_and_vjp_batches": [1],
            "resources": [Resources(num_wires=1)] * 12,
        }

    def test_tracking_resources(self):
        """Test that resources are tracked for the experimental default qubit device."""
        qs = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.CNOT(wires=[0, 2]),
                qml.RZ(1.23, 1),
                qml.CNOT(wires=[1, 2]),
                qml.Hadamard(0),
            ],
            [qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliY(2))],
        )

        expected_resources = Resources(
            num_wires=3,
            num_gates=6,
            gate_types={"Hadamard": 3, "CNOT": 2, "RZ": 1},
            gate_sizes={1: 4, 2: 2},
            depth=3,
        )

        dev = DefaultQubit2()
        with qml.Tracker(dev) as tracker:
            dev.execute(qs)

        assert len(tracker.history["resources"]) == 1
        assert tracker.history["resources"][0] == expected_resources


# pylint: disable=too-few-public-methods
class TestPreprocessing:
    """Unit tests for the preprocessing method."""

    def test_chooses_best_gradient_method(self):
        """Test that preprocessing chooses backprop as the best gradient method."""
        dev = DefaultQubit2()

        config = ExecutionConfig(
            gradient_method="best", use_device_gradient=None, grad_on_execution=None
        )
        circuit = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])
        _, _, new_config = dev.preprocess(circuit, config)

        assert new_config.gradient_method == "backprop"
        assert new_config.use_device_gradient
        assert not new_config.grad_on_execution

    def test_config_choices_for_adjoint(self):
        """Test that preprocessing request grad on execution and says to use the device gradient if adjoint is requested."""
        dev = DefaultQubit2()

        config = ExecutionConfig(
            gradient_method="adjoint", use_device_gradient=None, grad_on_execution=None
        )
        circuit = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])
        _, _, new_config = dev.preprocess(circuit, config)

        assert new_config.use_device_gradient
        assert new_config.grad_on_execution

    @pytest.mark.parametrize("max_workers", [None, 1, 2, 3])
    def test_config_choices_for_threading(self, max_workers):
        """Test that preprocessing request grad on execution and says to use the device gradient if adjoint is requested."""
        dev = DefaultQubit2()

        config = ExecutionConfig(device_options={"max_workers": max_workers})
        circuit = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])
        _, _, new_config = dev.preprocess(circuit, config)

        assert new_config.device_options["max_workers"] == max_workers


class TestSupportsDerivatives:
    """Test that DefaultQubit2 states what kind of derivatives it supports."""

    def test_supports_backprop(self):
        """Test that DefaultQubit2 says that it supports backpropagation."""
        dev = DefaultQubit2()
        assert dev.supports_derivatives() is True
        assert dev.supports_jvp() is True
        assert dev.supports_vjp() is True

        config = ExecutionConfig(gradient_method="backprop")
        assert dev.supports_derivatives(config) is True
        assert dev.supports_jvp(config) is True
        assert dev.supports_vjp(config) is True

        qs = qml.tape.QuantumScript([], [qml.state()])
        assert dev.supports_derivatives(config, qs) is True
        assert dev.supports_jvp(config, qs) is True
        assert dev.supports_vjp(config, qs) is True

        config = ExecutionConfig(gradient_method="backprop", device_options={"max_workers": 1})
        assert dev.supports_derivatives(config) is False
        assert dev.supports_jvp(config) is False
        assert dev.supports_vjp(config) is False

    def test_supports_adjoint(self):
        """Test that DefaultQubit2 says that it supports adjoint differentiation."""
        dev = DefaultQubit2()
        config = ExecutionConfig(gradient_method="adjoint", use_device_gradient=True)
        assert dev.supports_derivatives(config) is True
        assert dev.supports_jvp(config) is True
        assert dev.supports_vjp(config) is True

        qs = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])
        assert dev.supports_derivatives(config, qs) is True
        assert dev.supports_jvp(config, qs) is True
        assert dev.supports_vjp(config, qs) is True

        config = ExecutionConfig(gradient_method="adjoint", use_device_gradient=False)
        assert dev.supports_derivatives(config) is False
        assert dev.supports_jvp(config) is False
        assert dev.supports_vjp(config) is False

        assert dev.supports_derivatives(config, qs) is False
        assert dev.supports_jvp(config, qs) is False
        assert dev.supports_vjp(config, qs) is False

    def test_doesnt_support_adjoint_with_invalid_tape(self):
        """Tests that DefaultQubit2 does not support adjoint differentiation with invalid circuits."""
        dev = DefaultQubit2()
        config = ExecutionConfig(gradient_method="adjoint")
        circuit = qml.tape.QuantumScript([], [qml.probs()])
        assert dev.supports_derivatives(config, circuit=circuit) is False
        assert dev.supports_jvp(config, circuit=circuit) is False
        assert dev.supports_vjp(config, circuit=circuit) is False

    @pytest.mark.parametrize("gradient_method", ["parameter-shift", "finite-diff", "device"])
    def test_doesnt_support_other_gradient_methods(self, gradient_method):
        """Test that DefaultQubit2 currently does not support other gradient methods natively."""
        dev = DefaultQubit2()
        config = ExecutionConfig(gradient_method=gradient_method)
        assert dev.supports_derivatives(config) is False
        assert dev.supports_jvp(config) is False
        assert dev.supports_vjp(config) is False


class TestBasicCircuit:
    """Tests a basic circuit with one RX gate and two simple expectation values."""

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_basic_circuit_numpy(self, max_workers):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )

        dev = DefaultQubit2(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_basic_circuit_numpy_with_config(self, max_workers):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )

        dev = DefaultQubit2(max_workers=max_workers)
        config = ExecutionConfig(
            device_options={"max_workers": dev._max_workers}  # pylint: disable=protected-access
        )
        result = dev.execute(qs, execution_config=config)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

    @pytest.mark.autograd
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_autograd_results_and_backprop(self, max_workers):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        dev = DefaultQubit2(max_workers=max_workers)

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return qml.numpy.array(dev.execute(qs))

        result = f(phi)
        expected = np.array([-np.sin(phi), np.cos(phi)])
        assert qml.math.allclose(result, expected)

        if max_workers is not None:
            return

        g = qml.jacobian(f)(phi)
        expected = np.array([-np.cos(phi), -np.sin(phi)])
        assert qml.math.allclose(g, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_jax_results_and_backprop(self, use_jit, max_workers):
        """Tests execution and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        dev = DefaultQubit2(max_workers=max_workers)

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return dev.execute(qs)

        if use_jit:
            if max_workers is not None:
                return
            f = jax.jit(f)

        result = f(phi)
        assert qml.math.allclose(result[0], -np.sin(phi))
        assert qml.math.allclose(result[1], np.cos(phi))

        if max_workers is not None:
            return

        g = jax.jacobian(f)(phi)
        assert qml.math.allclose(g[0], -np.cos(phi))
        assert qml.math.allclose(g[1], -np.sin(phi))

    @pytest.mark.torch
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_torch_results_and_backprop(self, max_workers):
        """Tests execution and gradients of a simple circuit with torch."""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        dev = DefaultQubit2(max_workers=max_workers)

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return dev.execute(qs)

        result = f(phi)
        assert qml.math.allclose(result[0], -torch.sin(phi))
        assert qml.math.allclose(result[1], torch.cos(phi))

        if max_workers is not None:
            return

        g = torch.autograd.functional.jacobian(f, phi + 0j)
        assert qml.math.allclose(g[0], -torch.cos(phi))
        assert qml.math.allclose(g[1], -torch.sin(phi))

    # pylint: disable=invalid-unary-operand-type
    @pytest.mark.tf
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_tf_results_and_backprop(self, max_workers):
        """Tests execution and gradients of a simple circuit with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)

        dev = DefaultQubit2(max_workers=max_workers)

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = qml.tape.QuantumScript(
                [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            result = dev.execute(qs)

        assert qml.math.allclose(result[0], -tf.sin(phi))
        assert qml.math.allclose(result[1], tf.cos(phi))

        if max_workers is not None:
            return

        grad0 = grad_tape.jacobian(result[0], [phi])
        grad1 = grad_tape.jacobian(result[1], [phi])

        assert qml.math.allclose(grad0[0], -tf.cos(phi))
        assert qml.math.allclose(grad1[0], -tf.sin(phi))


class TestSampleMeasurements:
    """A copy of the `qubit.simulate` tests, but using the device"""

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_single_expval(self, max_workers):
        """Test a simple circuit with a single expval measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.expval(qml.PauliZ(0))], shots=10000)

        dev = DefaultQubit2(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, (float, np.ndarray))
        assert result.shape == ()
        assert np.allclose(result, np.cos(x), atol=0.1)

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_single_probs(self, max_workers):
        """Test a simple circuit with a single prob measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.probs(wires=0)], shots=10000)

        dev = DefaultQubit2(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, (float, np.ndarray))
        assert result.shape == (2,)
        assert np.allclose(result, [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2], atol=0.1)

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_single_sample(self, max_workers):
        """Test a simple circuit with a single sample measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=10000)

        dev = DefaultQubit2(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, (float, np.ndarray))
        assert result.shape == (10000, 2)
        assert np.allclose(
            np.sum(result, axis=0).astype(np.float32) / 10000, [np.sin(x / 2) ** 2, 0], atol=0.1
        )

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_multi_measurements(self, max_workers):
        """Test a simple circuit containing multiple measurements"""
        x, y = np.array(0.732), np.array(0.488)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
            [qml.expval(qml.Hadamard(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
            shots=10000,
        )

        dev = DefaultQubit2(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == 3

        assert all(isinstance(res, (float, np.ndarray)) for res in result)

        assert result[0].shape == ()
        assert np.allclose(result[0], np.cos(x) / np.sqrt(2), atol=0.1)

        assert result[1].shape == (4,)
        assert np.allclose(
            result[1],
            [
                np.cos(x / 2) ** 2 * np.cos(y / 2) ** 2,
                np.cos(x / 2) ** 2 * np.sin(y / 2) ** 2,
                np.sin(x / 2) ** 2 * np.sin(y / 2) ** 2,
                np.sin(x / 2) ** 2 * np.cos(y / 2) ** 2,
            ],
            atol=0.1,
        )

        assert result[2].shape == (10000, 2)

    shots_data = [
        [10000, 10000],
        [(10000, 2)],
        [10000, 20000],
        [(10000, 2), 20000],
        [(10000, 3), 20000, (30000, 2)],
    ]

    @pytest.mark.parametrize("shots", shots_data)
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_expval_shot_vector(self, max_workers, shots):
        """Test a simple circuit with a single expval measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.expval(qml.PauliZ(0))], shots=shots)

        dev = DefaultQubit2(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, (float, np.ndarray)) for res in result)
        assert all(res.shape == () for res in result)
        assert all(np.allclose(res, np.cos(x), atol=0.1) for res in result)

    @pytest.mark.parametrize("shots", shots_data)
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_probs_shot_vector(self, max_workers, shots):
        """Test a simple circuit with a single prob measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.probs(wires=0)], shots=shots)

        dev = DefaultQubit2(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, (float, np.ndarray)) for res in result)
        assert all(res.shape == (2,) for res in result)
        assert all(
            np.allclose(res, [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2], atol=0.1) for res in result
        )

    @pytest.mark.parametrize("shots", shots_data)
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_sample_shot_vector(self, max_workers, shots):
        """Test a simple circuit with a single sample measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=shots)

        dev = DefaultQubit2(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, (float, np.ndarray)) for res in result)
        assert all(res.shape == (s, 2) for res, s in zip(result, shots))
        assert all(
            np.allclose(
                np.sum(res, axis=0).astype(np.float32) / s, [np.sin(x / 2) ** 2, 0], atol=0.1
            )
            for res, s in zip(result, shots)
        )

    @pytest.mark.parametrize("shots", shots_data)
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_multi_measurement_shot_vector(self, max_workers, shots):
        """Test a simple circuit containing multiple measurements for shot vectors"""
        x, y = np.array(0.732), np.array(0.488)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
            [qml.expval(qml.Hadamard(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
            shots=shots,
        )

        dev = DefaultQubit2(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        for shot_res, s in zip(result, shots):
            assert isinstance(shot_res, tuple)
            assert len(shot_res) == 3

            assert all(isinstance(meas_res, (float, np.ndarray)) for meas_res in shot_res)

            assert shot_res[0].shape == ()
            assert np.allclose(shot_res[0], np.cos(x) / np.sqrt(2), atol=0.1)

            assert shot_res[1].shape == (4,)
            assert np.allclose(
                shot_res[1],
                [
                    np.cos(x / 2) ** 2 * np.cos(y / 2) ** 2,
                    np.cos(x / 2) ** 2 * np.sin(y / 2) ** 2,
                    np.sin(x / 2) ** 2 * np.sin(y / 2) ** 2,
                    np.sin(x / 2) ** 2 * np.cos(y / 2) ** 2,
                ],
                atol=0.1,
            )

            assert shot_res[2].shape == (s, 2)

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_custom_wire_labels(self, max_workers):
        """Test that custom wire labels works as expected"""
        x, y = np.array(0.732), np.array(0.488)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires="b"), qml.CNOT(wires=["b", "a"]), qml.RY(y, wires="a")],
            [
                qml.expval(qml.PauliZ("b")),
                qml.probs(wires=["a", "b"]),
                qml.sample(wires=["b", "a"]),
            ],
            shots=10000,
        )

        dev = DefaultQubit2(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == 3

        assert all(isinstance(res, (float, np.ndarray)) for res in result)

        assert result[0].shape == ()
        assert np.allclose(result[0], np.cos(x), atol=0.1)

        assert result[1].shape == (4,)
        assert np.allclose(
            result[1],
            [
                np.cos(x / 2) ** 2 * np.cos(y / 2) ** 2,
                np.sin(x / 2) ** 2 * np.sin(y / 2) ** 2,
                np.cos(x / 2) ** 2 * np.sin(y / 2) ** 2,
                np.sin(x / 2) ** 2 * np.cos(y / 2) ** 2,
            ],
            atol=0.1,
        )

        assert result[2].shape == (10000, 2)

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_batch_tapes(self, max_workers):
        """Test that a batch of tapes with sampling works as expected"""
        x = np.array(0.732)
        qs1 = qml.tape.QuantumScript([qml.RX(x, wires=0)], [qml.sample(wires=(0, 1))], shots=100)
        qs2 = qml.tape.QuantumScript([qml.RX(x, wires=0)], [qml.sample(wires=1)], shots=50)

        dev = DefaultQubit2(max_workers=max_workers)
        results = dev.execute((qs1, qs2))

        assert isinstance(results, tuple)
        assert len(results) == 2
        assert all(isinstance(res, (float, np.ndarray)) for res in results)
        assert results[0].shape == (100, 2)
        assert results[1].shape == (50,)

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_counts_wires(self, max_workers):
        """Test that a Counts measurement with wires works as expected"""
        x = np.array(np.pi / 2)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.counts(wires=[0, 1])], shots=10000)

        dev = DefaultQubit2(seed=123, max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"00", "10"}

        # check that the count values match the expected
        values = list(result.values())
        assert np.allclose(values[0] / (values[0] + values[1]), 0.5, atol=0.01)

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    @pytest.mark.parametrize("all_outcomes", [False, True])
    def test_counts_obs(self, all_outcomes, max_workers):
        """Test that a Counts measurement with an observable works as expected"""
        x = np.array(np.pi / 2)
        qs = qml.tape.QuantumScript(
            [qml.RY(x, wires=0)],
            [qml.counts(qml.PauliZ(0), all_outcomes=all_outcomes)],
            shots=10000,
        )

        dev = DefaultQubit2(seed=123, max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, dict)
        assert set(result.keys()) == {1, -1}

        # check that the count values match the expected
        values = list(result.values())
        assert np.allclose(values[0] / (values[0] + values[1]), 0.5, atol=0.01)


class TestExecutingBatches:
    """Tests involving executing multiple circuits at the same time."""

    @staticmethod
    def f(dev, phi):
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
        return dev.execute((qs1, qs2))

    @staticmethod
    def f_hashable(phi):
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
        """expected output of f."""
        out1 = (-qml.math.sin(phi) - 1, 3 * qml.math.cos(phi))

        x1 = qml.math.cos(phi / 2) ** 2 / 2
        x2 = qml.math.sin(phi / 2) ** 2 / 2
        out2 = x1 * np.array([1, 0, 1, 0]) + x2 * np.array([0, 1, 0, 1])
        return (out1, out2)

    @staticmethod
    def nested_compare(x1, x2):
        """Assert two ragged lists are equal."""
        assert len(x1) == len(x2)
        assert len(x1[0]) == len(x2[0])
        assert qml.math.allclose(x1[0][0], x2[0][0])
        assert qml.math.allclose(x1[0][1], x2[0][1])
        assert qml.math.allclose(x1[1], x2[1])

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_numpy(self, max_workers):
        """Test that results are expected when the parameter does not have a parameter."""
        dev = DefaultQubit2(max_workers=max_workers)

        phi = 0.892
        results = self.f(dev, phi)
        expected = self.expected(phi)

        self.nested_compare(results, expected)

    @pytest.mark.autograd
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_autograd(self, max_workers):
        """Test batches can be executed and have backprop derivatives in autograd."""
        dev = DefaultQubit2(max_workers=max_workers)

        phi = qml.numpy.array(-0.629)
        results = self.f(dev, phi)
        expected = self.expected(phi)

        self.nested_compare(results, expected)

        if max_workers is not None:
            return

        g0 = qml.jacobian(lambda x: qml.numpy.array(self.f(dev, x)[0]))(phi)
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

        f = jax.jit(self.f_hashable) if use_jit else self.f_hashable
        results = f(phi)
        expected = self.expected(phi)

        self.nested_compare(results, expected)

        g = jax.jacobian(f)(phi)
        g_expected = jax.jacobian(self.expected)(phi)

        self.nested_compare(g, g_expected)

    @pytest.mark.torch
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_torch(self, max_workers):
        """Test batches can be executed and have backprop derivatives in torch."""
        import torch

        dev = DefaultQubit2(max_workers=max_workers)

        x = torch.tensor(9.6243)

        results = self.f(dev, x)
        expected = self.expected(x)

        self.nested_compare(results, expected)

        if max_workers is not None:
            return

        g1 = torch.autograd.functional.jacobian(lambda y: self.f(dev, y)[0], x)
        assert qml.math.allclose(g1[0], -qml.math.cos(x))
        assert qml.math.allclose(g1[1], -3 * qml.math.sin(x))

        g1 = torch.autograd.functional.jacobian(lambda y: self.f(dev, y)[1], x)
        temp = -0.5 * qml.math.cos(x / 2) * qml.math.sin(x / 2)
        g3 = torch.tensor([temp, -temp, temp, -temp])
        assert qml.math.allclose(g1, g3)

    @pytest.mark.tf
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_tf(self, max_workers):
        """Test batches can be executed and have backprop derivatives in tf."""

        import tensorflow as tf

        dev = DefaultQubit2(max_workers=max_workers)

        x = tf.Variable(5.2281)
        with tf.GradientTape(persistent=True) as tape:
            results = self.f(dev, x)

        expected = self.expected(x)
        self.nested_compare(results, expected)

        if max_workers is not None:
            return

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
    def f(dev, scale, n_wires=10, offset=0.1, convert_to_hamiltonian=False):
        """Execute a quantum script with a large Hamiltonian."""
        ops = [qml.RX(offset + scale * i, wires=i) for i in range(n_wires)]

        t1 = 2.5 * qml.prod(*(qml.PauliZ(i) for i in range(n_wires)))
        t2 = 6.2 * qml.prod(*(qml.PauliY(i) for i in range(n_wires)))
        H = t1 + t2
        if convert_to_hamiltonian:
            H = H._pauli_rep.hamiltonian()  # pylint: disable=protected-access
        qs = qml.tape.QuantumScript(ops, [qml.expval(H)])
        return dev.execute(qs)

    @staticmethod
    def f_hashable(scale, n_wires=10, offset=0.1, convert_to_hamiltonian=False):
        """Execute a quantum script with a large Hamiltonian."""
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
        """expected output of f."""
        phase = offset + scale * qml.math.asarray(range(n_wires), like=like)
        cosines = qml.math.cos(phase)
        sines = qml.math.sin(phase)
        return 2.5 * qml.math.prod(cosines) + 6.2 * qml.math.prod(sines)

    @pytest.mark.autograd
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_autograd_backprop(self, convert_to_hamiltonian):
        """Test that backpropagation derivatives work in autograd with hamiltonians and large sums."""
        dev = DefaultQubit2()
        x = qml.numpy.array(0.52)
        out = self.f(dev, x, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_out = self.expected(x)
        assert qml.math.allclose(out, expected_out)

        g = qml.grad(self.f)(dev, x, convert_to_hamiltonian=convert_to_hamiltonian)
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
        f = jax.jit(self.f_hashable, static_argnums=(1, 2, 3)) if use_jit else self.f_hashable

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

        dev = DefaultQubit2()

        x = torch.tensor(-0.289, requires_grad=True)
        x2 = torch.tensor(-0.289, requires_grad=True)
        out = self.f(dev, x, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_out = self.expected(x2, like="torch")
        assert qml.math.allclose(out, expected_out)

        out.backward()  # pylint:disable=no-member
        expected_out.backward()
        assert qml.math.allclose(x.grad, x2.grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_tf_backprop(self, convert_to_hamiltonian):
        """Test that backpropagation derivatives work with tensorflow with hamiltonians and large sums."""
        import tensorflow as tf

        dev = DefaultQubit2()

        x = tf.Variable(0.5)

        with tf.GradientTape() as tape1:
            out = self.f(dev, x, convert_to_hamiltonian=convert_to_hamiltonian)

        with tf.GradientTape() as tape2:
            expected_out = self.expected(x)

        assert qml.math.allclose(out, expected_out)
        g1 = tape1.gradient(out, x)
        g2 = tape2.gradient(expected_out, x)
        assert qml.math.allclose(g1, g2)


@pytest.mark.parametrize("max_workers", [None, 1, 2])
class TestAdjointDifferentiation:
    """Tests adjoint differentiation integration with DefaultQubit2."""

    ec = ExecutionConfig(gradient_method="adjoint")

    def test_derivatives_single_circuit(self, max_workers):
        """Tests derivatives with a single circuit."""
        dev = DefaultQubit2(max_workers=max_workers)
        x = np.array(np.pi / 7)
        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        qs = validate_and_expand_adjoint(qs)
        expected_grad = -qml.math.sin(x)
        actual_grad = dev.compute_derivatives(qs, self.ec)
        assert isinstance(actual_grad, np.ndarray)
        assert actual_grad.shape == ()  # pylint: disable=no-member
        assert np.isclose(actual_grad, expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_derivatives(qs, self.ec)
        assert np.isclose(actual_val, expected_val)
        assert np.isclose(actual_grad, expected_grad)

    def test_derivatives_list_with_single_circuit(self, max_workers):
        """Tests a basic example with a batch containing a single circuit."""
        dev = DefaultQubit2(max_workers=max_workers)
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

    def test_derivatives_many_tapes_many_results(self, max_workers):
        """Tests a basic example with a batch of circuits of varying return shapes."""
        dev = DefaultQubit2(max_workers=max_workers)
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

    def test_derivatives_integration(self, max_workers):
        """Tests the expected workflow done by a calling method."""
        dev = DefaultQubit2(max_workers=max_workers)
        x = np.array(np.pi / 7)
        expected_grad = (-qml.math.sin(x), (qml.math.cos(x), -qml.math.sin(x)))
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )

        circuits, _, new_ec = dev.preprocess([single_meas, multi_meas], self.ec)
        actual_grad = dev.compute_derivatives(circuits, self.ec)

        assert new_ec.use_device_gradient
        assert new_ec.grad_on_execution

        assert np.isclose(actual_grad[0], expected_grad[0])
        assert isinstance(actual_grad[1], tuple)
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

    def test_jvps_single_circuit(self, max_workers):
        """Tests jvps with a single circuit."""
        dev = DefaultQubit2(max_workers=max_workers)
        x = np.array(np.pi / 7)
        tangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        qs = validate_and_expand_adjoint(qs)

        expected_grad = -qml.math.sin(x) * tangent[0]
        actual_grad = dev.compute_jvp(qs, tangent, self.ec)
        assert isinstance(actual_grad, np.ndarray)
        assert actual_grad.shape == ()  # pylint: disable=no-member
        assert np.isclose(actual_grad, expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_jvp(qs, tangent, self.ec)
        assert np.isclose(actual_val, expected_val)
        assert np.isclose(actual_grad, expected_grad)

    def test_jvps_list_with_single_circuit(self, max_workers):
        """Tests a basic example with a batch containing a single circuit."""
        dev = DefaultQubit2(max_workers=max_workers)
        x = np.array(np.pi / 7)
        tangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        qs = validate_and_expand_adjoint(qs)

        expected_grad = -qml.math.sin(x) * tangent[0]
        actual_grad = dev.compute_jvp([qs], [tangent], self.ec)
        assert isinstance(actual_grad, tuple)
        assert isinstance(actual_grad[0], np.ndarray)
        assert np.isclose(actual_grad[0], expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_jvp([qs], [tangent], self.ec)
        assert np.isclose(expected_val, actual_val[0])
        assert np.isclose(expected_grad, actual_grad[0])

    def test_jvps_many_tapes_many_results(self, max_workers):
        """Tests a basic example with a batch of circuits of varying return shapes."""
        dev = DefaultQubit2(max_workers=max_workers)
        x = np.array(np.pi / 7)
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        tangents = [(0.456,), (0.789,)]

        expected_grad = (
            -qml.math.sin(x) * tangents[0][0],
            (qml.math.cos(x) * tangents[1][0], -qml.math.sin(x) * tangents[1][0]),
        )
        actual_grad = dev.compute_jvp([single_meas, multi_meas], tangents, self.ec)
        assert np.isclose(actual_grad[0], expected_grad[0])
        assert isinstance(actual_grad[1], tuple)
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

        expected_val = (qml.math.cos(x), (qml.math.sin(x), qml.math.cos(x)))
        actual_val, actual_grad = dev.execute_and_compute_jvp(
            [single_meas, multi_meas], tangents, self.ec
        )
        assert np.isclose(actual_val[0], expected_val[0])
        assert qml.math.allclose(actual_val[1], expected_val[1])
        assert np.isclose(actual_grad[0], expected_grad[0])
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

    def test_jvps_integration(self, max_workers):
        """Tests the expected workflow done by a calling method."""
        dev = DefaultQubit2(max_workers=max_workers)
        x = np.array(np.pi / 7)

        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        tangents = [(0.456,), (0.789,)]

        circuits, _, new_ec = dev.preprocess([single_meas, multi_meas], self.ec)
        actual_grad = dev.compute_jvp(circuits, tangents, self.ec)
        expected_grad = (
            -qml.math.sin(x) * tangents[0][0],
            (qml.math.cos(x) * tangents[1][0], -qml.math.sin(x) * tangents[1][0]),
        )

        assert new_ec.use_device_gradient
        assert new_ec.grad_on_execution

        assert np.isclose(actual_grad[0], expected_grad[0])
        assert isinstance(actual_grad[1], tuple)
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

    def test_vjps_single_circuit(self, max_workers):
        """Tests vjps with a single circuit."""
        dev = DefaultQubit2(max_workers=max_workers)
        x = np.array(np.pi / 7)
        cotangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        qs = validate_and_expand_adjoint(qs)

        expected_grad = -qml.math.sin(x) * cotangent[0]
        actual_grad = dev.compute_vjp(qs, cotangent, self.ec)
        assert isinstance(actual_grad, np.ndarray)
        assert actual_grad.shape == ()  # pylint: disable=no-member
        assert np.isclose(actual_grad, expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_vjp(qs, cotangent, self.ec)
        assert np.isclose(actual_val, expected_val)
        assert np.isclose(actual_grad, expected_grad)

    def test_vjps_list_with_single_circuit(self, max_workers):
        """Tests a basic example with a batch containing a single circuit."""
        dev = DefaultQubit2(max_workers=max_workers)
        x = np.array(np.pi / 7)
        cotangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        qs = validate_and_expand_adjoint(qs)

        expected_grad = -qml.math.sin(x) * cotangent[0]
        actual_grad = dev.compute_vjp([qs], [cotangent], self.ec)
        assert isinstance(actual_grad, tuple)
        assert isinstance(actual_grad[0], np.ndarray)
        assert np.isclose(actual_grad[0], expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_vjp([qs], [cotangent], self.ec)
        assert np.isclose(expected_val, actual_val[0])
        assert np.isclose(expected_grad, actual_grad[0])

    def test_vjps_many_tapes_many_results(self, max_workers):
        """Tests a basic example with a batch of circuits of varying return shapes."""
        dev = DefaultQubit2(max_workers=max_workers)
        x = np.array(np.pi / 7)
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        cotangents = [(0.456,), (0.789, 0.123)]

        expected_grad = (
            -qml.math.sin(x) * cotangents[0][0],
            qml.math.cos(x) * cotangents[1][0] - qml.math.sin(x) * cotangents[1][1],
        )
        actual_grad = dev.compute_vjp([single_meas, multi_meas], cotangents, self.ec)
        assert np.isclose(actual_grad[0], expected_grad[0])
        assert np.isclose(actual_grad[1], expected_grad[1])

        expected_val = (qml.math.cos(x), (qml.math.sin(x), qml.math.cos(x)))
        actual_val, actual_grad = dev.execute_and_compute_vjp(
            [single_meas, multi_meas], cotangents, self.ec
        )
        assert np.isclose(actual_val[0], expected_val[0])
        assert qml.math.allclose(actual_val[1], expected_val[1])
        assert np.isclose(actual_grad[0], expected_grad[0])
        assert np.isclose(actual_grad[1], expected_grad[1])

    def test_vjps_integration(self, max_workers):
        """Tests the expected workflow done by a calling method."""
        dev = DefaultQubit2(max_workers=max_workers)
        x = np.array(np.pi / 7)

        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        cotangents = [(0.456,), (0.789, 0.123)]

        circuits, _, new_ec = dev.preprocess([single_meas, multi_meas], self.ec)
        actual_grad = dev.compute_vjp(circuits, cotangents, self.ec)
        expected_grad = (
            -qml.math.sin(x) * cotangents[0][0],
            qml.math.cos(x) * cotangents[1][0] - qml.math.sin(x) * cotangents[1][1],
        )

        assert new_ec.use_device_gradient
        assert new_ec.grad_on_execution

        assert np.isclose(actual_grad[0], expected_grad[0])
        assert np.isclose(actual_grad[1], expected_grad[1])


class TestPreprocessingIntegration:
    """Test preprocess produces output that can be executed by the device."""

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_preprocess_single_circuit(self, max_workers):
        """Test integration between preprocessing and execution with numpy parameters."""

        # pylint: disable=too-few-public-methods
        class MyTemplate(qml.operation.Operation):
            """Temp operator."""

            num_wires = 2

            # pylint: disable=missing-function-docstring
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

        dev = DefaultQubit2(max_workers=max_workers)

        batch, post_procesing_fn, config = dev.preprocess(qscript)

        assert len(batch) == 1
        execute_circuit = batch[0]
        assert qml.equal(execute_circuit[0], qml.RX(x, "a"))
        assert qml.equal(execute_circuit[1], qml.RY(y, "b"))
        assert qml.equal(execute_circuit[2], qml.CNOT(("a", "b")))
        assert qml.equal(execute_circuit[3], qml.expval(qml.PauliY("a")))
        assert qml.equal(execute_circuit[4], qml.expval(qml.PauliZ("a")))
        assert qml.equal(execute_circuit[5], qml.expval(qml.PauliX("b")))

        results = dev.execute(batch, config)
        assert len(results) == 1
        assert len(results[0]) == 3

        processed_results = post_procesing_fn(results)
        assert len(processed_results) == 3
        assert qml.math.allclose(processed_results[0], -np.sin(x) * np.sin(y))
        assert qml.math.allclose(processed_results[1], np.cos(x))
        assert qml.math.allclose(processed_results[2], np.sin(y))

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_preprocess_batch_circuit(self, max_workers):
        """Test preprocess integrates with default qubit when we start with a batch of circuits."""

        # pylint: disable=too-few-public-methods
        class CustomIsingXX(qml.operation.Operation):
            """Temp operator."""

            num_wires = 2

            # pylint: disable=missing-function-docstring
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

        dev = DefaultQubit2(max_workers=max_workers)
        batch, post_processing_fn, config = dev.preprocess(initial_batch)

        results = dev.execute(batch, config)
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


class TestRandomSeed:
    """Test that the device behaves correctly when provided with a random seed"""

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.sample(wires=0)],
            [qml.expval(qml.PauliZ(0))],
            [qml.probs(wires=0)],
            [qml.sample(wires=0), qml.expval(qml.PauliZ(0)), qml.probs(wires=0)],
        ],
    )
    def test_same_seed(self, measurements, max_workers):
        """Test that different devices given the same random seed will produce
        the same results"""
        qs = qml.tape.QuantumScript([qml.Hadamard(0)], measurements, shots=1000)

        dev1 = DefaultQubit2(seed=123, max_workers=max_workers)
        result1 = dev1.execute(qs)

        dev2 = DefaultQubit2(seed=123, max_workers=max_workers)
        result2 = dev2.execute(qs)

        if len(measurements) == 1:
            result1, result2 = [result1], [result2]

        assert all(np.all(res1 == res2) for res1, res2 in zip(result1, result2))

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_different_seed(self, max_workers):
        """Test that different devices given different random seeds will produce
        different results (with almost certainty)"""
        qs = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.sample(wires=0)], shots=1000)

        dev1 = DefaultQubit2(seed=None, max_workers=max_workers)
        result1 = dev1.execute(qs)

        dev2 = DefaultQubit2(seed=123, max_workers=max_workers)
        result2 = dev2.execute(qs)

        dev3 = DefaultQubit2(seed=456, max_workers=max_workers)
        result3 = dev3.execute(qs)

        # assert results are pairwise different
        assert np.any(result1 != result2)
        assert np.any(result1 != result3)
        assert np.any(result2 != result3)

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.sample(wires=0)],
            [qml.expval(qml.PauliZ(0))],
            [qml.probs(wires=0)],
            [qml.sample(wires=0), qml.expval(qml.PauliZ(0)), qml.probs(wires=0)],
        ],
    )
    def test_different_executions(self, measurements, max_workers):
        """Test that the same device will produce different results every execution"""
        qs = qml.tape.QuantumScript([qml.Hadamard(0)], measurements, shots=1000)

        dev = DefaultQubit2(seed=123, max_workers=max_workers)
        result1 = dev.execute(qs)
        result2 = dev.execute(qs)

        if len(measurements) == 1:
            result1, result2 = [result1], [result2]

        assert all(np.any(res1 != res2) for res1, res2 in zip(result1, result2))

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.sample(wires=0)],
            [qml.expval(qml.PauliZ(0))],
            [qml.probs(wires=0)],
            [qml.sample(wires=0), qml.expval(qml.PauliZ(0)), qml.probs(wires=0)],
        ],
    )
    def test_global_seed_and_device_seed(self, measurements, max_workers):
        """Test that a global seed does not affect the result of devices
        provided with a seed"""
        qs = qml.tape.QuantumScript([qml.Hadamard(0)], measurements, shots=1000)

        # expected result
        dev1 = DefaultQubit2(seed=123, max_workers=max_workers)
        result1 = dev1.execute(qs)

        # set a global seed both before initialization of the
        # device and before execution of the tape
        np.random.seed(456)
        dev2 = DefaultQubit2(seed=123, max_workers=max_workers)
        np.random.seed(789)
        result2 = dev2.execute(qs)

        if len(measurements) == 1:
            result1, result2 = [result1], [result2]

        assert all(np.all(res1 == res2) for res1, res2 in zip(result1, result2))

    def test_global_seed_no_device_seed_by_default(self):
        """Test that the global numpy seed initializes the rng if device seed is none."""
        np.random.seed(42)
        dev = DefaultQubit2()
        first_num = dev._rng.random()  # pylint: disable=protected-access

        np.random.seed(42)
        dev2 = DefaultQubit2()
        second_num = dev2._rng.random()  # pylint: disable=protected-access

        assert qml.math.allclose(first_num, second_num)

        np.random.seed(42)
        dev2 = DefaultQubit2(seed="global")
        third_num = dev2._rng.random()  # pylint: disable=protected-access

        assert qml.math.allclose(third_num, first_num)

    def test_None_seed_not_using_global_rng(self):
        """Test that if the seed is None, it is uncorrelated with the global rng."""
        np.random.seed(42)
        dev = DefaultQubit2(seed=None)
        first_nums = dev._rng.random(10)  # pylint: disable=protected-access

        np.random.seed(42)
        dev2 = DefaultQubit2(seed=None)
        second_nums = dev2._rng.random(10)  # pylint: disable=protected-access

        assert not qml.math.allclose(first_nums, second_nums)

    def test_rng_as_seed(self):
        """Test that a PRNG can be passed as a seed."""
        rng1 = np.random.default_rng(42)
        first_num = rng1.random()

        rng = np.random.default_rng(42)
        dev = DefaultQubit2(seed=rng)
        second_num = dev._rng.random()  # pylint: disable=protected-access

        assert qml.math.allclose(first_num, second_num)


class TestHamiltonianSamples:
    """Test that the measure_with_samples function works as expected for
    Hamiltonian and Sum observables

    This is a copy of the tests in test_sampling.py, but using the device instead"""

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_hamiltonian_expval(self, max_workers):
        """Test that sampling works well for Hamiltonian observables"""
        x, y = np.array(0.67), np.array(0.95)
        ops = [qml.RY(x, wires=0), qml.RZ(y, wires=0)]
        meas = [qml.expval(qml.Hamiltonian([0.8, 0.5], [qml.PauliZ(0), qml.PauliX(0)]))]

        dev = DefaultQubit2(seed=100, max_workers=max_workers)
        qs = qml.tape.QuantumScript(ops, meas, shots=10000)
        res = dev.execute(qs)

        expected = 0.8 * np.cos(x) + 0.5 * np.real(np.exp(y * 1j)) * np.sin(x)
        assert np.allclose(res, expected, atol=0.01)

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_sum_expval(self, max_workers):
        """Test that sampling works well for Sum observables"""
        x, y = np.array(0.67), np.array(0.95)
        ops = [qml.RY(x, wires=0), qml.RZ(y, wires=0)]
        meas = [qml.expval(qml.s_prod(0.8, qml.PauliZ(0)) + qml.s_prod(0.5, qml.PauliX(0)))]

        dev = DefaultQubit2(seed=100, max_workers=max_workers)
        qs = qml.tape.QuantumScript(ops, meas, shots=10000)
        res = dev.execute(qs)

        expected = 0.8 * np.cos(x) + 0.5 * np.real(np.exp(y * 1j)) * np.sin(x)
        assert np.allclose(res, expected, atol=0.01)

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_multi_wires(self, max_workers):
        """Test that sampling works for Sums with large numbers of wires"""
        n_wires = 10
        scale = 0.05
        offset = 0.8

        ops = [qml.RX(offset + scale * i, wires=i) for i in range(n_wires)]

        t1 = 2.5 * qml.prod(*(qml.PauliZ(i) for i in range(n_wires)))
        t2 = 6.2 * qml.prod(*(qml.PauliY(i) for i in range(n_wires)))
        H = t1 + t2

        dev = DefaultQubit2(seed=100, max_workers=max_workers)
        qs = qml.tape.QuantumScript(ops, [qml.expval(H)], shots=100000)
        res = dev.execute(qs)

        phase = offset + scale * np.array(range(n_wires))
        cosines = qml.math.cos(phase)
        sines = qml.math.sin(phase)
        expected = 2.5 * qml.math.prod(cosines) + 6.2 * qml.math.prod(sines)

        assert np.allclose(res, expected, atol=0.05)

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_complex_hamiltonian(self, max_workers):
        """Test that sampling works for complex Hamiltonians"""
        scale = 0.05
        offset = 0.4

        ops = [qml.RX(offset + scale * i, wires=i) for i in range(4)]

        # taken from qml.data
        H = qml.Hamiltonian(
            [
                -0.3796867241618816,
                0.1265398827193729,
                0.1265398827193729,
                0.15229282586796247,
                0.05080559325437572,
                -0.05080559325437572,
                -0.05080559325437572,
                0.05080559325437572,
                -0.10485523662149618,
                0.10102818539518765,
                -0.10485523662149615,
                0.15183377864956338,
                0.15183377864956338,
                0.10102818539518765,
                0.1593698831813122,
            ],
            [
                qml.Identity(wires=[0]),
                qml.PauliZ(wires=[0]),
                qml.PauliZ(wires=[1]),
                qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
                qml.PauliY(wires=[0])
                @ qml.PauliX(wires=[1])
                @ qml.PauliX(wires=[2])
                @ qml.PauliY(wires=[3]),
                qml.PauliY(wires=[0])
                @ qml.PauliY(wires=[1])
                @ qml.PauliX(wires=[2])
                @ qml.PauliX(wires=[3]),
                qml.PauliX(wires=[0])
                @ qml.PauliX(wires=[1])
                @ qml.PauliY(wires=[2])
                @ qml.PauliY(wires=[3]),
                qml.PauliX(wires=[0])
                @ qml.PauliY(wires=[1])
                @ qml.PauliY(wires=[2])
                @ qml.PauliX(wires=[3]),
                qml.PauliZ(wires=[2]),
                qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
                qml.PauliZ(wires=[3]),
                qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
                qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
                qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
                qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
            ],
        )

        dev = DefaultQubit2(seed=100, max_workers=max_workers)
        qs = qml.tape.QuantumScript(ops, [qml.expval(H)], shots=100000)
        res = dev.execute(qs)

        qs_exp = qml.tape.QuantumScript(ops, [qml.expval(H)])
        expected = dev.execute(qs_exp)

        assert np.allclose(res, expected, atol=0.001)


class TestClassicalShadows:
    """Test that classical shadow measurements works with the new device"""

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_shape_and_dtype(self, max_workers, n_qubits):
        """Test that the shape and dtype of the measurement is correct"""
        dev = DefaultQubit2(max_workers=max_workers)

        ops = [qml.Hadamard(i) for i in range(n_qubits)]
        qs = qml.tape.QuantumScript(ops, [qml.classical_shadow(range(n_qubits))], shots=100)
        res = dev.execute(qs)

        assert res.shape == (2, 100, n_qubits)
        assert res.dtype == np.int8

        # test that the bits are either 0 and 1
        assert np.all(np.logical_or(res[0] == 0, res[0] == 1))

        # test that the recipes are either 0, 1, or 2 (X, Y, or Z)
        assert np.all(np.logical_or(np.logical_or(res[1] == 0, res[1] == 1), res[1] == 2))

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_expval(self, max_workers):
        """Test that shadow expval measurements work as expected"""
        dev = DefaultQubit2(seed=100, max_workers=max_workers)

        ops = [qml.Hadamard(0), qml.Hadamard(1)]
        meas = [qml.shadow_expval(qml.PauliX(0) @ qml.PauliX(1), seed=200)]
        qs = qml.tape.QuantumScript(ops, meas, shots=1000)
        res = dev.execute(qs)

        assert res.shape == ()
        assert np.allclose(res, 1.0, atol=0.05)

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_reconstruct_bell_state(self, max_workers):
        """Test that a bell state can be faithfully reconstructed"""
        dev = DefaultQubit2(seed=100, max_workers=max_workers)

        ops = [qml.Hadamard(0), qml.CNOT([0, 1])]
        meas = [qml.classical_shadow(wires=[0, 1], seed=200)]
        qs = qml.tape.QuantumScript(ops, meas, shots=10000)

        # should prepare the bell state
        bits, recipes = dev.execute(qs)
        shadow = qml.shadows.ClassicalShadow(bits, recipes)
        global_snapshots = shadow.global_snapshots()

        state = np.sum(global_snapshots, axis=0) / shadow.snapshots
        bell_state = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
        assert qml.math.allclose(state, bell_state, atol=0.1)

        # reduced state should yield maximally mixed state
        local_snapshots = shadow.local_snapshots(wires=[0])
        assert qml.math.allclose(np.mean(local_snapshots, axis=0)[0], 0.5 * np.eye(2), atol=0.05)

        # alternative computation
        ops = [qml.Hadamard(0), qml.CNOT([0, 1])]
        meas = [qml.classical_shadow(wires=[0], seed=200)]
        qs = qml.tape.QuantumScript(ops, meas, shots=10000)
        bits, recipes = dev.execute(qs)

        shadow = qml.shadows.ClassicalShadow(bits, recipes)
        global_snapshots = shadow.global_snapshots()
        local_snapshots = shadow.local_snapshots(wires=[0])

        state = np.sum(global_snapshots, axis=0) / shadow.snapshots
        assert qml.math.allclose(state, 0.5 * np.eye(2), atol=0.1)
        assert np.all(local_snapshots[:, 0] == global_snapshots)

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    @pytest.mark.parametrize(
        "shots",
        [
            [1000, 1000],
            [(1000, 2)],
            [1000, 2000],
            [(1000, 2), 2000],
            [(1000, 3), 2000, (3000, 2)],
        ],
    )
    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_shot_vectors(self, max_workers, n_qubits, shots):
        """Test that classical shadows works when given a shot vector"""
        dev = DefaultQubit2(max_workers=max_workers)
        shots = qml.measurements.Shots(shots)

        ops = [qml.Hadamard(i) for i in range(n_qubits)]
        qs = qml.tape.QuantumScript(ops, [qml.classical_shadow(range(n_qubits))], shots=shots)
        res = dev.execute(qs)

        assert isinstance(res, tuple)
        assert len(res) == len(list(shots))

        for r, s in zip(res, shots):
            assert r.shape == (2, s, n_qubits)
            assert r.dtype == np.int8

            # test that the bits are either 0 and 1
            assert np.all(np.logical_or(r[0] == 0, r[0] == 1))

            # test that the recipes are either 0, 1, or 2 (X, Y, or Z)
            assert np.all(np.logical_or(np.logical_or(r[1] == 0, r[1] == 1), r[1] == 2))


@pytest.mark.parametrize("max_workers", [None, 1, 2])
def test_broadcasted_parameter(max_workers):
    """Test that DefaultQubit2 handles broadcasted parameters as expected."""
    dev = DefaultQubit2(max_workers=max_workers)
    x = np.array([0.536, 0.894])
    qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])

    config = ExecutionConfig()
    config.gradient_method = "adjoint"
    batch, post_processing_fn, config = dev.preprocess(qs, config)

    assert len(batch) == 2
    results = dev.execute(batch, config)
    processed_results = post_processing_fn(results)
    assert qml.math.allclose(processed_results, np.cos(x))
