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
"""Tests for default qubit."""
# pylint: disable=import-outside-toplevel, no-member, too-many-arguments

from unittest import mock

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices import DefaultQubit, ExecutionConfig
from pennylane.exceptions import DeviceError

max_workers_list = [
    None,
    pytest.param(1, marks=pytest.mark.slow),
    pytest.param(2, marks=pytest.mark.slow),
]


def test_name():
    """Tests the name of DefaultQubit."""
    assert DefaultQubit().name == "default.qubit"


def test_shots():
    """Test the shots property of DefaultQubit."""
    assert DefaultQubit().shots == qml.measurements.Shots(None)
    with pytest.warns(
        qml.exceptions.PennyLaneDeprecationWarning, match="shots on device is deprecated"
    ):
        assert DefaultQubit(shots=100).shots == qml.measurements.Shots(100)

    with pytest.raises(AttributeError):
        DefaultQubit().shots = 10


def test_wires():
    """Test that a device can be created with wires."""
    assert DefaultQubit().wires is None
    assert DefaultQubit(wires=2).wires == qml.wires.Wires([0, 1])
    assert DefaultQubit(wires=[0, 2]).wires == qml.wires.Wires([0, 2])

    with pytest.raises(AttributeError):
        DefaultQubit().wires = [0, 1]


def test_debugger_attribute():
    """Test that DefaultQubit has a debugger attribute and that it is `None`"""
    # pylint: disable=protected-access
    dev = DefaultQubit()

    assert hasattr(dev, "_debugger")
    assert dev._debugger is None


def test_snapshot_multiprocessing_qnode():
    """DefaultQubit cannot execute tapes with Snapshot if `max_workers` is not `None`"""
    dev = DefaultQubit(max_workers=2)

    @qml.qnode(dev)
    def circuit():
        qml.Snapshot("tag")
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Snapshot()
        return qml.expval(qml.PauliX(0) + qml.PauliY(0))

    with pytest.raises(
        DeviceError,
        match="Debugging with ``Snapshots`` is not available with multiprocessing.",
    ):
        qml.snapshots(circuit)()


# pylint: disable=protected-access
def test_applied_modifiers():
    """Test that default qubit has the `single_tape_support` and `simulator_tracking`
    modifiers applied.
    """
    dev = DefaultQubit()
    assert dev._applied_modifiers == [
        qml.devices.modifiers.single_tape_support,
        qml.devices.modifiers.simulator_tracking,
    ]


class TestSupportsDerivatives:
    """Test that DefaultQubit states what kind of derivatives it supports."""

    def test_supports_backprop(self):
        """Test that DefaultQubit says that it supports backpropagation."""
        dev = DefaultQubit()
        assert dev.supports_derivatives() is True
        assert dev.supports_jvp() is True
        assert dev.supports_vjp() is True

        config = ExecutionConfig(gradient_method="backprop", interface="auto")
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

        config = ExecutionConfig(gradient_method="backprop", interface=None)
        assert dev.supports_derivatives(config) is True
        assert dev.supports_jvp(config) is True
        assert dev.supports_vjp(config) is True

    @pytest.mark.parametrize(
        "device_wires, measurement",
        [
            (None, qml.expval(qml.PauliZ(0))),
            (2, qml.expval(qml.PauliZ(0))),
            (2, qml.probs()),
            (2, qml.probs([0])),
        ],
    )
    def test_supports_adjoint(self, device_wires, measurement):
        """Test that DefaultQubit says that it supports adjoint differentiation."""
        dev = DefaultQubit(wires=device_wires)
        config = ExecutionConfig(gradient_method="adjoint", use_device_gradient=True)
        assert dev.supports_derivatives(config) is True
        assert dev.supports_jvp(config) is True
        assert dev.supports_vjp(config) is True

        qs = qml.tape.QuantumScript([], [measurement])
        assert dev.supports_derivatives(config, qs) is True
        assert dev.supports_jvp(config, qs) is True
        assert dev.supports_vjp(config, qs) is True

    def test_doesnt_support_adjoint_with_invalid_tape(self):
        """Tests that DefaultQubit does not support adjoint differentiation with invalid circuits."""
        dev = DefaultQubit()
        config = ExecutionConfig(gradient_method="adjoint")
        circuit = qml.tape.QuantumScript([], [qml.sample()], shots=10)
        assert dev.supports_derivatives(config, circuit=circuit) is False
        assert dev.supports_jvp(config, circuit=circuit) is False
        assert dev.supports_vjp(config, circuit=circuit) is False

        circuit = qml.tape.QuantumScript(
            [qml.measurements.MidMeasureMP(0)], [qml.expval(qml.PauliZ(0))]
        )
        assert dev.supports_derivatives(config, circuit=circuit) is False
        assert dev.supports_jvp(config, circuit=circuit) is False
        assert dev.supports_vjp(config, circuit=circuit) is False

    @pytest.mark.parametrize("gradient_method", ["parameter-shift", "finite-diff", "device"])
    def test_doesnt_support_other_gradient_methods(self, gradient_method):
        """Test that DefaultQubit currently does not support other gradient methods natively."""
        dev = DefaultQubit()
        config = ExecutionConfig(gradient_method=gradient_method)
        assert dev.supports_derivatives(config) is False
        assert dev.supports_jvp(config) is False
        assert dev.supports_vjp(config) is False


class TestBasicCircuit:
    """Tests a basic circuit with one RX gate and two simple expectation values."""

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_basic_circuit_numpy(self, max_workers):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )

        dev = DefaultQubit(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_basic_circuit_numpy_with_config(self, max_workers):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )

        dev = DefaultQubit(max_workers=max_workers)
        config = ExecutionConfig(
            device_options={"max_workers": dev._max_workers}  # pylint: disable=protected-access
        )
        result = dev.execute(qs, execution_config=config)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

    @pytest.mark.autograd
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_autograd_results_and_backprop(self, max_workers):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        dev = DefaultQubit(max_workers=max_workers)

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
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_jax_results_and_backprop(self, use_jit, max_workers):
        """Tests execution and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        dev = DefaultQubit(max_workers=max_workers)

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
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_torch_results_and_backprop(self, max_workers):
        """Tests execution and gradients of a simple circuit with torch."""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        dev = DefaultQubit(max_workers=max_workers)

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
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_tf_results_and_backprop(self, max_workers):
        """Tests execution and gradients of a simple circuit with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873, dtype="float64")

        dev = DefaultQubit(max_workers=max_workers)

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

    @pytest.mark.tf
    @pytest.mark.parametrize("op,param", [(qml.RX, np.pi), (qml.BasisState, [1])])
    def test_qnode_returns_correct_interface(self, op, param):
        """Test that even if no interface parameters are given, result is correct."""
        dev = DefaultQubit()

        @qml.qnode(dev, interface="tf")
        def circuit(p):
            op(p, wires=[0])
            return qml.expval(qml.PauliZ(0))

        res = circuit(param)
        assert qml.math.get_interface(res) == "tensorflow"
        assert qml.math.allclose(res, -1)

    def test_basis_state_wire_order(self):
        """Test that the wire order is correct with a basis state if the tape wires have a non standard order."""

        dev = DefaultQubit()

        tape = qml.tape.QuantumScript([qml.BasisState([1], wires=1), qml.PauliZ(0)], [qml.state()])

        expected = np.array([0, 1, 0, 0], dtype=np.complex128)
        res = dev.execute(tape)
        assert qml.math.allclose(res, expected)


class TestSampleMeasurements:
    """A copy of the `qubit.simulate` tests, but using the device"""

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_single_expval(self, max_workers):
        """Test a simple circuit with a single expval measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.expval(qml.PauliZ(0))], shots=10000)

        dev = DefaultQubit(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, (float, np.ndarray))
        assert result.shape == ()
        assert np.allclose(result, np.cos(x), atol=0.1)

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_single_probs(self, max_workers):
        """Test a simple circuit with a single prob measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.probs(wires=0)], shots=10000)

        dev = DefaultQubit(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, (float, np.ndarray))
        assert result.shape == (2,)
        assert np.allclose(result, [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2], atol=0.1)

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_single_sample(self, max_workers):
        """Test a simple circuit with a single sample measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=10000)

        dev = DefaultQubit(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, (float, np.ndarray))
        assert result.shape == (10000, 2)
        assert np.allclose(
            np.sum(result, axis=0).astype(np.float32) / 10000, [np.sin(x / 2) ** 2, 0], atol=0.1
        )

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_multi_measurements(self, max_workers):
        """Test a simple circuit containing multiple measurements"""
        x, y = np.array(0.732), np.array(0.488)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
            [qml.expval(qml.Hadamard(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
            shots=10000,
        )

        dev = DefaultQubit(max_workers=max_workers)
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
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_expval_shot_vector(self, max_workers, shots):
        """Test a simple circuit with a single expval measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.expval(qml.PauliZ(0))], shots=shots)

        dev = DefaultQubit(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, (float, np.ndarray)) for res in result)
        assert all(res.shape == () for res in result)
        assert all(np.allclose(res, np.cos(x), atol=0.1) for res in result)

    @pytest.mark.parametrize("shots", shots_data)
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_probs_shot_vector(self, max_workers, shots):
        """Test a simple circuit with a single prob measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.probs(wires=0)], shots=shots)

        dev = DefaultQubit(max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, (float, np.ndarray)) for res in result)
        assert all(res.shape == (2,) for res in result)
        assert all(
            np.allclose(res, [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2], atol=0.1) for res in result
        )

    @pytest.mark.parametrize("shots", shots_data)
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_sample_shot_vector(self, max_workers, shots):
        """Test a simple circuit with a single sample measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=shots)

        dev = DefaultQubit(max_workers=max_workers)
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
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_multi_measurement_shot_vector(self, max_workers, shots):
        """Test a simple circuit containing multiple measurements for shot vectors"""
        x, y = np.array(0.732), np.array(0.488)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
            [qml.expval(qml.Hadamard(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
            shots=shots,
        )

        dev = DefaultQubit(max_workers=max_workers)
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

    @pytest.mark.parametrize("max_workers", max_workers_list)
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

        dev = DefaultQubit(max_workers=max_workers)
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

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_batch_tapes(self, max_workers):
        """Test that a batch of tapes with sampling works as expected"""
        x = np.array(0.732)
        qs1 = qml.tape.QuantumScript([qml.RX(x, wires=0)], [qml.sample(wires=(0, 1))], shots=100)
        qs2 = qml.tape.QuantumScript([qml.RX(x, wires=0)], [qml.sample(wires=1)], shots=50)

        dev = DefaultQubit(max_workers=max_workers)
        results = dev.execute((qs1, qs2))

        assert isinstance(results, tuple)
        assert len(results) == 2
        assert all(isinstance(res, (float, np.ndarray)) for res in results)
        assert results[0].shape == (100, 2)
        assert results[1].shape == (50, 1)

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_counts_wires(self, max_workers, seed):
        """Test that a Counts measurement with wires works as expected"""
        x = np.array(np.pi / 2)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.counts(wires=[0, 1])], shots=10000)

        dev = DefaultQubit(seed=seed, max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"00", "10"}

        # check that the count values match the expected
        values = list(result.values())
        assert np.allclose(values[0] / (values[0] + values[1]), 0.5, atol=0.02)

    @pytest.mark.parametrize("max_workers", max_workers_list)
    @pytest.mark.parametrize("all_outcomes", [False, True])
    def test_counts_obs(self, all_outcomes, max_workers, seed):
        """Test that a Counts measurement with an observable works as expected"""
        x = np.array(np.pi / 2)
        qs = qml.tape.QuantumScript(
            [qml.RY(x, wires=0)],
            [qml.counts(qml.PauliZ(0), all_outcomes=all_outcomes)],
            shots=10000,
        )

        dev = DefaultQubit(seed=seed, max_workers=max_workers)
        result = dev.execute(qs)

        assert isinstance(result, dict)
        assert set(result.keys()) == {1, -1}

        # check that the count values match the expected
        values = list(result.values())
        assert np.allclose(values[0] / (values[0] + values[1]), 0.5, rtol=0.05)


class TestExecutingBatches:
    """Tests involving executing multiple circuits at the same time."""

    @staticmethod
    def f(dev, phi):
        """A function that executes a batch of scripts on DefaultQubit without preprocessing."""
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
        """A function that executes a batch of scripts on DefaultQubit without preprocessing."""
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
        return DefaultQubit().execute((qs1, qs2))

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

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_numpy(self, max_workers):
        """Test that results are expected when the parameter does not have a parameter."""
        dev = DefaultQubit(max_workers=max_workers)

        phi = 0.892
        results = self.f(dev, phi)
        expected = self.expected(phi)

        self.nested_compare(results, expected)

    @pytest.mark.autograd
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_autograd(self, max_workers):
        """Test batches can be executed and have backprop derivatives in autograd."""
        dev = DefaultQubit(max_workers=max_workers)

        phi = qml.numpy.array(-0.629)
        results = self.f(dev, phi)
        expected = self.expected(phi)

        self.nested_compare(results, expected)

        if max_workers is not None:
            return

        g0 = qml.jacobian(lambda x: qml.numpy.array(self.f(dev, x)[0]))(phi)
        g0_expected = qml.jacobian(lambda x: qml.numpy.array(self.expected(x)[0]))(phi)
        assert qml.math.allclose(g0, g0_expected)

        g1 = qml.jacobian(lambda x: qml.numpy.array(self.f(dev, x)[1]))(phi)
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
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_torch(self, max_workers):
        """Test batches can be executed and have backprop derivatives in torch."""
        import torch

        dev = DefaultQubit(max_workers=max_workers)

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
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_tf(self, max_workers):
        """Test batches can be executed and have backprop derivatives in tf."""

        import tensorflow as tf

        dev = DefaultQubit(max_workers=max_workers)

        x = tf.Variable(5.2281, dtype="float64")
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

    @pytest.mark.jax
    def test_warning_if_jitting_batch(self):
        """Test that a warning is given if end-to-end jitting is enabled with a batch."""
        config = qml.devices.ExecutionConfig(convert_to_numpy=False, interface="jax-jit")
        batch = [qml.tape.QuantumScript([qml.RX(i, 0)], [qml.expval(qml.Z(0))]) for i in range(11)]
        with pytest.warns(UserWarning, match="substantial classical overhead"):
            _ = DefaultQubit().execute(batch, config)


@pytest.mark.slow
class TestSumOfTermsDifferentiability:
    """Basically a copy of the `qubit.simulate` test but using the device instead."""

    @staticmethod
    def f(dev, scale, n_wires=10, offset=0.1, style="sum"):
        """Execute a quantum script with a large Hamiltonian."""
        ops = [qml.RX(offset + scale * i, wires=i) for i in range(n_wires)]

        t1 = 2.5 * qml.prod(*(qml.PauliZ(i) for i in range(n_wires)))
        t2 = 6.2 * qml.prod(*(qml.PauliY(i) for i in range(n_wires)))
        H = t1 + t2
        if style == "hermitian":
            H = qml.Hermitian(H.matrix(), wires=H.wires)
        qs = qml.tape.QuantumScript(ops, [qml.expval(H)])
        return dev.execute(qs)

    @staticmethod
    def f_hashable(scale, n_wires=10, offset=0.1, style="sum"):
        """Execute a quantum script with a large Hamiltonian."""
        ops = [qml.RX(offset + scale * i, wires=i) for i in range(n_wires)]

        t1 = 2.5 * qml.prod(*(qml.PauliZ(i) for i in range(n_wires)))
        t2 = 6.2 * qml.prod(*(qml.PauliY(i) for i in range(n_wires)))
        H = t1 + t2
        if style == "hermitian":
            H = qml.Hermitian(H.matrix(), wires=H.wires)
        qs = qml.tape.QuantumScript(ops, [qml.expval(H)])
        return DefaultQubit().execute(qs)

    @staticmethod
    def expected(scale, n_wires=10, offset=0.1, like="numpy"):
        """expected output of f."""
        phase = offset + scale * qml.math.asarray(range(n_wires), like=like)
        cosines = qml.math.cos(phase)
        sines = qml.math.sin(phase)
        return 2.5 * qml.math.prod(cosines) + 6.2 * qml.math.prod(sines)

    @pytest.mark.autograd
    @pytest.mark.parametrize("style", ("sum", "hermitian"))
    def test_autograd_backprop(self, style):
        """Test that backpropagation derivatives work in autograd with hamiltonians and large sums."""
        dev = DefaultQubit()
        x = qml.numpy.array(0.52)
        out = self.f(dev, x, style=style)
        expected_out = self.expected(x)
        assert qml.math.allclose(out, expected_out)

        g = qml.grad(self.f)(dev, x, style=style)
        expected_g = qml.grad(self.expected)(x)
        assert qml.math.allclose(g, expected_g)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    @pytest.mark.parametrize("style", ("sum", "hermitian"))
    def test_jax_backprop(self, style, use_jit):
        """Test that backpropagation derivatives work with jax with hamiltonians and large sums."""
        import jax

        x = jax.numpy.array(0.52, dtype=jax.numpy.float64)
        f = jax.jit(self.f_hashable, static_argnums=(1, 2, 3)) if use_jit else self.f_hashable

        out = f(x, style=style)
        expected_out = self.expected(x)
        assert qml.math.allclose(out, expected_out, atol=1e-6)

        g = jax.grad(f)(x, style=style)
        expected_g = jax.grad(self.expected)(x)
        assert qml.math.allclose(g, expected_g)

    @pytest.mark.torch
    @pytest.mark.parametrize("style", ("sum", "hermitian"))
    def test_torch_backprop(self, style):
        """Test that backpropagation derivatives work with torch with hamiltonians and large sums."""
        import torch

        dev = DefaultQubit()

        x = torch.tensor(-0.289, requires_grad=True)
        x2 = torch.tensor(-0.289, requires_grad=True)
        out = self.f(dev, x, style=style)
        expected_out = self.expected(x2, like="torch")
        assert qml.math.allclose(out, expected_out)

        out.backward()  # pylint:disable=no-member
        expected_out.backward()
        assert qml.math.allclose(x.grad, x2.grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("style", ("sum", "hermitian"))
    def test_tf_backprop(self, style):
        """Test that backpropagation derivatives work with tensorflow with hamiltonians and large sums."""
        import tensorflow as tf

        dev = DefaultQubit()

        x = tf.Variable(0.5, dtype="float64")

        with tf.GradientTape() as tape1:
            out = self.f(dev, x, style=style)

        with tf.GradientTape() as tape2:
            expected_out = self.expected(x)

        assert qml.math.allclose(out, expected_out)
        g1 = tape1.gradient(out, x)
        g2 = tape2.gradient(expected_out, x)
        assert qml.math.allclose(g1, g2)


@pytest.mark.parametrize("max_workers", max_workers_list)
@pytest.mark.parametrize("config", (None, ExecutionConfig(gradient_method="adjoint")))
class TestAdjointDifferentiation:
    """Tests adjoint differentiation integration with DefaultQubit."""

    def test_derivatives_single_circuit(self, max_workers, config):
        """Tests derivatives with a single circuit."""
        dev = DefaultQubit(max_workers=max_workers)
        x = np.array(np.pi / 7)
        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])

        batch, _ = dev.preprocess_transforms(config)((qs,))
        qs = batch[0]
        expected_grad = -qml.math.sin(x)
        actual_grad = dev.compute_derivatives(qs, config)
        assert isinstance(actual_grad, np.ndarray)
        assert actual_grad.shape == ()  # pylint: disable=no-member
        assert np.isclose(actual_grad, expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_derivatives(qs, config)
        assert np.isclose(actual_val, expected_val)
        assert np.isclose(actual_grad, expected_grad)

    def test_derivatives_list_with_single_circuit(self, max_workers, config):
        """Tests a basic example with a batch containing a single circuit."""
        dev = DefaultQubit(max_workers=max_workers)
        x = np.array(np.pi / 7)
        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])

        batch, _ = dev.preprocess_transforms(config)((qs,))
        qs = batch[0]
        expected_grad = -qml.math.sin(x)
        actual_grad = dev.compute_derivatives([qs], config)
        assert isinstance(actual_grad, tuple)
        assert isinstance(actual_grad[0], np.ndarray)
        assert np.isclose(actual_grad[0], expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_derivatives([qs], config)
        assert np.isclose(expected_val, actual_val[0])
        assert np.isclose(expected_grad, actual_grad[0])

    def test_derivatives_many_tapes_many_results(self, max_workers, config):
        """Tests a basic example with a batch of circuits of varying return shapes."""
        dev = DefaultQubit(max_workers=max_workers)
        x = np.array(np.pi / 7)
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        expected_grad = (-qml.math.sin(x), (qml.math.cos(x), -qml.math.sin(x)))
        actual_grad = dev.compute_derivatives([single_meas, multi_meas], config)
        assert np.isclose(actual_grad[0], expected_grad[0])
        assert isinstance(actual_grad[1], tuple)
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

    def test_derivatives_integration(self, max_workers, config):
        """Tests the expected workflow done by a calling method."""
        dev = DefaultQubit(max_workers=max_workers)
        x = np.array(np.pi / 7)
        expected_grad = (-qml.math.sin(x), (qml.math.cos(x), -qml.math.sin(x)))
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )

        program, new_ec = dev.preprocess(config)
        circuits, _ = program([single_meas, multi_meas])
        actual_grad = dev.compute_derivatives(circuits, new_ec)

        if config and config.gradient_method == "adjoint":
            assert new_ec.use_device_gradient
            assert new_ec.grad_on_execution

        assert np.isclose(actual_grad[0], expected_grad[0])
        assert isinstance(actual_grad[1], tuple)
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

    def test_jvps_single_circuit(self, max_workers, config):
        """Tests jvps with a single circuit."""
        dev = DefaultQubit(max_workers=max_workers)
        x = np.array(np.pi / 7)
        tangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])

        batch, _ = dev.preprocess_transforms(config)((qs,))
        qs = batch[0]

        expected_grad = -qml.math.sin(x) * tangent[0]
        actual_grad = dev.compute_jvp(qs, tangent, config)
        assert isinstance(actual_grad, np.ndarray)
        assert actual_grad.shape == ()  # pylint: disable=no-member
        assert np.isclose(actual_grad, expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_jvp(qs, tangent, config)
        assert np.isclose(actual_val, expected_val)
        assert np.isclose(actual_grad, expected_grad)

    def test_jvps_list_with_single_circuit(self, max_workers, config):
        """Tests a basic example with a batch containing a single circuit."""
        dev = DefaultQubit(max_workers=max_workers)
        x = np.array(np.pi / 7)
        tangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])

        batch, _ = dev.preprocess_transforms(config)((qs,))
        qs = batch[0]

        expected_grad = -qml.math.sin(x) * tangent[0]
        actual_grad = dev.compute_jvp([qs], [tangent], config)
        assert isinstance(actual_grad, tuple)
        assert isinstance(actual_grad[0], np.ndarray)
        assert np.isclose(actual_grad[0], expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_jvp([qs], [tangent], config)
        assert np.isclose(expected_val, actual_val[0])
        assert np.isclose(expected_grad, actual_grad[0])

    def test_jvps_many_tapes_many_results(self, max_workers, config):
        """Tests a basic example with a batch of circuits of varying return shapes."""
        dev = DefaultQubit(max_workers=max_workers)
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
        actual_grad = dev.compute_jvp([single_meas, multi_meas], tangents, config)
        assert np.isclose(actual_grad[0], expected_grad[0])
        assert isinstance(actual_grad[1], tuple)
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

        expected_val = (qml.math.cos(x), (qml.math.sin(x), qml.math.cos(x)))
        actual_val, actual_grad = dev.execute_and_compute_jvp(
            [single_meas, multi_meas], tangents, config
        )
        assert np.isclose(actual_val[0], expected_val[0])
        assert qml.math.allclose(actual_val[1], expected_val[1])
        assert np.isclose(actual_grad[0], expected_grad[0])
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

    def test_jvps_integration(self, max_workers, config):
        """Tests the expected workflow done by a calling method."""
        dev = DefaultQubit(max_workers=max_workers)
        x = np.array(np.pi / 7)

        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        tangents = [(0.456,), (0.789,)]
        circuits = [single_meas, multi_meas]
        program, new_ec = dev.preprocess(config)
        circuits, _ = program(circuits)
        actual_grad = dev.compute_jvp(circuits, tangents, new_ec)
        expected_grad = (
            -qml.math.sin(x) * tangents[0][0],
            (qml.math.cos(x) * tangents[1][0], -qml.math.sin(x) * tangents[1][0]),
        )

        if config and config.gradient_method == "adjoint":
            assert new_ec.use_device_gradient
            assert new_ec.grad_on_execution

        assert np.isclose(actual_grad[0], expected_grad[0])
        assert isinstance(actual_grad[1], tuple)
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

    def test_vjps_single_circuit(self, max_workers, config):
        """Tests vjps with a single circuit."""
        dev = DefaultQubit(max_workers=max_workers)
        x = np.array(np.pi / 7)
        cotangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        batch, _ = dev.preprocess_transforms(config)((qs,))
        qs = batch[0]

        expected_grad = -qml.math.sin(x) * cotangent[0]
        actual_grad = dev.compute_vjp(qs, cotangent, config)
        assert np.isclose(actual_grad, expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_vjp(qs, cotangent, config)
        assert np.isclose(actual_val, expected_val)
        assert np.isclose(actual_grad, expected_grad)

    def test_vjps_list_with_single_circuit(self, max_workers, config):
        """Tests a basic example with a batch containing a single circuit."""
        dev = DefaultQubit(max_workers=max_workers)
        x = np.array(np.pi / 7)
        cotangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        batch, _ = dev.preprocess_transforms(config)((qs,))
        qs = batch[0]

        expected_grad = -qml.math.sin(x) * cotangent[0]
        actual_grad = dev.compute_vjp([qs], [cotangent], config)
        assert isinstance(actual_grad, tuple)
        assert np.isclose(actual_grad[0], expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_vjp([qs], [cotangent], config)
        assert np.isclose(expected_val, actual_val[0])
        assert np.isclose(expected_grad, actual_grad[0])

    def test_vjps_many_tapes_many_results(self, max_workers, config):
        """Tests a basic example with a batch of circuits of varying return shapes."""
        dev = DefaultQubit(max_workers=max_workers)
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
        actual_grad = dev.compute_vjp([single_meas, multi_meas], cotangents, config)
        assert np.isclose(actual_grad[0], expected_grad[0])
        assert np.isclose(actual_grad[1], expected_grad[1])

        expected_val = (qml.math.cos(x), (qml.math.sin(x), qml.math.cos(x)))
        actual_val, actual_grad = dev.execute_and_compute_vjp(
            [single_meas, multi_meas], cotangents, config
        )
        assert np.isclose(actual_val[0], expected_val[0])
        assert qml.math.allclose(actual_val[1], expected_val[1])
        assert np.isclose(actual_grad[0], expected_grad[0])
        assert np.isclose(actual_grad[1], expected_grad[1])

    def test_vjps_integration(self, max_workers, config):
        """Tests the expected workflow done by a calling method."""
        dev = DefaultQubit(max_workers=max_workers)
        x = np.array(np.pi / 7)

        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        cotangents = [(0.456,), (0.789, 0.123)]
        circuits = [single_meas, multi_meas]
        program, new_ec = dev.preprocess(config)
        circuits, _ = program(circuits)

        actual_grad = dev.compute_vjp(circuits, cotangents, new_ec)
        expected_grad = (
            -qml.math.sin(x) * cotangents[0][0],
            qml.math.cos(x) * cotangents[1][0] - qml.math.sin(x) * cotangents[1][1],
        )

        if config and config.gradient_method == "adjoint":
            assert new_ec.use_device_gradient
            assert new_ec.grad_on_execution

        assert np.isclose(actual_grad[0], expected_grad[0])
        assert np.isclose(actual_grad[1], expected_grad[1])


class TestRandomSeed:
    """Test that the device behaves correctly when provided with a random seed"""

    @pytest.mark.parametrize("max_workers", max_workers_list)
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

        dev1 = DefaultQubit(seed=123, max_workers=max_workers)
        result1 = dev1.execute(qs)

        dev2 = DefaultQubit(seed=123, max_workers=max_workers)
        result2 = dev2.execute(qs)

        if len(measurements) == 1:
            result1, result2 = [result1], [result2]

        assert all(np.all(res1 == res2) for res1, res2 in zip(result1, result2))

    @pytest.mark.slow
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_different_seed(self, max_workers):
        """Test that different devices given different random seeds will produce
        different results (with almost certainty)"""
        qs = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.sample(wires=0)], shots=1000)

        dev1 = DefaultQubit(seed=None, max_workers=max_workers)
        result1 = dev1.execute(qs)

        dev2 = DefaultQubit(seed=123, max_workers=max_workers)
        result2 = dev2.execute(qs)

        dev3 = DefaultQubit(seed=456, max_workers=max_workers)
        result3 = dev3.execute(qs)

        # assert results are pairwise different
        assert np.any(result1 != result2)
        assert np.any(result1 != result3)
        assert np.any(result2 != result3)

    @pytest.mark.parametrize("max_workers", max_workers_list)
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

        dev = DefaultQubit(seed=123, max_workers=max_workers)
        result1 = dev.execute(qs)
        result2 = dev.execute(qs)

        if len(measurements) == 1:
            result1, result2 = [result1], [result2]

        assert all(np.any(res1 != res2) for res1, res2 in zip(result1, result2))

    @pytest.mark.parametrize("max_workers", max_workers_list)
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
        dev1 = DefaultQubit(seed=123, max_workers=max_workers)
        result1 = dev1.execute(qs)

        # set a global seed both before initialization of the
        # device and before execution of the tape
        np.random.seed(456)
        dev2 = DefaultQubit(seed=123, max_workers=max_workers)
        np.random.seed(789)
        result2 = dev2.execute(qs)

        if len(measurements) == 1:
            result1, result2 = [result1], [result2]

        assert all(np.all(res1 == res2) for res1, res2 in zip(result1, result2))

    def test_global_seed_no_device_seed_by_default(self):
        """Test that the global numpy seed initializes the rng if device seed is none."""
        np.random.seed(42)
        dev = DefaultQubit()
        first_num = dev._rng.random()  # pylint: disable=protected-access

        np.random.seed(42)
        dev2 = DefaultQubit()
        second_num = dev2._rng.random()  # pylint: disable=protected-access

        assert qml.math.allclose(first_num, second_num)

        np.random.seed(42)
        dev2 = DefaultQubit(seed="global")
        third_num = dev2._rng.random()  # pylint: disable=protected-access

        assert qml.math.allclose(third_num, first_num)

    def test_None_seed_not_using_global_rng(self):
        """Test that if the seed is None, it is uncorrelated with the global rng."""
        np.random.seed(42)
        dev = DefaultQubit(seed=None)
        first_nums = dev._rng.random(10)  # pylint: disable=protected-access

        np.random.seed(42)
        dev2 = DefaultQubit(seed=None)
        second_nums = dev2._rng.random(10)  # pylint: disable=protected-access

        assert not qml.math.allclose(first_nums, second_nums)

    def test_rng_as_seed(self):
        """Test that a PRNG can be passed as a seed."""
        rng1 = np.random.default_rng(42)
        first_num = rng1.random()

        rng = np.random.default_rng(42)
        dev = DefaultQubit(seed=rng)
        second_num = dev._rng.random()  # pylint: disable=protected-access

        assert qml.math.allclose(first_num, second_num)


@pytest.mark.jax
class TestPRNGKeySeed:
    """Test that the device behaves correctly when provided with a PRNG key and using the JAX interface"""

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_same_device_prng_key(self, max_workers):
        """Test a device with a given jax.random.PRNGKey will produce
        the same samples repeatedly."""
        import jax

        qs = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.sample(wires=0)], shots=1000)
        config = ExecutionConfig(interface="jax")

        dev = DefaultQubit(max_workers=max_workers, seed=jax.random.PRNGKey(123))
        result1 = dev.execute(qs, config)
        for _ in range(10):
            result2 = dev.execute(qs, config)

            assert np.all(result1 == result2)

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_prng_key_multi_tapes(self, max_workers):
        """Test a device with a given jax.random.PRNGKey will produce
        different results for the same (batched) tape."""
        import jax

        qs = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.sample(wires=0)], shots=1000)
        config = ExecutionConfig(interface="jax")

        dev = DefaultQubit(max_workers=max_workers, seed=jax.random.PRNGKey(123))
        result1, result2 = dev.execute([qs] * 2, config)

        assert not np.all(result1 == result2)

    @pytest.mark.xfail  # [sc-90338]
    def test_different_max_workers_same_prng_key(self):
        """Test that devices with the same jax.random.PRNGKey but different threading will produce
        the same samples."""
        import jax

        qs = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.sample(wires=0)], shots=1000)
        config = ExecutionConfig(interface="jax")

        dev1 = DefaultQubit(max_workers=None, seed=jax.random.PRNGKey(123))
        result1 = dev1.execute([qs] * 10, config)
        for max_workers in range(1, 3):
            dev2 = DefaultQubit(max_workers=max_workers, seed=jax.random.PRNGKey(123))
            result2 = dev2.execute([qs] * 10, config)

            assert len(result1) == len(result2)
            for r1, r2 in zip(result1, result2):
                assert np.all(r1 == r2)

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_same_prng_key(self, max_workers):
        """Test that different devices given the same random jax.random.PRNGKey as a seed will produce
        the same results for sample, even with different seeds"""
        import jax

        qs = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.sample(wires=0)], shots=1000)
        config = ExecutionConfig(interface="jax")

        dev1 = DefaultQubit(max_workers=max_workers, seed=jax.random.PRNGKey(123))
        result1 = dev1.execute(qs, config)

        dev2 = DefaultQubit(max_workers=max_workers, seed=jax.random.PRNGKey(123))
        result2 = dev2.execute(qs, config)

        assert np.all(result1 == result2)

    def test_get_prng_keys(self):
        """Test the get_prng_keys method."""
        import jax

        dev = DefaultQubit(seed=jax.random.PRNGKey(123))

        assert len(dev.get_prng_keys()) == 1
        assert len(dev.get_prng_keys(num=1)) == 1
        assert len(dev.get_prng_keys(num=2)) == 2

        with pytest.raises(ValueError):
            dev.get_prng_keys(num=0)

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_different_prng_key(self, max_workers):
        """Test that different devices given different jax.random.PRNGKey values will produce
        different results"""
        import jax

        qs = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.sample(wires=0)], shots=1000)
        config = ExecutionConfig(interface="jax")

        dev1 = DefaultQubit(max_workers=max_workers, seed=jax.random.PRNGKey(246))
        result1 = dev1.execute(qs, config)

        dev2 = DefaultQubit(max_workers=max_workers, seed=jax.random.PRNGKey(123))
        result2 = dev2.execute(qs, config)

        assert np.any(result1 != result2)

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_different_executions_same_prng_key(self, max_workers):
        """Test that the same device will produce the same results every execution if
        the seed is a jax.random.PRNGKey"""
        import jax

        qs = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.sample(wires=0)], shots=1000)
        config = ExecutionConfig(interface="jax")

        dev = DefaultQubit(max_workers=max_workers, seed=jax.random.PRNGKey(77))
        result1 = dev.execute(qs, config)
        result2 = dev.execute(qs, config)

        assert np.all(result1 == result2)

    # @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_finite_shots_postselection_defer_measurements(self):
        """Test that the number of shots returned with postselection with a PRNGKey is different
        when executing a batch of tapes and the same when using `dev.execute` with the same tape
        multiple times."""
        import jax

        dev = qml.device("default.qubit", seed=jax.random.PRNGKey(234))

        mv = qml.measure(0, postselect=1)
        qs = qml.tape.QuantumScript(
            [qml.Hadamard(0), mv.measurements[0]], [qml.sample(wires=0)], shots=500
        )
        n_tapes = 5
        tapes = qml.defer_measurements(qs)[0] * 5
        config = ExecutionConfig(interface="jax")

        # Executing all tapes as a batch should give different results
        res = dev.execute(tapes, config)
        shapes = [qml.math.shape(r) for r in res]
        assert len(set(shapes)) == len(shapes) == n_tapes

        # Executing with different calls to dev.execute should give the same results
        res = [dev.execute(tape, config) for tape in tapes]
        shapes = [qml.math.shape(r) for r in res]
        assert len(shapes) == n_tapes
        assert len(set(shapes)) == 1
        # The following iterator validates that the samples for each tape are the same
        iterator = iter(res)
        first = next(iterator)
        assert all(np.array_equal(first, rest) for rest in iterator)

    def test_integrate_prng_key_jitting(self):
        """Test that a PRNGKey can be used with a jitted qnode."""

        import jax

        @jax.jit
        def workflow(key, param):

            dev = qml.device("default.qubit", seed=key)

            @qml.set_shots(100)
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(x, 0)
                return qml.sample(wires=0)

            return circuit(param)

        key1 = jax.random.PRNGKey(123456)
        key2 = jax.random.PRNGKey(8877655)
        x = jax.numpy.array(0.5)
        # no leaked tracer errors
        res1 = workflow(key1, x)
        res1_again = workflow(key1, x)
        res2 = workflow(key2, x)

        assert qml.math.allclose(res1, res1_again)
        assert not qml.math.allclose(res1, res2)


class TestHamiltonianSamples:
    """Test that the measure_with_samples function works as expected for
    Hamiltonian and Sum observables

    This is a copy of the tests in test_sampling.py, but using the device instead"""

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_hamiltonian_expval(self, max_workers, seed):
        """Test that sampling works well for Hamiltonian observables"""
        x, y = np.array(0.67), np.array(0.95)
        ops = [qml.RY(x, wires=0), qml.RZ(y, wires=0)]
        meas = [qml.expval(qml.Hamiltonian([0.8, 0.5], [qml.PauliZ(0), qml.PauliX(0)]))]

        dev = DefaultQubit(seed=seed, max_workers=max_workers)
        qs = qml.tape.QuantumScript(ops, meas, shots=10000)
        res = dev.execute(qs)

        expected = 0.8 * np.cos(x) + 0.5 * np.real(np.exp(y * 1j)) * np.sin(x)
        assert np.allclose(res, expected, atol=0.02)

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_sum_expval(self, max_workers, seed):
        """Test that sampling works well for Sum observables"""
        x, y = np.array(0.67), np.array(0.95)
        ops = [qml.RY(x, wires=0), qml.RZ(y, wires=0)]
        meas = [qml.expval(qml.s_prod(0.8, qml.PauliZ(0)) + qml.s_prod(0.5, qml.PauliX(0)))]

        dev = DefaultQubit(seed=seed, max_workers=max_workers)
        qs = qml.tape.QuantumScript(ops, meas, shots=10000)
        res = dev.execute(qs)

        expected = 0.8 * np.cos(x) + 0.5 * np.real(np.exp(y * 1j)) * np.sin(x)
        assert np.allclose(res, expected, atol=0.02)

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_multi_wires(self, max_workers, seed):
        """Test that sampling works for Sums with large numbers of wires"""
        n_wires = 10
        scale = 0.05
        offset = 0.8

        ops = [qml.RX(offset + scale * i, wires=i) for i in range(n_wires)]

        t1 = 2.5 * qml.prod(*(qml.PauliZ(i) for i in range(n_wires)))
        t2 = 6.2 * qml.prod(*(qml.PauliY(i) for i in range(n_wires)))
        H = t1 + t2

        dev = DefaultQubit(seed=seed, max_workers=max_workers)
        qs = qml.tape.QuantumScript(ops, [qml.expval(H)], shots=30000)
        res = dev.execute(qs)

        phase = offset + scale * np.array(range(n_wires))
        cosines = qml.math.cos(phase)
        sines = qml.math.sin(phase)
        expected = 2.5 * qml.math.prod(cosines) + 6.2 * qml.math.prod(sines)

        assert np.allclose(res, expected, rtol=0.05)

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_complex_hamiltonian(self, max_workers, seed):
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

        dev = DefaultQubit(seed=seed, max_workers=max_workers)
        qs = qml.tape.QuantumScript(ops, [qml.expval(H)], shots=50000)
        res = dev.execute(qs)

        qs_exp = qml.tape.QuantumScript(ops, [qml.expval(H)])
        expected = dev.execute(qs_exp)

        assert np.allclose(res, expected, atol=0.02)


class TestClassicalShadows:
    """Test that classical shadow measurements works with the new device"""

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_shape_and_dtype(self, max_workers, n_qubits):
        """Test that the shape and dtype of the measurement is correct"""
        dev = DefaultQubit(max_workers=max_workers)

        ops = [qml.Hadamard(i) for i in range(n_qubits)]
        qs = qml.tape.QuantumScript(ops, [qml.classical_shadow(range(n_qubits))], shots=100)
        res = dev.execute(qs)

        assert res.shape == (2, 100, n_qubits)
        assert res.dtype == np.int8

        # test that the bits are either 0 and 1
        assert np.all(np.logical_or(res[0] == 0, res[0] == 1))

        # test that the recipes are either 0, 1, or 2 (X, Y, or Z)
        assert np.all(np.logical_or(np.logical_or(res[1] == 0, res[1] == 1), res[1] == 2))

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_expval(self, max_workers, seed):
        """Test that shadow expval measurements work as expected"""

        dev = DefaultQubit(seed=seed, max_workers=max_workers)

        ops = [qml.Hadamard(0), qml.Hadamard(1)]
        meas = [qml.shadow_expval(qml.PauliX(0) @ qml.PauliX(1), seed=seed)]
        qs = qml.tape.QuantumScript(ops, meas, shots=10000)
        res = dev.execute(qs)

        assert res.shape == ()
        assert np.allclose(res, 1.0, rtol=0.05)

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_multiple_shadow_measurements(self, n_qubits, max_workers):
        """Test that multiple classical shadow measurements work as expected"""
        dev = DefaultQubit(max_workers=max_workers)

        ops = [qml.Hadamard(i) for i in range(n_qubits)]
        mps = [qml.classical_shadow(range(n_qubits)), qml.classical_shadow(range(n_qubits))]
        qs = qml.tape.QuantumScript(ops, mps, shots=100)
        res = dev.execute(qs)

        assert isinstance(res, tuple)
        assert len(res) == 2

        for r in res:
            assert r.shape == (2, 100, n_qubits)
            assert r.dtype == np.int8

            # test that the bits are either 0 and 1
            assert np.all(np.logical_or(r[0] == 0, r[0] == 1))

            # test that the recipes are either 0, 1, or 2 (X, Y, or Z)
            assert np.all(np.logical_or(np.logical_or(r[1] == 0, r[1] == 1), r[1] == 2))

        # check that the samples are different
        assert not np.all(res[0] == res[1])

    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_reconstruct_bell_state(self, max_workers, seed):
        """Test that a bell state can be faithfully reconstructed"""
        dev = DefaultQubit(seed=seed, max_workers=max_workers)

        ops = [qml.Hadamard(0), qml.CNOT([0, 1])]
        meas = [qml.classical_shadow(wires=[0, 1], seed=seed)]
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
        meas = [qml.classical_shadow(wires=[0], seed=seed)]
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
    @pytest.mark.parametrize("max_workers", max_workers_list)
    def test_shot_vectors(self, max_workers, n_qubits, shots):
        """Test that classical shadows works when given a shot vector"""
        dev = DefaultQubit(max_workers=max_workers)
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


@pytest.mark.parametrize("n_wires", [1, 2, 3])
@pytest.mark.parametrize("max_workers", max_workers_list)
def test_projector_dynamic_type(max_workers, n_wires):
    """Test that qml.Projector yields the expected results for both of its subclasses."""
    wires = list(range(n_wires))
    dev = DefaultQubit(max_workers=max_workers)
    ops = [qml.adjoint(qml.Hadamard(q)) for q in wires]
    basis_state = np.zeros((n_wires,))
    state_vector = np.zeros((2**n_wires,))
    state_vector[0] = 1

    for state in [basis_state, state_vector]:
        qs = qml.tape.QuantumScript(ops, [qml.expval(qml.Projector(state, wires))])
        res = dev.execute(qs)
        assert np.isclose(res, 1 / 2**n_wires)


@pytest.mark.integration
@pytest.mark.parametrize(
    "interface",
    [
        "numpy",
        pytest.param("autograd", marks=pytest.mark.autograd),
        pytest.param("torch", marks=pytest.mark.torch),
        pytest.param("jax", marks=pytest.mark.jax),
        pytest.param("tensorflow", marks=pytest.mark.tf),
    ],
)
@pytest.mark.parametrize("use_jit", [True, False])
class TestPostselection:
    """Various integration tests for postselection of mid-circuit measurements."""

    @pytest.mark.parametrize(
        "mp",
        [
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(0)),
            qml.probs(wires=[0, 1]),
            qml.density_matrix(wires=0),
            qml.purity(0),
            qml.vn_entropy(0),
            qml.mutual_info(0, 1),
        ],
    )
    @pytest.mark.parametrize("param", np.linspace(np.pi / 4, 3 * np.pi / 4, 3))
    def test_postselection_valid_analytic(self, param, mp, interface, use_jit):
        """Test that the results of a circuit with postselection is expected
        with analytic execution."""
        if use_jit and interface != "jax":
            pytest.skip("Cannot JIT in non-JAX interfaces.")

        dev = qml.device("default.qubit")
        param = qml.math.asarray(param, like=interface)

        @qml.defer_measurements
        @qml.qnode(dev, interface=interface)
        def circ_postselect(theta):
            qml.RX(theta, 0)
            qml.CNOT([0, 1])
            qml.measure(0, postselect=1)
            return qml.apply(mp)

        @qml.defer_measurements
        @qml.qnode(dev, interface=interface)
        def circ_expected():
            qml.RX(np.pi, 0)
            qml.CNOT([0, 1])
            return qml.apply(mp)

        if use_jit:
            import jax

            circ_postselect = jax.jit(circ_postselect)

        res = circ_postselect(param)
        expected = circ_expected()

        assert qml.math.allclose(res, expected)
        assert qml.math.get_interface(res) == qml.math.get_interface(expected)

    @pytest.mark.parametrize(
        "mp",
        [
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(0)),
            qml.probs(wires=[0, 1]),
            qml.shadow_expval(qml.Hamiltonian([1.0, -1.0], [qml.PauliZ(0), qml.PauliX(0)])),
            # qml.sample, qml.classical_shadow, qml.counts are not included because their
            # shape/values are dependent on the number of shots, which will be changed
            # randomly per the binomial distribution and the probability of the postselected
            # state
        ],
    )
    @pytest.mark.parametrize("param", np.linspace(np.pi / 4, 3 * np.pi / 4, 3))
    @pytest.mark.parametrize("shots", [50000, (50000, 50000)])
    def test_postselection_valid_finite_shots(self, param, mp, shots, interface, use_jit, seed):
        """Test that the results of a circuit with postselection is expected with
        finite shots."""
        if use_jit and (interface != "jax" or isinstance(shots, tuple)):
            pytest.skip("Cannot JIT in non-JAX interfaces, or with shot vectors.")

        if isinstance(mp, qml.measurements.ClassicalShadowMP):
            mp.seed = seed

        dev = qml.device("default.qubit", seed=seed)
        param = qml.math.asarray(param, like=interface)

        @qml.set_shots(shots=shots)
        @qml.defer_measurements
        @qml.qnode(dev, interface=interface)
        def circ_postselect(theta):
            qml.RX(theta, 0)
            qml.CNOT([0, 1])
            qml.measure(0, postselect=1)
            return qml.apply(mp)

        @qml.set_shots(shots=shots)
        @qml.defer_measurements
        @qml.qnode(dev, interface=interface)
        def circ_expected():
            qml.RX(np.pi, 0)
            qml.CNOT([0, 1])
            return qml.apply(mp)

        if use_jit:
            import jax

            circ_postselect = jax.jit(circ_postselect)

        res = circ_postselect(param)
        expected = circ_expected()

        if not isinstance(shots, tuple):
            assert qml.math.allclose(res, expected, atol=0.1, rtol=0)
            assert qml.math.get_interface(res) == qml.math.get_interface(expected)

        else:
            assert isinstance(res, tuple)
            for r, e in zip(res, expected):
                assert qml.math.allclose(r, e, atol=0.1, rtol=0)
                assert qml.math.get_interface(r) == qml.math.get_interface(e)

    @pytest.mark.parametrize(
        "mp, expected_shape",
        [
            (qml.sample(wires=[0, 2]), (5, 2)),
            (qml.classical_shadow(wires=[0, 2]), (2, 5, 2)),
            (qml.sample(wires=[0]), (5, 1)),
            (qml.classical_shadow(wires=[0]), (2, 5, 1)),
        ],
    )
    @pytest.mark.parametrize("param", np.linspace(np.pi / 4, 3 * np.pi / 4, 3))
    @pytest.mark.parametrize("shots", [10, (10, 10)])
    def test_postselection_valid_finite_shots_varied_shape(
        self, mp, param, expected_shape, shots, interface, use_jit
    ):
        """Test that qml.sample and qml.classical_shadow work correctly.
        Separate test because their shape is non-deterministic."""

        if use_jit:
            pytest.skip("Cannot JIT while mocking function.")

        # Setting the device RNG to None forces the functions to use the global Numpy random
        # module rather than the functions directly exposed by a local RNG. This makes
        # mocking easier.
        dev = qml.device("default.qubit")
        dev._rng = None
        param = qml.math.asarray(param, like=interface)

        with mock.patch("numpy.random.binomial", lambda *args, **kwargs: 5):

            @qml.set_shots(shots=shots)
            @qml.defer_measurements
            @qml.qnode(dev, interface=interface)
            def circ_postselect(theta):
                qml.RX(theta, 0)
                qml.CNOT([0, 1])
                qml.measure(0, postselect=1)
                return qml.apply(mp)

            res = circ_postselect(param)

        if not isinstance(shots, tuple):
            assert qml.math.get_interface(res) == interface if interface != "autograd" else "numpy"
            assert qml.math.shape(res) == expected_shape

        else:
            assert isinstance(res, tuple)
            for r in res:
                assert (
                    qml.math.get_interface(r) == interface if interface != "autograd" else "numpy"
                )
                assert qml.math.shape(r) == expected_shape

    @pytest.mark.parametrize(
        "mp, autograd_interface, is_nan",
        [
            (qml.expval(qml.PauliZ(0)), "autograd", True),
            (qml.var(qml.PauliZ(0)), "autograd", True),
            (qml.probs(wires=[0, 1]), "autograd", True),
            (qml.density_matrix(wires=0), "autograd", True),
            (qml.purity(0), "numpy", True),
            (qml.vn_entropy(0), "numpy", False),
            (qml.mutual_info(0, 1), "numpy", False),
        ],
    )
    def test_postselection_invalid_analytic(
        self, mp, autograd_interface, is_nan, interface, use_jit
    ):
        """Test that the results of a qnode are nan values of the correct shape if the state
        that we are postselecting has a zero probability of occurring."""

        if (isinstance(mp, qml.measurements.MutualInfoMP) and interface != "jax") or (
            isinstance(mp, qml.measurements.VnEntropyMP) and interface == "tensorflow"
        ):
            pytest.skip("Unsupported measurements and interfaces.")

        if use_jit:
            pytest.skip("Jitting tested in different test.")

        # Wires are specified so that the shape for measurements can be determined correctly
        dev = qml.device("default.qubit")

        @qml.defer_measurements
        @qml.qnode(dev, interface=None if interface == "numpy" else interface)
        def circ():
            qml.RX(np.pi, 0)
            qml.CNOT([0, 1])
            qml.measure(0, postselect=0)
            return qml.apply(mp)

        res = circ()
        if interface == "autograd":
            assert qml.math.get_interface(res) == autograd_interface
        else:
            assert qml.math.get_interface(res) == interface

        assert qml.math.shape(res) == mp.shape(dev, qml.measurements.Shots(None))
        if is_nan:
            assert qml.math.all(qml.math.isnan(res))
        else:
            assert qml.math.allclose(res, 0.0)

    @pytest.mark.parametrize(
        "mp, expected",
        [
            (qml.expval(qml.PauliZ(0)), 1.0),
            (qml.var(qml.PauliZ(0)), 0.0),
            (qml.probs(wires=[0, 1]), [1.0, 0.0, 0.0, 0.0]),
            (qml.density_matrix(wires=0), [[1.0, 0.0], [0.0, 0.0]]),
            (qml.purity(0), 1.0),
            (qml.vn_entropy(0), 0.0),
            (qml.mutual_info(0, 1), 0.0),
        ],
    )
    def test_postselection_invalid_analytic_jit(self, mp, expected, interface, use_jit):
        """Test that the results of a qnode give the postselected results even when the
        probability of the postselected state is zero when jitting."""
        if interface != "jax" or not use_jit:
            pytest.skip("Test is only for jitting.")

        import jax

        # Wires are specified so that the shape for measurements can be determined correctly
        dev = qml.device("default.qubit")

        @jax.jit
        @qml.defer_measurements
        @qml.qnode(dev, interface=interface)
        def circ():
            qml.RX(np.pi, 0)
            qml.CNOT([0, 1])
            qml.measure(0, postselect=0)
            return qml.apply(mp)

        res = circ()

        assert qml.math.get_interface(res) == "jax"
        assert qml.math.shape(res) == mp.shape(dev, qml.measurements.Shots(None))
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize(
        "mp, expected_shape",
        [
            (qml.expval(qml.PauliZ(0)), ()),
            (qml.var(qml.PauliZ(0)), ()),
            (qml.sample(qml.PauliZ(0)), (0,)),
            (qml.classical_shadow(wires=0), (2, 0, 1)),
            (qml.shadow_expval(qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliX(0)])), ()),
            # qml.probs and qml.counts are not tested because they fail in this case
        ],
    )
    @pytest.mark.parametrize("shots", [10, (10, 10)])
    def test_postselection_invalid_finite_shots(
        self, mp, expected_shape, shots, interface, use_jit
    ):
        """Test that the results of a qnode are nan values of the correct shape if the state
        that we are postselecting has a zero probability of occurring with finite shots."""

        if use_jit and interface != "jax":
            pytest.skip("Can't jit with non-jax interfaces.")

        dev = qml.device("default.qubit")

        @qml.set_shots(shots=shots)
        @qml.defer_measurements
        @qml.qnode(dev, interface=interface)
        def circ():
            qml.RX(np.pi, 0)
            qml.CNOT([0, 1])
            qml.measure(0, postselect=0)
            return qml.apply(mp)

        if use_jit:
            pytest.xfail(
                reason="defer measurements + hw-like does not work with JAX jit yet. See sc-96593 or #7981."
            )
            # import jax
            # circ = jax.jit(circ)

        res = circ()

        if not isinstance(shots, tuple):
            assert qml.math.shape(res) == expected_shape
            assert qml.math.get_interface(res) == interface if interface != "autograd" else "numpy"
            if not 0 in expected_shape:  # No nan values if array is empty
                assert qml.math.all(qml.math.isnan(res))
        else:
            assert isinstance(res, tuple)
            for r in res:
                assert qml.math.shape(r) == expected_shape
                assert (
                    qml.math.get_interface(r) == interface if interface != "autograd" else "numpy"
                )
                if not 0 in expected_shape:  # No nan values if array is empty
                    assert qml.math.all(qml.math.isnan(r))

    @pytest.mark.parametrize(
        "shots, postselect_mode, error",
        [
            (10, "fill-shots", True),
            (None, "fill-shots", False),
            (10, "hw-like", False),
            (None, "hw-like", False),
        ],
    )
    def test_defer_measurements_fill_shots_zero_prob_postselection_error(
        self, shots, postselect_mode, error, interface, use_jit
    ):
        """Test that an error is raised if `postselect_mode="fill-shots"` with finite shots
        and the postselection probability is zero when using defer_measurements."""
        if use_jit and interface != "jax":
            pytest.skip("Can't jit with non-jax interfaces.")

        dev = DefaultQubit()

        @qml.qnode(
            dev,
            shots=shots,
            interface=interface,
            mcm_method="deferred",
            postselect_mode=postselect_mode,
        )
        def circuit(x):
            # Applying a parametrized gate to make the state abstract with jax.jit
            qml.RZ(x, 0)
            # State is g * |0> for some global phase g (because we applied an RZ gate),
            # so postselection probability is zero
            qml.measure(0, postselect=1)
            return qml.expval(qml.Z(0))

        if use_jit:
            if postselect_mode == "hw-like":
                pytest.xfail(
                    reason="defer measurements + hw-like does not work with JAX jit yet. See sc-96593 or #7981."
                )

            # pylint: disable=import-outside-toplevel
            import jax

            # We do not raise an error if using jax.jit, because we cannot check whether or not
            # the probability is zero. But, this is only the case with analytic execution because
            # with shots, we perform the execution in a pure callback, so the state is concrete.
            error = error if shots else False
            circuit = jax.jit(circuit)

            # When jitting, we go through JAX's error handling, so the expected error is not the same
            # as without jitting
            expected_error = Exception
            err_message = ""

        else:
            expected_error = RuntimeError
            err_message = "The probability of the postselected"

        if error:
            with pytest.raises(expected_error, match=err_message):
                circuit(0.0)

        else:
            # no error
            circuit(0.0)


class TestIntegration:
    """Various integration tests"""

    @pytest.mark.parametrize("wires,expected", [(None, [1, 0]), (3, [0, 0, 1])])
    def test_sample_uses_device_wires(self, wires, expected):
        """Test that if device wires are given, then they are used by sample."""
        dev = DefaultQubit(wires=wires)

        @qml.qnode(dev, shots=5)
        def circuit():
            qml.PauliX(2)
            qml.Identity(0)
            return qml.sample()

        assert np.array_equal(circuit(), [expected] * 5)

    @pytest.mark.parametrize(
        "wires,expected",
        [
            (None, [0, 0, 1, 0]),
            (3, [0, 1] + [0] * 6),
        ],
    )
    def test_state_uses_device_wires(self, wires, expected):
        """Test that if device wires are given, then they are used by state."""
        dev = DefaultQubit(wires=wires)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(2)
            qml.Identity(0)
            return qml.state()

        assert np.array_equal(circuit(), expected)

    @pytest.mark.parametrize(
        "wires,expected",
        [
            (None, [0, 0, 1, 0]),
            (3, [0, 1] + [0] * 6),
        ],
    )
    def test_probs_uses_device_wires(self, wires, expected):
        """Test that if device wires are given, then they are used by probs."""
        dev = DefaultQubit(wires=wires)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(2)
            qml.Identity(0)
            return qml.probs()

        assert np.array_equal(circuit(), expected)

    @pytest.mark.parametrize(
        "wires,all_outcomes,expected",
        [
            (None, False, {"10": 10}),
            (None, True, {"10": 10, "00": 0, "01": 0, "11": 0}),
            (3, False, {"001": 10}),
            (
                3,
                True,
                {"001": 10, "000": 0, "010": 0, "011": 0, "100": 0, "101": 0, "110": 0, "111": 0},
            ),
        ],
    )
    def test_counts_uses_device_wires(self, wires, all_outcomes, expected):
        """Test that if device wires are given, then they are used by probs."""
        dev = DefaultQubit(wires=wires)

        @qml.qnode(dev, interface=None, shots=10)
        def circuit():
            qml.PauliX(2)
            qml.Identity(0)
            return qml.counts(all_outcomes=all_outcomes)

        assert circuit() == expected

    @pytest.mark.jax
    @pytest.mark.parametrize("measurement_func", [qml.expval, qml.var])
    def test_differentiate_jitted_qnode(self, measurement_func):
        """Test that a jitted qnode can be correctly differentiated"""
        import jax

        dev = DefaultQubit()

        def qfunc(x, y):
            qml.RX(x, 0)
            return measurement_func(qml.Hamiltonian(y, [qml.Z(0)]))

        qnode = qml.QNode(qfunc, dev, interface="jax")
        qnode_jit = jax.jit(qml.QNode(qfunc, dev, interface="jax"))

        x = jax.numpy.array(0.5)
        y = jax.numpy.array([0.5])

        res = qnode(x, y)
        res_jit = qnode_jit(x, y)

        assert qml.math.allclose(res, res_jit)

        grad = jax.grad(qnode)(x, y)
        grad_jit = jax.grad(qnode_jit)(x, y)

        assert qml.math.allclose(grad, grad_jit)

    def test_snapshot_with_defer_measurement(self):
        """Test that snapshots can be taken with defer_measurements."""

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            qml.measure(0)
            qml.Snapshot("label")
            return qml.probs(wires=0)

        snapshots = qml.snapshots(func)()
        assert snapshots["label"].shape == (4,)
        assert qml.math.allclose(snapshots["execution_results"], np.array([0.5, 0.5]))


@pytest.mark.parametrize("max_workers", max_workers_list)
def test_broadcasted_parameter(max_workers):
    """Test that DefaultQubit handles broadcasted parameters as expected."""
    dev = DefaultQubit(max_workers=max_workers)
    x = np.array([0.536, 0.894])
    qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])

    config = ExecutionConfig(gradient_method="adjoint")
    program, config = dev.preprocess(config)
    batch, pre_processing_fn = program([qs])
    assert len(batch) == 2
    results = dev.execute(batch, config)
    processed_results = pre_processing_fn(results)
    assert qml.math.allclose(processed_results, np.cos(x))


@pytest.mark.jax
def test_renomalization_issue():
    """Test that no normalization error occurs with the following workflow in float32 mode.
    Just tests executes without error.  Not producing a more minimal example due to difficulty
    finding an exact case that leads to renomalization issues.
    """
    import jax
    from jax import numpy as jnp

    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)

    def gaussian_fn(p, t):
        return p[0] * jnp.exp(-((t - p[1]) ** 2) / (2 * p[2] ** 2))

    global_drive = qml.pulse.rydberg_drive(
        amplitude=gaussian_fn, phase=0, detuning=0, wires=[0, 1, 2]
    )

    a = 5

    coordinates = [(0, 0), (a, 0), (a / 2, np.sqrt(a**2 - (a / 2) ** 2))]

    settings = {"interaction_coeff": 862619.7915580727}

    H_interaction = qml.pulse.rydberg_interaction(coordinates, **settings)

    max_amplitude = 2.0
    displacement = 1.0
    sigma = 0.3

    amplitude_params = [max_amplitude, displacement, sigma]

    params = [amplitude_params]
    ts = [0.0, 1.75]

    def circuit(params):
        qml.evolve(H_interaction + global_drive)(params, ts)
        return qml.counts()

    circuit_qml = qml.QNode(circuit, qml.device("default.qubit"), interface="jax", shots=1000)

    circuit_qml(params)
    jax.config.update("jax_enable_x64", initial_mode)
