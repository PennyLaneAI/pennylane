# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for null.qubit."""

from collections import defaultdict as dd

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig, NullQubit
from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.measurements import (
    ClassicalShadowMP,
    SampleMeasurement,
    ShadowExpvalMP,
    StateMeasurement,
)


def test_name():
    """Tests the name of NullQubit."""
    assert NullQubit().name == "null.qubit"


def test_shots():
    """Test the shots property of NullQubit."""
    assert NullQubit().shots == qml.measurements.Shots(None)
    with pytest.warns(PennyLaneDeprecationWarning, match="shots on device is deprecated"):
        assert NullQubit(shots=100).shots == qml.measurements.Shots(100)

    with pytest.raises(AttributeError):
        NullQubit().shots = 10


def test_wires():
    """Test that a device can be created with wires."""
    assert NullQubit().wires is None
    assert NullQubit(wires=2).wires == qml.wires.Wires([0, 1])
    assert NullQubit(wires=[0, 2]).wires == qml.wires.Wires([0, 2])

    with pytest.raises(AttributeError):
        NullQubit().wires = [0, 1]


def test_debugger_attribute():
    """Test that NullQubit has a debugger attribute and that it is `None`"""
    # pylint: disable=protected-access
    dev = NullQubit()

    assert hasattr(dev, "_debugger")
    assert dev._debugger is None


def test_resource_tracking_attributes():
    """Test NullQubit track_resources attribute"""
    default_dev = NullQubit()
    assert "track_resources" in default_dev.device_kwargs
    assert default_dev.device_kwargs["track_resources"] is False
    assert "resources_filename" not in default_dev.device_kwargs
    assert "compute_depth" not in default_dev.device_kwargs

    dev = NullQubit(track_resources=True, resources_filename="test.json", compute_depth=True)
    assert "track_resources" in dev.device_kwargs
    assert dev.device_kwargs["track_resources"] is True
    assert "resources_filename" in dev.device_kwargs
    assert dev.device_kwargs["resources_filename"] == "test.json"
    assert "compute_depth" in dev.device_kwargs
    assert dev.device_kwargs["compute_depth"] is True


def test_set_device_target():
    """Test that the device target can be set and retrieved correctly."""
    # pylint: disable=protected-access
    default_dev = NullQubit()
    assert default_dev._target_device is None
    assert default_dev.config_filepath is None

    # Pick something other than DefaultQubit, which is already the default
    to_target = qml.devices.ReferenceQubit()

    dev = NullQubit(target_device=to_target)
    assert dev._target_device == to_target
    assert dev.config_filepath == to_target.config_filepath

    program1, _ = dev.preprocess(ExecutionConfig())
    program2, _ = to_target.preprocess(ExecutionConfig())

    # Check that the preprocess function mimics the given target

    assert len(program1) == len(program2)
    for t1, t2 in zip(program1, program2):
        assert t1.transform == t2.transform

        assert len(t1.args) == len(t2.args)
        for i, arg in enumerate(t1.args):
            if not callable(arg):
                assert arg == t2.args[i]

        assert len(t1.kwargs) == len(t2.kwargs)
        for k in t1.kwargs:
            assert k in t2.kwargs
            if not callable(t1.kwargs[k]):
                assert t1.kwargs[k] == t2.kwargs[k]

    # Check that passing a NullQubit device takes the underlying target
    dev2 = NullQubit(target_device=dev)
    assert dev2._target_device == to_target
    assert dev2.config_filepath == to_target.config_filepath


@pytest.mark.parametrize("shots", (None, 10))
def test_supports_operator_without_decomp(shots):
    """Test that null.qubit automatically supports any operation without a decomposition."""

    # pylint: disable=too-few-public-methods
    class MyOp(qml.operation.Operator):
        pass

    tape = qml.tape.QuantumScript([MyOp(wires=(0, 1))], [qml.expval(qml.Z(0))], shots=shots)
    dev = NullQubit()

    program = dev.preprocess_transforms()
    batch, _ = program((tape,))

    assert isinstance(batch[0][0], MyOp)

    with dev.tracker:
        _ = dev.execute(batch)

    assert dev.tracker.latest["resources"].gate_types["MyOp"] == 1


def test_tracking():
    """Test some tracking values for null.qubit"""

    qs = qml.tape.QuantumScript(
        [qml.Hadamard(0), qml.FlipSign([1, 0], [0, 1])], [qml.expval(qml.PauliZ(0))]
    )
    dev = NullQubit()
    config = ExecutionConfig(gradient_method="device")

    with qml.Tracker(dev) as tracker:
        dev.execute(qs)
        dev.compute_derivatives(qs, config)
        dev.execute_and_compute_derivatives([qs] * 2, config)
        dev.compute_jvp([qs] * 3, [(0,)] * 3, config)
        dev.execute_and_compute_jvp([qs] * 4, [(0,)] * 4, config)
        dev.compute_vjp([qs] * 5, [(0,)] * 5, config)
        dev.execute_and_compute_vjp([qs] * 6, [(0,)] * 6, config)

    assert tracker.history == {
        "batches": [1],
        "results": [np.array(0)],
        "simulations": [1],
        "executions": [1, 2, 4, 6],
        "derivatives": [1, 2],
        "derivative_batches": [1],
        "execute_and_derivative_batches": [1],
        "jvps": [3, 4],
        "jvp_batches": [1],
        "execute_and_jvp_batches": [1],
        "vjps": [5, 6],
        "vjp_batches": [1],
        "execute_and_vjp_batches": [1],
        "resources": [
            qml.resource.Resources(
                num_wires=2,
                num_gates=2,
                gate_types=dd(int, {"Hadamard": 1, "FlipSign": 1}),
                gate_sizes=dd(int, {1: 1, 2: 1}),
                depth=2,
            )
        ]
        * 13,
        "errors": [{}] * 13,
    }


class TestSupportsDerivatives:
    """Test that NullQubit states what kind of derivatives it supports."""

    def test_supported_config(self):
        """Test that NullQubit says that it supports backpropagation."""
        dev = NullQubit()
        assert dev.supports_derivatives() is True
        assert dev.supports_jvp() is True
        assert dev.supports_vjp() is True

        config = ExecutionConfig(gradient_method="device")
        assert dev.supports_derivatives(config) is True
        assert dev.supports_jvp(config) is True
        assert dev.supports_vjp(config) is True

        config = ExecutionConfig(gradient_method="backprop")
        assert dev.supports_derivatives(config) is True
        assert dev.supports_jvp(config) is True
        assert dev.supports_vjp(config) is True

        qs = qml.tape.QuantumScript([], [qml.state()])
        assert dev.supports_derivatives(config, qs) is True
        assert dev.supports_jvp(config, qs) is True
        assert dev.supports_vjp(config, qs) is True

    @pytest.mark.parametrize("gradient_method", ["parameter-shift", "finite-diff", None])
    def test_doesnt_support_other_gradient_methods(self, gradient_method):
        """Test that NullQubit currently does not support other gradient methods natively."""
        dev = NullQubit()
        config = ExecutionConfig(gradient_method=gradient_method)
        assert dev.supports_derivatives(config) is False
        assert dev.supports_jvp(config) is False
        assert dev.supports_vjp(config) is False

    def test_swaps_adjoint_to_mean_device(self):
        """Test that null.qubit interprets 'adjoint' as device derivatives."""
        dev = NullQubit()
        config = ExecutionConfig(gradient_method="adjoint")
        config = dev.setup_execution_config(config)
        assert config.gradient_method == "device"
        assert dev.supports_derivatives(config) is True
        assert dev.supports_jvp(config) is True
        assert dev.supports_vjp(config) is True


class TestBasicCircuit:
    """Tests a basic circuit with one RX gate and two simple expectation values."""

    def test_basic_circuit_numpy(self):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )

        dev = NullQubit()
        result = dev.execute(qs)
        dq_res = qml.device("default.qubit").execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert qml.math.shape(dq_res) == qml.math.shape(result)
        assert qml.math.allequal(result, 0)

    @pytest.mark.autograd
    def test_autograd_results_and_backprop(self):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        dev = NullQubit()

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return qml.numpy.array(dev.execute(qs))

        expected = np.zeros(2)
        assert np.array_equal(f(phi), expected)
        assert np.array_equal(qml.jacobian(f)(phi), expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_results_and_backprop(self, use_jit):
        """Tests execution and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        dev = NullQubit()

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return dev.execute(qs)

        if use_jit:
            f = jax.jit(f)

        expected = np.zeros(2)
        assert np.array_equal(f(phi), expected)
        assert np.array_equal(jax.jacobian(f)(phi), expected)

    @pytest.mark.torch
    def test_torch_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with torch."""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        dev = NullQubit()

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return dev.execute(qs, ExecutionConfig(interface="torch", gradient_method="backprop"))

        expected = np.zeros(2)
        assert np.array_equal(f(phi), expected)
        assert np.array_equal(torch.autograd.functional.jacobian(f, phi + 0j), expected)

    @pytest.mark.xfail(reason="tf can't track derivatives")
    @pytest.mark.tf
    def test_tf_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)

        dev = NullQubit()

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = qml.tape.QuantumScript(
                [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            result = dev.execute(qs)

        expected = np.zeros(2)
        assert np.array_equal(result, expected)
        grads = [grad_tape.jacobian(result[0], [phi]), grad_tape.jacobian(result[1], [phi])]
        assert np.array_equal(grads, expected)

    @pytest.mark.tf
    @pytest.mark.parametrize("op,param", [(qml.RX, np.pi), (qml.BasisState, [1])])
    def test_qnode_returns_correct_interface(self, op, param):
        """Test that even if no interface parameters are given, result is correct."""
        dev = NullQubit()

        import tensorflow as tf

        @qml.qnode(dev, interface="tf")
        def circuit(p):
            op(p, wires=[0])
            return qml.expval(qml.PauliZ(0))

        res = circuit(param)
        assert qml.math.get_interface(res) == "tensorflow"
        # float64 due to float64 input variables
        assert qml.math.allclose(res, tf.Variable(0.0, dtype=tf.float64))  # input variables float64

    def test_basis_state_wire_order(self):
        """Test that the wire order is correct with a basis state if the tape wires have a non standard order."""

        dev = NullQubit()

        tape = qml.tape.QuantumScript([qml.BasisState([1], wires=1), qml.PauliZ(0)], [qml.state()])

        expected = np.array([1, 0, 0, 0], dtype=np.complex128)
        res = dev.execute(tape)
        assert qml.math.allclose(res, expected)


class TestSampleMeasurements:
    """A copy of the `qubit.simulate` tests, but using the device"""

    def test_single_expval(self):
        """Test a simple circuit with a single expval measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.expval(qml.PauliZ(0))], shots=10000)

        dev = NullQubit()
        result = dev.execute(qs)

        assert np.array_equal(result, 0.0)

    def test_single_probs(self):
        """Test a simple circuit with a single prob measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.probs(wires=0)], shots=10000)

        dev = NullQubit()
        result = dev.execute(qs)
        assert np.array_equal(result, [1.0, 0.0])

    def test_single_sample(self):
        """Test a simple circuit with a single sample measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=10000)

        dev = NullQubit()
        result = dev.execute(qs)

        assert np.array_equal(result, np.zeros((10000, 2)))

    def test_multi_measurements(self):
        """Test a simple circuit containing multiple measurements"""
        x, y = np.array(0.732), np.array(0.488)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
            [qml.expval(qml.Hadamard(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
            shots=10000,
        )

        dev = NullQubit()
        result = dev.execute(qs)

        assert isinstance(result, tuple)

        exp, probs, sample = result
        assert np.array_equal(exp, 0.0)
        assert np.array_equal(probs, [1.0, 0.0, 0.0, 0.0])
        assert np.array_equal(sample, np.zeros((10000, 2)))

    shots_data = [
        [10000, 10000],
        [(10000, 2)],
        [10000, 20000],
        [(10000, 2), 20000],
        [(10000, 3), 20000, (30000, 2)],
    ]

    @pytest.mark.parametrize("shots", shots_data)
    def test_expval_shot_vector(self, shots):
        """Test a simple circuit with a single expval measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.expval(qml.PauliZ(0))], shots=shots)

        dev = NullQubit()
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))
        assert all(r == 0.0 for r in result)

    @pytest.mark.parametrize("shots", shots_data)
    def test_probs_shot_vector(self, shots):
        """Test a simple circuit with a single prob measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.probs(wires=0)], shots=shots)

        dev = NullQubit()
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(np.array_equal(res, [1.0, 0.0]) for res in result)

    @pytest.mark.parametrize("shots", shots_data)
    def test_sample_shot_vector(self, shots):
        """Test a simple circuit with a single sample measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=shots)

        dev = NullQubit()
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))
        assert all(np.array_equal(res, np.zeros((s, 2))) for res, s in zip(result, shots))

    @pytest.mark.parametrize("shots", shots_data)
    def test_multi_measurement_shot_vector(self, shots):
        """Test a simple circuit containing multiple measurements for shot vectors"""

        x, y = np.array(0.732), np.array(0.488)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
            [qml.expval(qml.Hadamard(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
            shots=shots,
        )

        dev = NullQubit()
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        for shot_res, s in zip(result, shots):
            assert isinstance(shot_res, tuple)
            assert len(shot_res) == 3
            assert np.array_equal(shot_res[0], 0.0)
            assert np.array_equal(shot_res[1], [1.0, 0.0, 0.0, 0.0])
            assert np.array_equal(shot_res[2], np.zeros((s, 2)))

    def test_batch_tapes(self):
        """Test that a batch of tapes with sampling works as expected"""
        x = np.array(0.732)
        qs1 = qml.tape.QuantumScript([qml.RX(x, wires=0)], [qml.sample(wires=(0, 1))], shots=100)
        qs2 = qml.tape.QuantumScript([qml.RX(x, wires=0)], [qml.sample(wires=1)], shots=50)

        dev = NullQubit()
        res1, res2 = dev.execute((qs1, qs2))

        assert np.array_equal(res1, np.zeros((100, 2)))
        assert np.array_equal(res2, np.zeros((50, 1)))

    @pytest.mark.parametrize("all_outcomes", [True, False])
    def test_counts_wires(self, all_outcomes):
        """Test that a Counts measurement with wires works as expected"""
        x = np.array(np.pi / 2)
        qs = qml.tape.QuantumScript(
            [qml.RY(x, wires=0)], [qml.counts(wires=[0, 1], all_outcomes=all_outcomes)], shots=10000
        )

        dev = NullQubit()
        result = dev.execute(qs)

        expected = {"00": 10000, "01": 0, "10": 0, "11": 0} if all_outcomes else {"00": 10000}
        assert result == expected

    @pytest.mark.parametrize("all_outcomes", [True, False])
    def test_counts_wires_batched(self, all_outcomes):
        """Test that a Counts measurement with wires and batching works as expected"""
        x = np.array([np.pi / 2, np.pi / 4])
        qs = qml.tape.QuantumScript(
            [qml.RY(x, wires=0)],
            [qml.counts(wires=[0, 1], all_outcomes=all_outcomes)],
            shots=[50, 100, 150],
        )

        dev = NullQubit()
        result = dev.execute(qs)

        if all_outcomes:
            assert result == tuple(({"00": s, "01": 0, "10": 0, "11": 0},) * 2 for s in qs.shots)
        else:
            assert result == tuple(({"00": s},) * 2 for s in qs.shots)

    @pytest.mark.parametrize("all_outcomes", [False, True])
    def test_counts_obs(self, all_outcomes):
        """Test that a Counts measurement with an observable works as expected"""
        x = np.array(np.pi / 2)
        qs = qml.tape.QuantumScript(
            [qml.RY(x, wires=0)],
            [qml.counts(qml.PauliZ(0), all_outcomes=all_outcomes)],
            shots=10000,
        )

        dev = NullQubit()
        result = dev.execute(qs)
        assert result == ({-1: 10000, 1: 0} if all_outcomes else {-1: 10000})

    @pytest.mark.parametrize("all_outcomes", [False, True])
    def test_counts_obs_batched(self, all_outcomes):
        """Test that a Counts measurement with an observable and batching works as expected"""
        x = np.array([np.pi / 2, np.pi / 4])
        qs = qml.tape.QuantumScript(
            [qml.RY(x, wires=0)],
            [qml.counts(qml.PauliZ(0), all_outcomes=all_outcomes)],
            shots=[50, 100, 150],
        )

        dev = NullQubit()
        result = dev.execute(qs)
        assert (
            result == tuple(({-1: s, 1: 0},) * 2 for s in qs.shots)
            if all_outcomes
            else tuple(({-1: s},) * 2 for s in qs.shots)
        )


class TestExecutingBatches:
    """Tests involving executing multiple circuits at the same time."""

    @staticmethod
    def f(dev, phi):
        """A function that executes a batch of scripts on NullQubit without preprocessing."""
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
        config = ExecutionConfig(interface=qml.math.get_interface(phi), gradient_method="backprop")
        return dev.execute((qs1, qs2), config)

    @staticmethod
    def f_hashable(phi):
        """A function that executes a batch of scripts on NullQubit without preprocessing."""
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
        config = qml.devices.ExecutionConfig(
            gradient_method="backprop", interface=qml.math.get_interface(phi)
        )
        return NullQubit().execute((qs1, qs2), execution_config=config)

    @staticmethod
    def assert_results(results):
        """Assert that the results match f's return values."""
        assert isinstance(results, tuple)
        expvals, probs = results
        assert isinstance(expvals, tuple)
        assert np.array_equal(expvals, np.zeros(2))
        assert np.array_equal(probs, [1, 0, 0, 0])

    def test_numpy(self):
        """Test that results are expected when the parameter does not have a parameter."""
        dev = NullQubit()

        phi = 0.892
        results = self.f(dev, phi)
        self.assert_results(results)

    @pytest.mark.autograd
    def test_autograd(self):
        """Test batches can be executed and have backprop derivatives in autograd."""
        dev = NullQubit()

        phi = qml.numpy.array(-0.629)
        results = self.f(dev, phi)
        self.assert_results(results)

        g0 = qml.jacobian(lambda x: qml.numpy.array(self.f(dev, x)[0]))(phi)
        assert np.array_equal(g0, np.zeros(2))

        g1 = qml.jacobian(lambda x: qml.numpy.array(self.f(dev, x)[1]))(phi)
        assert qml.math.allclose(g1, np.zeros(4))

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax(self, use_jit):
        """Test batches can be executed and have backprop derivatives in jax."""
        import jax

        phi = jax.numpy.array(0.123)

        f = jax.jit(self.f_hashable) if use_jit else self.f_hashable
        results = f(phi)
        self.assert_results(results)

        g0, g1 = jax.jacobian(f)(phi)
        assert isinstance(g0, tuple)
        assert np.array_equal(g0, np.zeros(2))
        assert np.array_equal(g1, np.zeros(4))

    @pytest.mark.torch
    def test_torch(self):
        """Test batches can be executed and have backprop derivatives in torch."""
        import torch

        dev = NullQubit()

        x = torch.tensor(9.6243)

        results = self.f(dev, x)
        self.assert_results(results)

        g0 = torch.autograd.functional.jacobian(lambda y: self.f(dev, y)[0], x)
        assert np.array_equal(g0, np.zeros(2))
        g1 = torch.autograd.functional.jacobian(lambda y: self.f(dev, y)[1], x)
        assert np.array_equal(g1, np.zeros(4))

    @pytest.mark.xfail(reason="tf can't track derivatives")
    @pytest.mark.tf
    def test_tf(self):
        """Test batches can be executed and have backprop derivatives in tf."""

        import tensorflow as tf

        dev = NullQubit()

        x = tf.Variable(5.2281)
        with tf.GradientTape(persistent=True) as tape:
            results = self.f(dev, x)

        self.assert_results(results)

        g00 = tape.gradient(results[0][0], x)
        assert g00 == 0
        g01 = tape.gradient(results[0][1], x)
        assert g01 == 0

        g1 = tape.jacobian(results[1], x)
        assert np.array_equal(g1, np.zeros(4))


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
        config = ExecutionConfig(interface=qml.math.get_interface(scale))
        return dev.execute(qs, config)

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
        return NullQubit().execute(qs)

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
        dev = NullQubit()
        x = qml.numpy.array(0.52)
        assert self.f(dev, x, style=style) == 0
        assert qml.grad(self.f)(dev, x, style=style) == 0

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    @pytest.mark.parametrize("style", ("sum", "hermitian"))
    def test_jax_backprop(self, style, use_jit):
        """Test that backpropagation derivatives work with jax with hamiltonians and large sums."""
        import jax

        x = jax.numpy.array(0.52, dtype=jax.numpy.float64)
        f = jax.jit(self.f_hashable, static_argnums=(1, 2, 3)) if use_jit else self.f_hashable

        assert f(x, style=style) == 0
        assert jax.grad(f)(x, style=style) == 0

    @pytest.mark.xfail(reason="torch backprop does not work")
    @pytest.mark.torch
    @pytest.mark.parametrize("style", ("sum", "hermitian"))
    def test_torch_backprop(self, style):
        """Test that backpropagation derivatives work with torch with hamiltonians and large sums."""
        import torch

        dev = NullQubit()

        x = torch.tensor(-0.289, requires_grad=True)
        out = self.f(dev, x, style=style)
        assert out == torch.tensor(0)

        out.backward()  # pylint:disable=no-member
        assert x.grad == 0

    @pytest.mark.xfail(reason="tf can't track derivatives")
    @pytest.mark.tf
    @pytest.mark.parametrize("style", ("sum", "hermitian"))
    def test_tf_backprop(self, style):
        """Test that backpropagation derivatives work with tensorflow with hamiltonians and large sums."""
        import tensorflow as tf

        dev = NullQubit()

        x = tf.Variable(0.5)

        with tf.GradientTape() as tape1:
            out = self.f(dev, x, style=style)

        assert out == 0

        g1 = tape1.gradient(out, x)
        assert g1 == 0


@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
@pytest.mark.parametrize("config", [None, ExecutionConfig(gradient_method="device")])
class TestDeviceDifferentiation:
    """Tests device differentiation integration with NullQubit."""

    def test_derivatives_single_circuit(self, config):
        """Tests derivatives with a single circuit."""
        dev = NullQubit()
        x = np.array(np.pi / 7)
        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])

        [qs], _ = dev.preprocess_transforms(config)((qs,))
        actual_grad = dev.compute_derivatives(qs, config)
        assert isinstance(actual_grad, np.ndarray)
        assert actual_grad.shape == ()  # pylint: disable=no-member
        assert actual_grad == 0

        actual_val, actual_grad = dev.execute_and_compute_derivatives(qs, config)
        assert actual_val == 0
        assert actual_grad == 0

    def test_derivatives_list_with_single_circuit(self, config):
        """Tests a basic example with a batch containing a single circuit."""
        dev = NullQubit()
        x = np.array(np.pi / 7)
        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])

        transform_program = dev.preprocess_transforms(config)
        [qs], _ = transform_program((qs,))
        actual_grad = dev.compute_derivatives([qs], config)
        assert actual_grad == (np.array(0.0),)

        actual_val, actual_grad = dev.execute_and_compute_derivatives([qs], config)
        assert actual_val == (np.array(0.0),)
        assert actual_grad == (np.array(0.0),)

    def test_derivatives_many_tapes_many_results(self, config):
        """Tests a basic example with a batch of circuits of varying return shapes."""
        dev = NullQubit()
        x = np.array(np.pi / 7)
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        actual_grad = dev.compute_derivatives([single_meas, multi_meas], config)
        assert actual_grad == (0.0, (0.0, 0.0))

    def test_derivatives_integration(self, config):
        """Tests the expected workflow done by a calling method."""
        dev = NullQubit()
        x = np.array(np.pi / 7)
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )

        program, new_ec = dev.preprocess(config)
        circuits, _ = program([single_meas, multi_meas])
        actual_grad = dev.compute_derivatives(circuits, new_ec)

        if config and config.gradient_method == "device":
            assert new_ec.use_device_gradient
            assert new_ec.grad_on_execution

        assert actual_grad == (0.0, (0.0, 0.0))

    def test_jvps_single_circuit(self, config):
        """Tests jvps with a single circuit."""
        dev = NullQubit()
        x = np.array(np.pi / 7)
        tangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        [qs], _ = dev.preprocess_transforms(config)((qs,))

        actual_grad = dev.compute_jvp(qs, tangent, config)
        assert isinstance(actual_grad, np.ndarray)
        assert actual_grad.shape == ()  # pylint: disable=no-member
        assert actual_grad == 0.0

        actual_val_and_grad = dev.execute_and_compute_jvp(qs, tangent, config)
        assert actual_val_and_grad == (0, 0)

    def test_jvps_list_with_single_circuit(self, config):
        """Tests a basic example with a batch containing a single circuit."""
        dev = NullQubit()
        x = np.array(np.pi / 7)
        tangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        [qs], _ = dev.preprocess_transforms(config)((qs,))

        actual_grad = dev.compute_jvp([qs], [tangent], config)
        assert actual_grad == (np.array(0.0),)

        actual_val, actual_grad = dev.execute_and_compute_jvp([qs], [tangent], config)
        assert actual_val == (np.array(0.0),)
        assert actual_grad == (np.array(0.0),)

    def test_jvps_many_tapes_many_results(self, config):
        """Tests a basic example with a batch of circuits of varying return shapes."""
        dev = NullQubit()
        x = np.array(np.pi / 7)
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        tangents = [(0.456,), (0.789,)]

        actual_grad = dev.compute_jvp([single_meas, multi_meas], tangents, config)
        assert actual_grad == (0.0, (0.0, 0.0))

        actual_val, actual_grad = dev.execute_and_compute_jvp(
            [single_meas, multi_meas], tangents, config
        )
        assert actual_val == (0.0, (0.0, 0.0))
        assert actual_grad == (0.0, (0.0, 0.0))

    def test_jvps_integration(self, config):
        """Tests the expected workflow done by a calling method."""
        dev = NullQubit()
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

        if config and config.gradient_method == "device":
            assert new_ec.use_device_gradient
            assert new_ec.grad_on_execution

        assert actual_grad == (0.0, (0.0, 0.0))

    def test_vjps_single_circuit(self, config):
        """Tests vjps with a single circuit."""
        dev = NullQubit()
        x = np.array(np.pi / 7)
        cotangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        [qs], _ = dev.preprocess_transforms(config)((qs,))

        actual_grad = dev.compute_vjp(qs, cotangent, config)
        assert actual_grad == (0.0,)

        actual_val_and_grad = dev.execute_and_compute_vjp(qs, cotangent, config)
        assert actual_val_and_grad == ((0.0,), (0.0,))

    def test_vjps_list_with_single_circuit(self, config):
        """Tests a basic example with a batch containing a single circuit."""
        dev = NullQubit()
        x = np.array(np.pi / 7)
        cotangent = (0.456,)

        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        [qs], _ = dev.preprocess_transforms(config)((qs,))

        actual_grad = dev.compute_vjp([qs], [cotangent], config)
        assert actual_grad == ((0.0,),)

        actual_val, actual_grad = dev.execute_and_compute_vjp([qs], [cotangent], config)
        assert actual_val == ((0.0,),)
        assert actual_grad == ((0.0,),)

    def test_vjps_many_tapes_many_results(self, config):
        """Tests a basic example with a batch of circuits of varying return shapes."""
        dev = NullQubit()
        x = np.array(np.pi / 7)
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        cotangents = [(0.456,), (0.789, 0.123)]

        actual_grad = dev.compute_vjp([single_meas, multi_meas], cotangents, config)
        assert actual_grad == ((0.0,), (0.0,))

        actual_val, actual_grad = dev.execute_and_compute_vjp(
            [single_meas, multi_meas], cotangents, config
        )
        assert actual_val == (0.0, (0.0, 0.0))
        assert actual_grad == ((0.0,), (0.0,))

    def test_vjps_integration(self, config):
        """Tests the expected workflow done by a calling method."""
        dev = NullQubit()
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

        if config and config.gradient_method == "device":
            assert new_ec.use_device_gradient
            assert new_ec.grad_on_execution

        assert actual_grad == ((0.0,), (0.0,))


class TestClassicalShadows:
    """Test that classical shadow measurements works with the new device"""

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_shape_and_dtype(self, n_qubits):
        """Test that the shape and dtype of the measurement is correct"""
        dev = NullQubit()

        ops = [qml.Hadamard(i) for i in range(n_qubits)]
        qs = qml.tape.QuantumScript(ops, [qml.classical_shadow(range(n_qubits))], shots=100)
        res = dev.execute(qs)

        assert np.array_equal(res, np.zeros((2, 100, n_qubits)))
        assert res.dtype == np.int8

    def test_expval(self, seed):
        """Test that shadow expval measurements work as expected"""
        dev = NullQubit()

        ops = [qml.Hadamard(0), qml.Hadamard(1)]
        meas = [qml.shadow_expval(qml.PauliX(0) @ qml.PauliX(1), seed=seed)]
        qs = qml.tape.QuantumScript(ops, meas, shots=1000)
        assert dev.execute(qs) == np.array(0.0)

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_multiple_shadow_measurements(self, n_qubits):
        """Test that multiple classical shadow measurements work as expected"""
        dev = NullQubit()

        ops = [qml.Hadamard(i) for i in range(n_qubits)]
        mps = [qml.classical_shadow(range(n_qubits)), qml.classical_shadow(range(n_qubits))]
        qs = qml.tape.QuantumScript(ops, mps, shots=100)
        res = dev.execute(qs)

        assert isinstance(res, tuple)
        assert len(res) == 2
        assert all(r.dtype == np.int8 for r in res)
        assert np.array_equal(res, np.zeros((2, 2, 100, n_qubits)))

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
    def test_shot_vectors(self, n_qubits, shots):
        """Test that classical shadows works when given a shot vector"""
        dev = NullQubit()
        shots = qml.measurements.Shots(shots)

        ops = [qml.Hadamard(i) for i in range(n_qubits)]
        qs = qml.tape.QuantumScript(ops, [qml.classical_shadow(range(n_qubits))], shots=shots)
        res = dev.execute(qs)

        assert isinstance(res, tuple)
        assert len(res) == len(list(shots))

        for r, s in zip(res, shots):
            assert np.array_equal(r, np.zeros((2, s, n_qubits)))
            assert r.dtype == np.int8

    def test_batching_not_supported(self):
        """Test that classical_shadow does not work with batching."""
        dev = NullQubit()

        ops = [qml.RX([1.1, 2.2], 0)]
        qs = qml.tape.QuantumScript(ops, [qml.classical_shadow([0])], shots=100)
        with pytest.raises(ValueError, match="broadcasting is not supported"):
            _ = dev.execute(qs)


@pytest.mark.parametrize("n_wires", [1, 2, 3])
def test_projector_dynamic_type(n_wires):
    """Test that qml.Projector yields the expected results for both of its subclasses."""
    wires = list(range(n_wires))
    dev = NullQubit()
    ops = [qml.adjoint(qml.Hadamard(q)) for q in wires]
    # non-zero states will still yield zero-state results
    basis_state = np.ones((n_wires,))
    state_vector = np.zeros((2**n_wires,))
    state_vector[-1] = 1

    for state in [basis_state, state_vector]:
        qs = qml.tape.QuantumScript(ops, [qml.expval(qml.Projector(state, wires))])
        assert dev.execute(qs) == 0.0


class TestIntegration:
    """Various integration tests"""

    @pytest.mark.parametrize("wires,expected", [(None, [0, 0]), (3, [0, 0, 0])])
    def test_sample_uses_device_wires(self, wires, expected):
        """Test that if device wires are given, then they are used by sample."""
        dev = NullQubit(wires=wires)

        @qml.qnode(dev, shots=5)
        def circuit():
            qml.PauliX(2)
            qml.Identity(0)
            return qml.sample()

        assert np.array_equal(circuit(), [expected] * 5)

    @pytest.mark.parametrize(
        "wires,expected",
        [
            (None, [1, 0, 0, 0]),
            (3, [1] + [0] * 7),
        ],
    )
    def test_state_uses_device_wires(self, wires, expected):
        """Test that if device wires are given, then they are used by state."""
        dev = NullQubit(wires=wires)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(2)
            qml.Identity(0)
            return qml.state()

        assert np.array_equal(circuit(), expected)

    @pytest.mark.parametrize(
        "wires,expected",
        [
            (None, [1, 0, 0, 0]),
            (3, [1] + [0] * 7),
        ],
    )
    def test_probs_uses_device_wires(self, wires, expected):
        """Test that if device wires are given, then they are used by probs."""
        dev = NullQubit(wires=wires)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(2)
            qml.Identity(0)
            return qml.probs()

        assert np.array_equal(circuit(), expected)

    @pytest.mark.parametrize(
        "wires,all_outcomes,expected",
        [
            (None, False, {"00": 10}),
            (None, True, {"00": 10, "10": 0, "01": 0, "11": 0}),
            (3, False, {"000": 10}),
            (
                3,
                True,
                {"000": 10, "001": 0, "010": 0, "011": 0, "100": 0, "101": 0, "110": 0, "111": 0},
            ),
        ],
    )
    def test_counts_uses_device_wires(self, wires, all_outcomes, expected):
        """Test that if device wires are given, then they are used by probs."""
        dev = NullQubit(wires=wires)

        @qml.qnode(dev, interface=None, shots=10)
        def circuit():
            qml.PauliX(2)
            qml.Identity(0)
            return qml.counts(all_outcomes=all_outcomes)

        assert circuit() == expected

    @pytest.mark.parametrize(
        "diff_method", ["device", "adjoint", "backprop", "finite-diff", "parameter-shift"]
    )
    def test_expected_shape_all_methods(self, diff_method, seed):
        """Test that the gradient shape is as expected with all diff methods."""
        n_wires = 4

        shape = qml.StronglyEntanglingLayers.shape(n_layers=5, n_wires=n_wires)
        rng = np.random.default_rng(seed=seed)
        params = qml.numpy.array(rng.random(shape))
        dev = qml.device("null.qubit")

        diff_method = "device"

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(params):
            qml.StronglyEntanglingLayers(params, wires=range(n_wires))
            return [qml.expval(qml.Z(i)) for i in range(n_wires)]

        def cost(params):
            out = circuit(params)
            return qml.math.sum(out)

        assert np.array_equal(qml.grad(cost)(params), np.zeros(shape))


@pytest.mark.parametrize(
    "method,device_vjp", [("device", False), ("device", True), ("parameter-shift", False)]
)
class TestJacobian:
    """Test that the jacobian of circuits can be computed."""

    @staticmethod
    def get_circuit(method, device_vjp):
        @qml.qnode(NullQubit(), diff_method=method, device_vjp=device_vjp)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.probs(wires=[0]), qml.expval(qml.PauliZ(0))

        return circuit

    def test_jacobian_autograd(self, method, device_vjp):
        """Test the jacobian with autograd."""

        @qml.qnode(NullQubit(), diff_method=method, device_vjp=device_vjp)
        def circuit(x, mp):
            qml.RX(x, wires=0)
            return qml.apply(mp)

        x = qml.numpy.array(0.1)
        probs_jac = qml.jacobian(circuit)(x, qml.probs(wires=[0]))
        expval_jac = qml.jacobian(circuit)(x, qml.expval(qml.PauliZ(0)))
        assert np.array_equal(probs_jac, np.zeros(2))
        assert np.array_equal(expval_jac, 0.0)

    @pytest.mark.jax
    def test_jacobian_jax(self, method, device_vjp):
        """Test the jacobian with jax."""
        import jax
        from jax import numpy as jnp

        x = jnp.array(0.1)
        if device_vjp:
            pytest.xfail(reason="cannot handle jax's processing of device VJPs")
        probs_jac, expval_jac = jax.jacobian(self.get_circuit(method, device_vjp))(x)
        assert np.array_equal(probs_jac, np.zeros(2))
        assert np.array_equal(expval_jac, 0.0)

    @pytest.mark.tf
    def test_jacobian_tf(self, method, device_vjp):
        """Test the jacobian with tf."""
        import tensorflow as tf

        x = tf.Variable(0.1)
        with tf.GradientTape(persistent=True) as tape:
            res = self.get_circuit(method, device_vjp)(x)

        probs_jac = tape.jacobian(res[0], x, experimental_use_pfor=not device_vjp)
        assert np.array_equal(probs_jac, np.zeros(2))
        if method == "parameter-shift":
            pytest.xfail(reason="TF panics when computing the second jacobian here.")
        expval_jac = tape.jacobian(res[1], x, experimental_use_pfor=not device_vjp)
        assert np.array_equal(expval_jac, 0.0)

    @pytest.mark.torch
    def test_jacobian_torch(self, method, device_vjp):
        """Test the jacobian with torch."""
        import torch

        x = torch.tensor(0.1, requires_grad=True)
        probs_jac, expval_jac = torch.autograd.functional.jacobian(
            self.get_circuit(method, device_vjp), x
        )
        assert np.array_equal(probs_jac, np.zeros(2))
        assert np.array_equal(expval_jac, 0.0)


@pytest.mark.parametrize("shots", [None, 100, [(100, 3)]])
@pytest.mark.parametrize("x", [1.1, [1.1, 2.2]])
@pytest.mark.parametrize(
    "mp",
    [
        qml.expval(qml.PauliZ(0)),
        qml.var(qml.PauliZ(0)),
        qml.state(),
        qml.probs(),
        qml.probs(wires=[1]),
        qml.sample(),
        qml.sample(wires=[1]),
        qml.sample(op=qml.PauliZ(0)),
        qml.mutual_info([0], [1]),
        qml.purity([1]),
        qml.purity([0, 1]),
        qml.density_matrix([1]),
        qml.density_matrix([0, 1]),
        qml.vn_entropy([1]),
        qml.vn_entropy([0, 1]),
        qml.classical_shadow([1]),
        qml.classical_shadow([0, 1]),
        qml.shadow_expval(qml.PauliZ(0)),
        qml.shadow_expval([qml.PauliZ(0), qml.PauliZ(1)]),
    ],
)
def test_measurement_shape_matches_default_qubit(mp, x, shots):
    """Test that all results match default.qubit in shape."""
    if shots and not isinstance(mp, (SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP)):
        pytest.skip(reason="MeasurementProcess doesn't work with finite shots.")

    if not shots and not isinstance(mp, StateMeasurement):
        pytest.skip(reason="MeasurementProcess doesn't work in analytic mode.")

    if isinstance(x, list) and isinstance(mp, (ClassicalShadowMP, ShadowExpvalMP)):
        pytest.xfail(reason="default.qubit cannot handle batching with shadow measurements")

    nq = qml.device("null.qubit")
    dq = qml.device("default.qubit")

    def circuit(param):
        qml.RX(param, 0)
        qml.CNOT([0, 1])
        return qml.apply(mp)

    res = qml.set_shots(qml.QNode(circuit, nq), shots=shots)(x)
    target = qml.set_shots(qml.QNode(circuit, dq), shots=shots)(x)
    assert qml.math.shape(res) == qml.math.shape(target)


# pylint: disable=unused-argument
@pytest.mark.capture
def test_execute_plxpr():
    """Test that null.qubit can execute plxpr."""

    import jax

    def f(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0)), qml.probs(), 4, qml.var(qml.X(0)), qml.state()

    jaxpr = jax.make_jaxpr(f)(0.5)

    dev = qml.device("null.qubit", wires=4)
    res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.0)
    assert qml.math.allclose(res[0], 0)
    assert qml.math.allclose(res[1], jax.numpy.zeros(2**4))
    assert qml.math.allclose(res[2], 0)  # other values are still just zero
    assert qml.math.allclose(res[3], 0)
    assert qml.math.allclose(res[4], jax.numpy.zeros(2**4, dtype=complex))


@pytest.mark.capture
def test_execute_plxpr_shots():
    import jax

    def f(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0)), 5, qml.sample(wires=(0, 1))

    jaxpr = jax.make_jaxpr(f)(0.5)

    dev = qml.device("null.qubit", wires=4)
    res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.0, shots=50)
    assert qml.math.allclose(res[0], 0)
    assert qml.math.allclose(res[1], 0)
    assert qml.math.allclose(res[2], jax.numpy.zeros((50, 2)))


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestNullQubitGraphModeExclusive:  # pylint: disable=too-few-public-methods
    """Tests for NullQubit features that require graph mode enabled.
    The legacy decomposition mode should not be able to run these tests.
    NOTE: All tests in this suite will auto-enable graph mode via fixture.
    """

    def test_insufficient_work_wires_causes_fallback(self):
        """Test that if a decomposition requires more work wires than available on null.qubit,
        that decomposition is discarded and fallback is used."""

        class MyNullQubitOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods
            num_wires = 1

        @qml.register_resources({qml.H: 2})
        def decomp_fallback(wires):
            qml.H(wires)
            qml.H(wires)

        @qml.register_resources({qml.X: 1}, work_wires={"burnable": 5})
        def decomp_with_work_wire(wires):
            qml.X(wires)

        qml.add_decomps(MyNullQubitOp, decomp_fallback, decomp_with_work_wire)

        tape = qml.tape.QuantumScript([MyNullQubitOp(0)])
        dev = qml.device("null.qubit", wires=1)  # Only 1 wire, but decomp needs 5 burnable
        program = dev.preprocess_transforms()
        (out_tape,), _ = program([tape])

        assert len(out_tape.operations) == 2
        assert out_tape.operations[0].name == "Hadamard"
        assert out_tape.operations[1].name == "Hadamard"

    def test_operator_without_graph_decomposition_runs_without_error(self):
        """Test that an operator with no graph-based decomposition outside the gateset
        still runs without error on NullQubit."""

        # Create a custom operator that's not in the standard gateset
        class CustomOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods
            num_wires = 2

            def decomposition(self):
                # Legacy decomposition only (no graph-based decomposition registered)
                return [qml.PauliX(self.wires[0]), qml.PauliY(self.wires[1])]

        # Create a tape with this custom operator
        tape = qml.tape.QuantumScript([CustomOp(wires=[0, 1])], [qml.expval(qml.Z(0))])
        dev = qml.device("null.qubit", wires=3)

        # This should not raise an error even though there's no graph-based decomposition
        program = dev.preprocess_transforms()
        (out_tape,), _ = program([tape])

        qml.assert_equal(out_tape, tape)

        # NullQubit should accept the operator even if it's not decomposed at preprocessing
        # The key point is that it runs without error

        # Execution should work without error
        result = dev.execute([out_tape])

        # Should return 0 (as expected for NullQubit)
        assert result[0] == 0
