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
"""Unit tests for simulate in devices/qubit."""

import pytest

import numpy as np

import pennylane as qml
from pennylane.devices.qubit import simulate, get_final_state, measure_final_state


class TestCurrentlyUnsupportedCases:
    # pylint: disable=too-few-public-methods
    def test_sample_based_observable(self):
        """Test sample-only measurements raise a notimplementedError."""

        qs = qml.tape.QuantumScript(measurements=[qml.sample(wires=0)])
        with pytest.raises(NotImplementedError):
            simulate(qs)


def test_custom_operation():
    """Test execution works with a manually defined operator if it has a matrix."""

    # pylint: disable=too-few-public-methods
    class MyOperator(qml.operation.Operator):
        num_wires = 1

        @staticmethod
        def compute_matrix():
            return qml.PauliX.compute_matrix()

    qs = qml.tape.QuantumScript([MyOperator(0)], [qml.expval(qml.PauliZ(0))])

    result = simulate(qs)
    assert qml.math.allclose(result, -1.0)


# pylint: disable=too-few-public-methods
class TestStatePrepBase:
    """Tests integration with various state prep methods."""

    def test_basis_state(self):
        """Test that the BasisState operator prepares the desired state."""
        qs = qml.tape.QuantumScript(
            measurements=[qml.probs(wires=(0, 1, 2))], prep=[qml.BasisState([0, 1], wires=(0, 1))]
        )
        probs = simulate(qs)
        expected = np.zeros(8)
        expected[2] = 1.0
        assert qml.math.allclose(probs, expected)


class TestBasicCircuit:
    """Tests a basic circuit with one rx gate and two simple expectation values."""

    def test_basic_circuit_numpy(self):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

        state, is_state_batched = get_final_state(qs)
        result = measure_final_state(qs, state, is_state_batched)

        assert np.allclose(state, np.array([np.cos(phi / 2), -1j * np.sin(phi / 2)]))
        assert not is_state_batched

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

    @pytest.mark.autograd
    def test_autograd_results_and_backprop(self):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return qml.numpy.array(simulate(qs))

        result = f(phi)
        expected = np.array([-np.sin(phi), np.cos(phi)])
        assert qml.math.allclose(result, expected)

        g = qml.jacobian(f)(phi)
        expected = np.array([-np.cos(phi), -np.sin(phi)])
        assert qml.math.allclose(g, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_results_and_backprop(self, use_jit):
        """Tests exeuction and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return simulate(qs)

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

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return simulate(qs)

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

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = qml.tape.QuantumScript(
                [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            result = simulate(qs)

        assert qml.math.allclose(result[0], -tf.sin(phi))
        assert qml.math.allclose(result[1], tf.cos(phi))

        grad0 = grad_tape.jacobian(result[0], [phi])
        grad1 = grad_tape.jacobian(result[1], [phi])

        assert qml.math.allclose(grad0[0], -tf.cos(phi))
        assert qml.math.allclose(grad1[0], -tf.sin(phi))


class TestBroadcasting:
    """Test that simulate works with broadcasted parameters"""

    def test_broadcasted_prep_state(self):
        """Test that simulate works for state measurements
        when the state prep has broadcasted parameters"""
        x = np.array(1.2)

        ops = [qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]
        prep = [qml.StatePrep(np.eye(4), wires=[0, 1])]

        qs = qml.tape.QuantumScript(ops, measurements, prep)
        res = simulate(qs)

        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]))
        assert np.allclose(res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]))

        state, is_state_batched = get_final_state(qs)
        res = measure_final_state(qs, state, is_state_batched)
        expected_state = np.array(
            [
                [np.cos(x / 2), 0, 0, np.sin(x / 2)],
                [0, np.cos(x / 2), np.sin(x / 2), 0],
                [-np.sin(x / 2), 0, 0, np.cos(x / 2)],
                [0, -np.sin(x / 2), np.cos(x / 2), 0],
            ]
        ).reshape((4, 2, 2))

        assert np.allclose(state, expected_state)
        assert is_state_batched
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]))
        assert np.allclose(res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]))

    def test_broadcasted_op_state(self):
        """Test that simulate works for state measurements
        when an operation has broadcasted parameters"""
        x = np.array([0.8, 1.0, 1.2, 1.4])

        ops = [qml.PauliX(wires=1), qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]

        qs = qml.tape.QuantumScript(ops, measurements)
        res = simulate(qs)

        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.cos(x))
        assert np.allclose(res[1], -np.cos(x))

        state, is_state_batched = get_final_state(qs)
        res = measure_final_state(qs, state, is_state_batched)

        expected_state = np.zeros((4, 2, 2))
        expected_state[:, 0, 1] = np.cos(x / 2)
        expected_state[:, 1, 0] = np.sin(x / 2)

        assert np.allclose(state, expected_state)
        assert is_state_batched
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.cos(x))
        assert np.allclose(res[1], -np.cos(x))

    def test_broadcasted_prep_sample(self):
        """Test that simulate works for sample measurements
        when the state prep has broadcasted parameters"""
        x = np.array(1.2)

        ops = [qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]
        prep = [qml.StatePrep(np.eye(4), wires=[0, 1])]

        qs = qml.tape.QuantumScript(ops, measurements, prep, shots=qml.measurements.Shots(10000))
        res = simulate(qs, rng=123)

        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(
            res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]), atol=0.05
        )
        assert np.allclose(
            res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]), atol=0.05
        )

        state, is_state_batched = get_final_state(qs)
        res = measure_final_state(qs, state, is_state_batched, rng=123)
        expected_state = np.array(
            [
                [np.cos(x / 2), 0, 0, np.sin(x / 2)],
                [0, np.cos(x / 2), np.sin(x / 2), 0],
                [-np.sin(x / 2), 0, 0, np.cos(x / 2)],
                [0, -np.sin(x / 2), np.cos(x / 2), 0],
            ]
        ).reshape((4, 2, 2))

        assert np.allclose(state, expected_state)
        assert is_state_batched
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(
            res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]), atol=0.05
        )
        assert np.allclose(
            res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]), atol=0.05
        )

    def test_broadcasted_op_sample(self):
        """Test that simulate works for sample measurements
        when an operation has broadcasted parameters"""
        x = np.array([0.8, 1.0, 1.2, 1.4])

        ops = [qml.PauliX(wires=1), qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]

        qs = qml.tape.QuantumScript(ops, measurements, shots=qml.measurements.Shots(10000))
        res = simulate(qs, rng=123)

        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.cos(x), atol=0.05)
        assert np.allclose(res[1], -np.cos(x), atol=0.05)

        state, is_state_batched = get_final_state(qs)
        res = measure_final_state(qs, state, is_state_batched, rng=123)

        expected_state = np.zeros((4, 2, 2))
        expected_state[:, 0, 1] = np.cos(x / 2)
        expected_state[:, 1, 0] = np.sin(x / 2)

        assert np.allclose(state, expected_state)
        assert is_state_batched
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.cos(x), atol=0.05)
        assert np.allclose(res[1], -np.cos(x), atol=0.05)

    def test_broadcasting_with_extra_measurement_wires(self, mocker):
        """Test that broadcasting works when the operations don't act on all wires."""
        # I can't mock anything in `simulate` because the module name is the function name
        spy = mocker.spy(qml, "map_wires")
        x = np.array([0.8, 1.0, 1.2, 1.4])

        ops = [qml.PauliX(wires=2), qml.RY(x, wires=1), qml.CNOT(wires=[1, 2])]
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(3)]

        qs = qml.tape.QuantumScript(ops, measurements)
        res = simulate(qs)

        assert isinstance(res, tuple)
        assert len(res) == 3
        assert np.allclose(res[0], 1.0)
        assert np.allclose(res[1], np.cos(x))
        assert np.allclose(res[2], -np.cos(x))
        assert spy.call_args_list[0].args == (qs, {2: 0, 1: 1, 0: 2})


class TestDebugger:
    """Tests that the debugger works for a simple circuit"""

    class Debugger:
        """A dummy debugger class"""

        def __init__(self):
            self.active = True
            self.snapshots = {}

    def test_debugger_numpy(self):
        """Test debugger with numpy"""
        phi = np.array(0.397)
        ops = [qml.Snapshot(), qml.RX(phi, wires=0), qml.Snapshot("final_state")]
        qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])

        debugger = self.Debugger()
        result = simulate(qs, debugger=debugger)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert np.allclose(debugger.snapshots[0], [1, 0])
        assert np.allclose(
            debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        )

    @pytest.mark.autograd
    def test_debugger_autograd(self):
        """Tests debugger with autograd"""
        phi = qml.numpy.array(-0.52)
        debugger = self.Debugger()

        def f(x):
            ops = [qml.Snapshot(), qml.RX(x, wires=0), qml.Snapshot("final_state")]
            qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
            return qml.numpy.array(simulate(qs, debugger=debugger))

        result = f(phi)
        expected = np.array([-np.sin(phi), np.cos(phi)])
        assert qml.math.allclose(result, expected)

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        assert qml.math.allclose(
            debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        )

    @pytest.mark.jax
    def test_debugger_jax(self):
        """Tests debugger with JAX"""
        import jax

        phi = jax.numpy.array(0.678)
        debugger = self.Debugger()

        def f(x):
            ops = [qml.Snapshot(), qml.RX(x, wires=0), qml.Snapshot("final_state")]
            qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
            return simulate(qs, debugger=debugger)

        result = f(phi)
        assert qml.math.allclose(result[0], -np.sin(phi))
        assert qml.math.allclose(result[1], np.cos(phi))

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        assert qml.math.allclose(
            debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        )

    @pytest.mark.torch
    def test_debugger_torch(self):
        """Tests debugger with torch"""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)
        debugger = self.Debugger()

        def f(x):
            ops = [qml.Snapshot(), qml.RX(x, wires=0), qml.Snapshot("final_state")]
            qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
            return simulate(qs, debugger=debugger)

        result = f(phi)
        assert qml.math.allclose(result[0], -torch.sin(phi))
        assert qml.math.allclose(result[1], torch.cos(phi))

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        print(debugger.snapshots["final_state"])
        assert qml.math.allclose(
            debugger.snapshots["final_state"],
            torch.tensor([torch.cos(phi / 2), -torch.sin(phi / 2) * 1j]),
        )

    # pylint: disable=invalid-unary-operand-type
    @pytest.mark.tf
    def test_debugger_tf(self):
        """Tests debugger with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)
        debugger = self.Debugger()

        ops = [qml.Snapshot(), qml.RX(phi, wires=0), qml.Snapshot("final_state")]
        qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
        result = simulate(qs, debugger=debugger)

        assert qml.math.allclose(result[0], -tf.sin(phi))
        assert qml.math.allclose(result[1], tf.cos(phi))

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        assert qml.math.allclose(
            debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        )


class TestSampleMeasurements:
    """Tests circuits with sample-based measurements"""

    def test_single_expval(self):
        """Test a simple circuit with a single expval measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.expval(qml.PauliZ(0))], shots=10000)
        result = simulate(qs)

        assert isinstance(result, np.float64)
        assert result.shape == ()
        assert np.allclose(result, np.cos(x), atol=0.1)

    def test_single_probs(self):
        """Test a simple circuit with a single prob measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.probs(wires=0)], shots=10000)
        result = simulate(qs)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.allclose(result, [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2], atol=0.1)

    def test_single_sample(self):
        """Test a simple circuit with a single sample measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=10000)
        result = simulate(qs)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10000, 2)
        assert np.allclose(
            np.sum(result, axis=0).astype(np.float32) / 10000, [np.sin(x / 2) ** 2, 0], atol=0.1
        )

    def test_multi_measurements(self):
        """Test a simple circuit containing multiple measurements"""
        x, y = np.array(0.732), np.array(0.488)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
            [qml.expval(qml.PauliZ(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
            shots=10000,
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], np.float64)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)

        assert np.allclose(result[0], np.cos(x), atol=0.1)

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

    def test_shots_reuse(self, mocker):
        """Test that samples are reused when two measurements commute"""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1])]
        mps = [
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliX(1)),
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliX(1)),
            qml.var(qml.PauliY(0)),
            qml.probs(wires=[0]),
            qml.probs(wires=[0, 1]),
            qml.sample(wires=[0, 1]),
            qml.expval(
                qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(1)])
            ),
            qml.expval(qml.sum(qml.PauliX(0), qml.PauliZ(1), qml.PauliY(1))),
            qml.expval(qml.s_prod(2.0, qml.PauliX(0))),
            qml.expval(qml.prod(qml.PauliX(0), qml.PauliY(1))),
        ]

        qs = qml.tape.QuantumScript(ops, mps, shots=100)

        spy = mocker.spy(qml.devices.qubit.sampling, "sample_state")
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(mps)

        # check that samples are reused when possible
        # 3 groups for expval and var, 1 group for probs and sample, 2 groups each for
        # Hamiltonian and Sum, and 1 group each for SProd and Prod
        assert spy.call_count == 10

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
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, np.float64) for res in result)
        assert all(res.shape == () for res in result)
        assert all(np.allclose(res, np.cos(x), atol=0.1) for res in result)

    @pytest.mark.parametrize("shots", shots_data)
    def test_probs_shot_vector(self, shots):
        """Test a simple circuit with a single prob measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.probs(wires=0)], shots=shots)
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, np.ndarray) for res in result)
        assert all(res.shape == (2,) for res in result)
        assert all(
            np.allclose(res, [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2], atol=0.1) for res in result
        )

    @pytest.mark.parametrize("shots", shots_data)
    def test_sample_shot_vector(self, shots):
        """Test a simple circuit with a single sample measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=shots)
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, np.ndarray) for res in result)
        assert all(res.shape == (s, 2) for res, s in zip(result, shots))
        assert all(
            np.allclose(
                np.sum(res, axis=0).astype(np.float32) / s, [np.sin(x / 2) ** 2, 0], atol=0.1
            )
            for res, s in zip(result, shots)
        )

    @pytest.mark.parametrize("shots", shots_data)
    def test_multi_measurement_shot_vector(self, shots):
        """Test a simple circuit containing multiple measurements for shot vectors"""
        x, y = np.array(0.732), np.array(0.488)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
            [qml.expval(qml.PauliZ(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
            shots=shots,
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        for shot_res, s in zip(result, shots):
            assert isinstance(shot_res, tuple)
            assert len(shot_res) == 3

            assert isinstance(shot_res[0], np.float64)
            assert isinstance(shot_res[1], np.ndarray)
            assert isinstance(shot_res[2], np.ndarray)

            assert np.allclose(shot_res[0], np.cos(x), atol=0.1)

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

    def test_custom_wire_labels(self):
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
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], np.float64)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)

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


class TestOperatorArithmetic:
    def test_numpy_op_arithmetic(self):
        """Test an operator arithmetic circuit with non-integer wires with numpy."""
        phi = 1.2
        ops = [
            qml.PauliX("a"),
            qml.PauliX("b"),
            qml.ctrl(qml.RX(phi, "target") ** 2, ("a", "b", -3), control_values=[1, 1, 0]),
        ]

        qs = qml.tape.QuantumScript(
            ops,
            [
                qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
            ],
        )

        results = simulate(qs)
        assert qml.math.allclose(results[0], -np.sin(2 * phi) - 1)
        assert qml.math.allclose(results[1], 3 * np.cos(2 * phi))

    @pytest.mark.autograd
    def test_autograd_op_arithmetic(self):
        """Test operator arithmetic circuit with non-integer wires works with autograd."""

        phi = qml.numpy.array(1.2)

        def f(x):
            ops = [
                qml.PauliX("a"),
                qml.PauliX("b"),
                qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
            ]

            qs = qml.tape.QuantumScript(
                ops,
                [
                    qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                    qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
                ],
            )

            return qml.numpy.array(simulate(qs))

        results = f(phi)
        assert qml.math.allclose(results[0], -np.sin(phi) - 1)
        assert qml.math.allclose(results[1], 3 * np.cos(phi))

        g = qml.jacobian(f)(phi)
        assert qml.math.allclose(g[0], -np.cos(phi))
        assert qml.math.allclose(g[1], -3 * np.sin(phi))

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_op_arithmetic(self, use_jit):
        """Test operator arithmetic circuit with non-integer wires works with jax."""
        import jax

        phi = jax.numpy.array(1.2)

        def f(x):
            ops = [
                qml.PauliX("a"),
                qml.PauliX("b"),
                qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
            ]

            qs = qml.tape.QuantumScript(
                ops,
                [
                    qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                    qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
                ],
            )

            return simulate(qs)

        if use_jit:
            f = jax.jit(f)

        results = f(phi)
        assert qml.math.allclose(results[0], -np.sin(phi) - 1)
        assert qml.math.allclose(results[1], 3 * np.cos(phi))

        g = jax.jacobian(f)(phi)
        assert qml.math.allclose(g[0], -np.cos(phi))
        assert qml.math.allclose(g[1], -3 * np.sin(phi))

    @pytest.mark.torch
    def test_torch_op_arithmetic(self):
        """Test operator arithmetic circuit with non-integer wires works with torch."""
        import torch

        phi = torch.tensor(-0.7290, requires_grad=True)

        def f(x):
            ops = [
                qml.PauliX("a"),
                qml.PauliX("b"),
                qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
            ]

            qs = qml.tape.QuantumScript(
                ops,
                [
                    qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                    qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
                ],
            )

            return simulate(qs)

        results = f(phi)
        assert qml.math.allclose(results[0], -torch.sin(phi) - 1)
        assert qml.math.allclose(results[1], 3 * torch.cos(phi))

        g = torch.autograd.functional.jacobian(f, phi)
        assert qml.math.allclose(g[0], -torch.cos(phi))
        assert qml.math.allclose(g[1], -3 * torch.sin(phi))

    @pytest.mark.tf
    def test_tensorflow_op_arithmetic(self):
        """Test operator arithmetic circuit with non-integer wires works with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(0.4203)

        def f(x):
            ops = [
                qml.PauliX("a"),
                qml.PauliX("b"),
                qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
            ]

            qs = qml.tape.QuantumScript(
                ops,
                [
                    qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                    qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
                ],
            )

            return simulate(qs)

        with tf.GradientTape(persistent=True) as tape:
            results = f(phi)

        assert qml.math.allclose(results[0], -np.sin(phi) - 1)
        assert qml.math.allclose(results[1], 3 * np.cos(phi))

        g0 = tape.gradient(results[0], phi)
        assert qml.math.allclose(g0, -np.cos(phi))
        g1 = tape.gradient(results[1], phi)
        assert qml.math.allclose(g1, -3 * np.sin(phi))


class TestQInfoMeasurements:
    measurements = [
        qml.density_matrix(0),
        qml.density_matrix(1),
        qml.density_matrix((0, 1)),
        qml.vn_entropy(0),
        qml.vn_entropy(1),
        qml.mutual_info(0, 1),
    ]

    def expected_results(self, phi):
        density_i = np.array([[np.cos(phi / 2) ** 2, 0], [0, np.sin(phi / 2) ** 2]])
        density_both = np.array(
            [
                [np.cos(phi / 2) ** 2, 0, 0, 0.0 + np.sin(phi) * 0.5j],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.0 - np.sin(phi) * 0.5j, 0, 0, np.sin(phi / 2) ** 2],
            ]
        )
        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]
        rho_log_rho = eigs * np.log(eigs)
        expected_entropy = -np.sum(rho_log_rho)
        mutual_info = 2 * expected_entropy

        return (density_i, density_i, density_both, expected_entropy, expected_entropy, mutual_info)

    def calculate_entropy_grad(self, phi):
        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = np.maximum(eigs, 1e-08)

        return -(
            (np.log(eigs[0]) + 1)
            * (np.sin(phi / 2) ** 3 * np.cos(phi / 2) - np.sin(phi / 2) * np.cos(phi / 2) ** 3)
            / np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)
        ) - (
            (np.log(eigs[1]) + 1)
            * (np.sin(phi / 2) * np.cos(phi / 2) * (np.cos(phi / 2) ** 2 - np.sin(phi / 2) ** 2))
            / np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)
        )

    def expected_grad(self, phi):
        p_2 = phi / 2
        g_density_i = np.array([[-np.cos(p_2) * np.sin(p_2), 0], [0, np.sin(p_2) * np.cos(p_2)]])
        g_density_both = np.array(
            [
                [-np.cos(p_2) * np.sin(p_2), 0, 0, 0.0 + 0.5j * np.cos(phi)],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.0 - 0.5j * np.cos(phi), 0, 0, np.sin(p_2) * np.cos(p_2)],
            ]
        )
        g_entropy = self.calculate_entropy_grad(phi)
        g_mutual_info = 2 * g_entropy
        return (g_density_i, g_density_i, g_density_both, g_entropy, g_entropy, g_mutual_info)

    def test_qinfo_numpy(self):
        """Test quantum info measurements with numpy"""
        phi = -0.623
        qs = qml.tape.QuantumScript([qml.IsingXX(phi, wires=(0, 1))], self.measurements)

        results = simulate(qs)
        for val1, val2 in zip(results, self.expected_results(phi)):
            assert qml.math.allclose(val1, val2)

    @pytest.mark.autograd
    def test_qinfo_autograd_execute(self):
        """Tests autograd execution with qinfo measurements."""
        phi = qml.numpy.array(6.920)

        def f(x):
            qs = qml.tape.QuantumScript([qml.IsingXX(x, wires=(0, 1))], self.measurements)
            return simulate(qs)

        results = f(phi)
        expected = self.expected_results(phi)
        for val1, val2 in zip(results, expected):
            assert qml.math.allclose(val1, val2)

    @pytest.mark.autograd
    def test_qinfo_autograd_backprop(self):
        """Tests autograd derivatives with qinfo measurements.
        Only simulates one measurement at a time due to autograd differentiation limitations."""

        phi = qml.numpy.array(0.263)

        def f(x, m_ind, real=True):
            qs = qml.tape.QuantumScript([qml.IsingXX(x, wires=(0, 1))], [self.measurements[m_ind]])
            out = simulate(qs)
            return qml.math.real(out) if real else qml.math.imag(out)

        expected_grads = self.expected_grad(phi)

        for i in (0, 1, 3, 4, 5):  # skip density matrix on both wires
            g = qml.jacobian(f)(phi, i, real=True)
            assert qml.math.allclose(expected_grads[i], g)

        # density matrix on both wires case
        g_real = qml.jacobian(f)(phi, 2, True)
        g_imag = qml.jacobian(f)(phi, 2, False)
        assert qml.math.allclose(g_real + 1j * g_imag, expected_grads[2])

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_qinfo_jax(self, use_jit):
        """Test qinfo meausrements work with jax and jitting."""

        import jax

        def f(x):
            qs = qml.tape.QuantumScript([qml.IsingXX(x, wires=(0, 1))], self.measurements)
            return simulate(qs)

        if use_jit:
            f = jax.jit(f)

        phi = jax.numpy.array(-0.792)

        results = f(phi)
        for val1, val2 in zip(results, self.expected_results(phi)):
            assert qml.math.allclose(val1, val2)

        def real_out(phi):
            return tuple(jax.numpy.real(r) for r in f(phi))

        grad_real = jax.jacobian(real_out)(phi)

        def imag_out(phi):
            return tuple(jax.numpy.imag(r) for r in f(phi))

        grad_imag = jax.jacobian(imag_out)(phi)
        grads = tuple(r + 1j * i for r, i in zip(grad_real, grad_imag))
        expected_grads = self.expected_grad(phi)

        # Writing this way makes it easier to figure out which is failing
        # density 0
        assert qml.math.allclose(grads[0], expected_grads[0])
        # density 1
        assert qml.math.allclose(grads[1], expected_grads[1])
        # density both
        assert qml.math.allclose(grads[2], expected_grads[2])
        # entropy 0
        assert qml.math.allclose(grads[3], expected_grads[3])
        # entropy 1
        assert qml.math.allclose(grads[4], expected_grads[4])
        # mutual info
        assert qml.math.allclose(grads[5], expected_grads[5])

    @pytest.mark.torch
    def test_qinfo_torch(self):
        """Test qinfo measurements with torch."""
        import torch

        phi = torch.tensor(1.928, requires_grad=True)

        def f(x):
            qs = qml.tape.QuantumScript([qml.IsingXX(x, wires=(0, 1))], self.measurements)
            return simulate(qs)

        results = f(phi)
        for val1, val2 in zip(results, self.expected_results(phi.detach().numpy())):
            assert qml.math.allclose(val1, val2)

        def real_out(phi):
            return tuple(torch.real(r) for r in f(torch.real(phi)))

        grad_real = torch.autograd.functional.jacobian(real_out, phi)

        def imag_out(phi):
            return tuple(torch.imag(r) if torch.is_complex(r) else torch.tensor(0) for r in f(phi))

        grad_imag = torch.autograd.functional.jacobian(imag_out, phi)
        grads = tuple(r + 1j * i for r, i in zip(grad_real, grad_imag))

        expected_grads = self.expected_grad(phi.detach().numpy())

        # Writing this way makes it easier to figure out which is failing
        # density 0
        assert qml.math.allclose(grads[0], expected_grads[0])
        # density 1
        assert qml.math.allclose(grads[1], expected_grads[1])
        # density both
        assert qml.math.allclose(grads[2], expected_grads[2])
        # entropy 0
        assert qml.math.allclose(grads[3], expected_grads[3])
        # entropy 1
        assert qml.math.allclose(grads[4], expected_grads[4])
        # mutual info
        assert qml.math.allclose(grads[5], expected_grads[5])

    @pytest.mark.tf
    def test_qinfo_tf(self):
        """Test qinfo measurements with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(0.9276)

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = qml.tape.QuantumScript([qml.IsingXX(phi, wires=(0, 1))], self.measurements)
            results = simulate(qs)

            result2_real = tf.math.real(results[2])
            result2_imag = tf.math.imag(results[2])

        expected = self.expected_results(phi)
        for val1, val2 in zip(results, expected):
            assert qml.math.allclose(val1, val2)

        expected_grads = self.expected_grad(phi)

        grad0 = grad_tape.jacobian(results[0], phi)
        assert qml.math.allclose(grad0, expected_grads[0])

        grad1 = grad_tape.jacobian(results[1], phi)
        assert qml.math.allclose(grad1, expected_grads[1])

        grad2_real = grad_tape.jacobian(result2_real, phi)
        grad2_imag = grad_tape.jacobian(result2_imag, phi)
        assert qml.math.allclose(
            np.array(grad2_real) + 1j * np.array(grad2_imag), expected_grads[2]
        )

        grad3 = grad_tape.jacobian(results[3], phi)
        assert qml.math.allclose(grad3, expected_grads[3])

        grad4 = grad_tape.jacobian(results[4], phi)
        assert qml.math.allclose(grad4, expected_grads[4])

        grad5 = grad_tape.jacobian(results[5], phi)
        assert qml.math.allclose(grad5, expected_grads[5])
