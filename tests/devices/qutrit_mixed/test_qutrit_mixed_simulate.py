# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for simulate in devices/qutrit_mixed."""

import pytest
import numpy as np
import pennylane as qml
from pennylane import math

# TODO change when added to __init__
from pennylane.devices.qutrit_mixed.simulate import simulate, get_final_state, measure_final_state


def expected_TRX_circ_expval_values(phi, subspace):
    """Find the expect-values of GellManns 2,3,5,8
    on a circuit with a TRX gate"""
    if subspace == (0, 1):
        gellmann_2 = -np.sin(phi)
        gellmann_3 = np.cos(phi)
        gellmann_5 = 0
        gellmann_8 = np.sqrt(1 / 3)
    if subspace == (0, 2):
        gellmann_2 = 0
        gellmann_3 = np.cos(phi / 2) ** 2
        gellmann_5 = -np.sin(phi)
        gellmann_8 = np.sqrt(1 / 3) * (np.cos(phi) - np.sin(phi / 2) ** 2)
    return np.array([gellmann_2, gellmann_3, gellmann_5, gellmann_8])


def expected_TRX_circ_expval_jacobians(phi, subspace):
    """Find the Jacobians of expect-values of GellManns 2,3,5,8
    on a circuit with a TRX gate"""
    if subspace == (0, 1):
        gellmann_2 = -np.cos(phi)
        gellmann_3 = -np.sin(phi)
        gellmann_5 = 0
        gellmann_8 = 0
    if subspace == (0, 2):
        gellmann_2 = 0
        gellmann_3 = -np.sin(phi) / 2
        gellmann_5 = -np.cos(phi)
        gellmann_8 = np.sqrt(1 / 3) * -(1.5 * np.sin(phi))
    return np.array([gellmann_2, gellmann_3, gellmann_5, gellmann_8])


def expected_TRX_circ_state(phi, subspace):
    """Find the state after applying TRX gate on |0>"""
    expected_vector = np.array([0, 0, 0], dtype=complex)
    expected_vector[subspace[0]] = np.cos(phi / 2)
    expected_vector[subspace[1]] = -1j * np.sin(phi / 2)
    return np.outer(expected_vector, np.conj(expected_vector))


class TestCurrentlyUnsupportedCases:
    """Test currently unsupported cases, such as sampling expval or samples without shots"""

    # pylint: disable=too-few-public-methods
    def test_sample_based_observable(self):
        """Test sample-only measurements raise a NotImplementedError."""

        qs = qml.tape.QuantumScript(measurements=[qml.sample(wires=0)])
        with pytest.raises(NotImplementedError):
            simulate(qs)

    @pytest.mark.parametrize("mp", [qml.probs(0), qml.probs(op=qml.GellMann(0, 2))])
    def test_invalid_samples(self, mp):
        """Test Sampling MeasurementProcesses that are currently unsupported on this device"""
        qs = qml.tape.QuantumScript(ops=[qml.TAdd(wires=(0, 1))], measurements=[mp], shots=10)
        with pytest.raises(NotImplementedError):
            simulate(qs)


def test_custom_operation():
    """Test execution works with a manually defined operator if it has a matrix."""

    # pylint: disable=too-few-public-methods
    class MyOperator(qml.operation.Operator):
        num_wires = 1

        @staticmethod
        def compute_matrix():
            return np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    qs = qml.tape.QuantumScript([MyOperator(0)], [qml.expval(qml.GellMann(0, 8))])

    result = simulate(qs)
    assert qml.math.allclose(result, -np.sqrt(4 / 3))


class TestStatePrepBase:
    """Tests integration with various state prep methods."""

    def test_basis_state(self):
        """Test that the BasisState operator prepares the desired state."""
        qs = qml.tape.QuantumScript(
            ops=[qml.QutritBasisState((2, 1), wires=(0, 1))],
            measurements=[qml.probs(wires=(0, 1, 2))],
        )
        probs = simulate(qs)
        expected = np.zeros(27)
        expected[21] = 1.0
        assert qml.math.allclose(probs, expected)


class TestBasicCircuit:
    """Tests a basic circuit with one rx gate and two simple expectation values."""

    @staticmethod
    def get_TRX_quantum_script(phi, subspace):
        """Get the quantum script where TRX is applied then GellMann observables are measured"""
        ops = [qml.TRX(phi, wires=0, subspace=subspace)]
        obs = [
            qml.expval(qml.GellMann(0, 2)),
            qml.expval(qml.GellMann(0, 3)),
            qml.expval(qml.GellMann(0, 5)),
            qml.expval(qml.GellMann(0, 8)),
        ]
        return qml.tape.QuantumScript(ops, obs)

    @pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
    def test_basic_circuit_numpy(self, subspace):
        """Test execution with a basic circuit."""

        phi = np.array(0.397)

        qs = self.get_TRX_quantum_script(phi, subspace)
        result = simulate(qs)

        expected_measurements = expected_TRX_circ_expval_values(phi, subspace)
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert np.allclose(result, expected_measurements)

        state, is_state_batched = get_final_state(qs)
        result = measure_final_state(qs, state, is_state_batched)

        expected_state = expected_TRX_circ_state(phi, subspace)

        assert np.allclose(state, expected_state)
        assert not is_state_batched

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert np.allclose(result, expected_measurements)

    @pytest.mark.autograd
    @pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
    def test_autograd_results_and_backprop(self, subspace):
        """Tests execution and gradients with autograd"""

        phi = qml.numpy.array(-0.52)

        def f(x):
            qs = self.get_TRX_quantum_script(x, subspace)
            return qml.numpy.array(simulate(qs))

        result = f(phi)
        expected = expected_TRX_circ_expval_values(phi, subspace)
        assert qml.math.allclose(result, expected)

        g = qml.jacobian(f)(phi)
        expected = expected_TRX_circ_expval_jacobians(phi, subspace)
        assert qml.math.allclose(g, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    @pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
    def test_jax_results_and_backprop(self, use_jit, subspace):
        """Tests exeuction and gradients with jax."""

        import jax

        phi = jax.numpy.array(0.678)

        def f(x):
            qs = self.get_TRX_quantum_script(x, subspace)
            return simulate(qs)

        if use_jit:
            f = jax.jit(f)

        result = f(phi)
        expected = expected_TRX_circ_expval_values(phi, subspace)
        assert qml.math.allclose(result, expected)

        g = jax.jacobian(f)(phi)
        expected = expected_TRX_circ_expval_jacobians(phi, subspace)
        assert qml.math.allclose(g, expected)

    @pytest.mark.torch
    @pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
    def test_torch_results_and_backprop(self, subspace):
        """Tests execution and gradients of a simple circuit with torch."""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        def f(x):
            qs = self.get_TRX_quantum_script(x, subspace)
            return simulate(qs)

        result = f(phi)
        expected = expected_TRX_circ_expval_values(phi.detach().numpy(), subspace)
        assert qml.math.allclose(result[0], expected[0])
        assert qml.math.allclose(result[1], expected[1])
        assert qml.math.allclose(result[2], expected[2])
        assert qml.math.allclose(result[3], expected[3])

        g = torch.autograd.functional.jacobian(f, phi + 0j)
        expected = expected_TRX_circ_expval_jacobians(phi.detach().numpy(), subspace)
        assert qml.math.allclose(g[0], expected[0])
        assert qml.math.allclose(g[1], expected[1])
        assert qml.math.allclose(g[2], expected[2])
        assert qml.math.allclose(g[3], expected[3])

    @pytest.mark.tf
    @pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
    def test_tf_results_and_backprop(self, subspace):
        """Tests execution and gradients of a simple circuit with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = self.get_TRX_quantum_script(phi, subspace)
            result = simulate(qs)

        expected = expected_TRX_circ_expval_values(phi, subspace)
        assert qml.math.allclose(result, expected)

        grad0 = grad_tape.jacobian(result[0], [phi])
        grad1 = grad_tape.jacobian(result[1], [phi])
        grad2 = grad_tape.jacobian(result[2], [phi])
        grad3 = grad_tape.jacobian(result[3], [phi])

        expected = expected_TRX_circ_expval_jacobians(phi, subspace)
        assert qml.math.allclose(grad0[0], expected[0])
        assert qml.math.allclose(grad1[0], expected[1])
        assert qml.math.allclose(grad2[0], expected[2])
        assert qml.math.allclose(grad3[0], expected[3])

    @pytest.mark.jax
    @pytest.mark.parametrize("op", [qml.TRX(np.pi, 0), qml.QutritBasisState([1], 0)])
    def test_result_has_correct_interface(self, op):
        """Test that even if no interface parameters are given, result is correct."""

        qs = qml.tape.QuantumScript([op], [qml.expval(qml.GellMann(0, 3))])
        res = simulate(qs, interface="jax")

        assert qml.math.get_interface(res) == "jax"
        assert qml.math.allclose(res, -1)


@pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
class TestBroadcasting:
    """Test that simulate works with broadcasted parameters"""

    @staticmethod
    def get_state(x, subspace):
        state = []
        for x_val in x:
            vec = np.zeros(9, dtype=complex)
            vec[subspace[1]] = -1j * np.cos(x_val / 2)
            vec[3 + (2 * subspace[1])] = -1j * np.sin(x_val / 2)
            state.append(np.outer(vec, np.conj(vec)).reshape((3,) * 4))
        return state

    @staticmethod
    def get_expectation_values(x, subspace):
        if subspace == (0, 1):
            return [np.cos(x), -np.cos(x / 2) ** 2]
        return [np.cos(x / 2) ** 2, -np.sin(x / 2) ** 2]

    def test_broadcasted_op_state(self, subspace):
        """Test that simulate works for state measurements
        when an operation has broadcasted parameters"""
        x = np.array([0.8, 1.0, 1.2, 1.4])
        ops = [
            qml.TRX(np.pi, wires=1, subspace=subspace),
            qml.TRY(x, wires=0, subspace=subspace),
            qml.TAdd(wires=[0, 1]),
        ]
        measurements = [qml.expval(qml.GellMann(i, 3)) for i in range(2)]

        qs = qml.tape.QuantumScript(ops, measurements)
        res = simulate(qs)

        expected = self.get_expectation_values(x, subspace)
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res, expected)

        state, is_state_batched = get_final_state(qs)
        res = measure_final_state(qs, state, is_state_batched)

        assert np.allclose(state, self.get_state(x, subspace))
        assert is_state_batched
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res, expected)

    def test_broadcasted_op_sample(self, subspace):
        """Test that simulate works for sample measurements
        when an operation has broadcasted parameters"""
        x = np.array([0.8, 1.0, 1.2, 1.4])

        ops = [
            qml.TRX(np.pi, wires=1, subspace=subspace),
            qml.TRY(x, wires=0, subspace=subspace),
            qml.TAdd(wires=[0, 1]),
        ]
        measurements = [qml.expval(qml.GellMann(i, 3)) for i in range(2)]

        qs = qml.tape.QuantumScript(ops, measurements, shots=qml.measurements.Shots(10000))
        res = simulate(qs, rng=123)

        expected = self.get_expectation_values(x, subspace)
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res, expected, atol=0.05)

        state, is_state_batched = get_final_state(qs)
        res = measure_final_state(qs, state, is_state_batched, rng=123)

        assert np.allclose(state, self.get_state(x, subspace))
        assert is_state_batched
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res, expected, atol=0.05)

    def test_broadcasting_with_extra_measurement_wires(self, mocker, subspace):
        """Test that broadcasting works when the operations don't act on all wires."""
        # I can't mock anything in `simulate` because the module name is the function name
        spy = mocker.spy(qml, "map_wires")
        x = np.array([0.8, 1.0, 1.2, 1.4])

        ops = [
            qml.TRX(np.pi, wires=2, subspace=subspace),
            qml.TRY(x, wires=1, subspace=subspace),
            qml.TAdd(wires=[1, 2]),
        ]
        measurements = [qml.expval(qml.GellMann(i, 3)) for i in range(3)]

        qs = qml.tape.QuantumScript(ops, measurements)
        res = simulate(qs)

        assert isinstance(res, tuple)
        assert len(res) == 3
        assert np.allclose(res[0], 1.0)
        assert np.allclose(res[1:], self.get_expectation_values(x, subspace))
        assert spy.call_args_list[0].args == (qs, {2: 0, 1: 1, 0: 2})


@pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
class TestDebugger:
    """Tests that the debugger works for a simple circuit"""

    class Debugger:
        """A dummy debugger class"""

        def __init__(self):
            self.active = True
            self.snapshots = {}

    basis_state = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])

    @staticmethod
    def get_debugger_quantum_script(phi, subspace):
        """Get the quantum script with debugging where TRX is applied
        then GellMann observables are measured"""
        ops = [
            qml.Snapshot(),
            qml.TRX(phi, wires=0, subspace=subspace),
            qml.Snapshot("final_state"),
        ]
        obs = [
            qml.expval(qml.GellMann(0, 2)),
            qml.expval(qml.GellMann(0, 3)),
            qml.expval(qml.GellMann(0, 5)),
            qml.expval(qml.GellMann(0, 8)),
        ]
        return qml.tape.QuantumScript(ops, obs)

    def test_debugger_numpy(self, subspace):
        """Test debugger with numpy"""
        phi = np.array(0.397)
        qs = self.get_debugger_quantum_script(phi, subspace)
        debugger = self.Debugger()
        result = simulate(qs, debugger=debugger)

        assert isinstance(result, tuple)
        assert len(result) == 4

        expected = expected_TRX_circ_expval_values(phi, subspace)
        assert np.allclose(result, expected)

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert np.allclose(debugger.snapshots[0], self.basis_state)

        expected_final_state = expected_TRX_circ_state(phi, subspace)
        assert np.allclose(debugger.snapshots["final_state"], expected_final_state)

    @pytest.mark.autograd
    def test_debugger_autograd(self, subspace):
        """Tests debugger with autograd"""
        phi = qml.numpy.array(-0.52)
        debugger = self.Debugger()

        def f(x):
            qs = self.get_debugger_quantum_script(x, subspace)
            return qml.numpy.array(simulate(qs, debugger=debugger))

        result = f(phi)
        expected = expected_TRX_circ_expval_values(phi, subspace)
        assert qml.math.allclose(result, expected)

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], self.basis_state)

        expected_final_state = expected_TRX_circ_state(phi, subspace)
        assert qml.math.allclose(debugger.snapshots["final_state"], expected_final_state)

    @pytest.mark.jax
    def test_debugger_jax(self, subspace):
        """Tests debugger with JAX"""
        import jax

        phi = jax.numpy.array(0.678)
        debugger = self.Debugger()

        def f(x):
            qs = self.get_debugger_quantum_script(x, subspace)
            return simulate(qs, debugger=debugger)

        result = f(phi)
        expected = expected_TRX_circ_expval_values(phi, subspace)
        assert qml.math.allclose(result, expected)

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], self.basis_state)

        expected_final_state = expected_TRX_circ_state(phi, subspace)
        assert qml.math.allclose(debugger.snapshots["final_state"], expected_final_state)

    @pytest.mark.torch
    def test_debugger_torch(self, subspace):
        """Tests debugger with torch"""
        import torch

        phi = torch.tensor(-0.526, requires_grad=True)
        debugger = self.Debugger()

        def f(x):
            qs = self.get_debugger_quantum_script(x, subspace)
            return simulate(qs, debugger=debugger)

        results = f(phi)
        expected_values = expected_TRX_circ_expval_values(phi.detach().numpy(), subspace)
        for result, expected in zip(results, expected_values):
            assert qml.math.allclose(result, expected)

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], self.basis_state)

        expected_final_state = math.asarray(
            expected_TRX_circ_state(phi.detach().numpy(), subspace), like="torch"
        )
        assert qml.math.allclose(debugger.snapshots["final_state"], expected_final_state)

    # pylint: disable=invalid-unary-operand-type
    @pytest.mark.tf
    def test_debugger_tf(self, subspace):
        """Tests debugger with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)
        debugger = self.Debugger()

        qs = self.get_debugger_quantum_script(phi, subspace)
        result = simulate(qs, debugger=debugger)

        expected = expected_TRX_circ_expval_values(phi, subspace)
        assert qml.math.allclose(result, expected)

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], self.basis_state)

        expected_final_state = expected_TRX_circ_state(phi, subspace)
        assert qml.math.allclose(debugger.snapshots["final_state"], expected_final_state)


@pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
class TestSampleMeasurements:
    """Tests circuits with sample-based measurements"""

    @staticmethod
    def expval_of_TRY_circ(x, subspace):
        """Find the expval of GellMann_3 on simple TRY circuit"""
        if subspace[1] == 1:
            return np.cos(x)
        return np.cos(x / 2) ** 2

    @staticmethod
    def sample_sum_of_TRY_circ(x, subspace):
        """Find the expval of computational basis for both wires on simple TRY circuit"""
        if subspace[1] == 1:
            return [np.sin(x / 2) ** 2, 0]
        return [2 * np.sin(x / 2) ** 2, 0]

    @staticmethod
    def expval_of_2_qutrit_circ(x, subspace):
        """expval of GellMann_3 on wire=0 on the 2 qutrit circuit used"""
        if subspace[1] == 1:
            return np.cos(x)
        return np.cos(x / 2) ** 2

    @staticmethod
    def probs_of_2_qutrit_circ(x, y, subspace):
        """possible measurement values and probabilityies for the 2 qutrit circuit used"""
        probs = np.array(
            [
                np.cos(x / 2) * np.cos(y / 2),
                np.cos(x / 2) * np.sin(y / 2),
                np.sin(x / 2) * np.sin(y / 2),
                np.sin(x / 2) * np.cos(y / 2),
            ]
        )
        probs **= 2
        if subspace[1] == 1:
            keys = ["00", "01", "10", "11"]
        else:
            keys = ["00", "02", "20", "22"]
        return keys, probs

    def test_single_expval(self, subspace):
        """Test a simple circuit with a single expval measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript(
            [qml.TRY(x, wires=0, subspace=subspace)],
            [qml.expval(qml.GellMann(0, 3))],
            shots=1000000,
        )
        result = simulate(qs)
        assert isinstance(result, np.float64)
        assert result.shape == ()
        assert np.allclose(result, self.expval_of_TRY_circ(x, subspace), atol=0.1)

    def test_single_sample(self, subspace):
        """Test a simple circuit with a single sample measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript(
            [qml.TRY(x, wires=0, subspace=subspace)], [qml.sample(wires=range(2))], shots=10000
        )
        result = simulate(qs)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10000, 2)
        assert np.allclose(
            np.sum(result, axis=0).astype(np.float32) / 10000,
            self.sample_sum_of_TRY_circ(x, subspace),
            atol=0.1,
        )

    def test_multi_measurements(self, subspace):
        """Test a simple circuit containing multiple measurements"""
        num_shots = 10000
        x, y = np.array(0.732), np.array(0.488)
        qs = qml.tape.QuantumScript(
            [
                qml.TRX(x, wires=0, subspace=subspace),
                qml.TAdd(wires=[0, 1]),
                qml.TRY(y, wires=1, subspace=subspace),
            ],
            [
                qml.expval(qml.GellMann(0, 3)),
                qml.counts(wires=range(2)),
                qml.sample(wires=range(2)),
            ],
            shots=num_shots,
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == 3

        assert np.allclose(result[0], self.expval_of_2_qutrit_circ(x, subspace), atol=0.1)

        expected_keys, expected_probs = self.probs_of_2_qutrit_circ(x, y, subspace)
        assert list(result[1].keys()) == expected_keys
        assert np.allclose(
            np.array(list(result[1].values())) / num_shots,
            expected_probs,
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
    def test_expval_shot_vector(self, shots, subspace):
        """Test a simple circuit with a single expval measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript(
            [qml.TRY(x, wires=0, subspace=subspace)], [qml.expval(qml.GellMann(0, 3))], shots=shots
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        expected = self.expval_of_TRY_circ(x, subspace)
        assert all(isinstance(res, np.float64) for res in result)
        assert all(res.shape == () for res in result)
        assert all(np.allclose(res, expected, atol=0.1) for res in result)

    @pytest.mark.parametrize("shots", shots_data)
    def test_sample_shot_vector(self, shots, subspace):
        """Test a simple circuit with a single sample measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript(
            [qml.TRY(x, wires=0, subspace=subspace)], [qml.sample(wires=range(2))], shots=shots
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        expected = self.sample_sum_of_TRY_circ(x, subspace)
        assert all(isinstance(res, np.ndarray) for res in result)
        assert all(res.shape == (s, 2) for res, s in zip(result, shots))
        assert all(
            np.allclose(np.sum(res, axis=0).astype(np.float32) / s, expected, atol=0.1)
            for res, s in zip(result, shots)
        )

    @pytest.mark.parametrize("shots", shots_data)
    def test_multi_measurement_shot_vector(self, shots, subspace):
        """Test a simple circuit containing multiple measurements for shot vectors"""
        x, y = np.array(0.732), np.array(0.488)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript(
            [
                qml.TRX(x, wires=0, subspace=subspace),
                qml.TAdd(wires=[0, 1]),
                qml.TRY(y, wires=1, subspace=subspace),
            ],
            [
                qml.expval(qml.GellMann(0, 3)),
                qml.counts(wires=range(2)),
                qml.sample(wires=range(2)),
            ],
            shots=shots,
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        for shot_res, s in zip(result, shots):
            assert isinstance(shot_res, tuple)
            assert len(shot_res) == 3

            assert isinstance(shot_res[0], np.float64)
            assert isinstance(shot_res[1], dict)
            assert isinstance(shot_res[2], np.ndarray)

            assert np.allclose(shot_res[0], self.expval_of_TRY_circ(x, subspace), atol=0.1)

            expected_keys, expected_probs = self.probs_of_2_qutrit_circ(x, y, subspace)
            assert list(shot_res[1].keys()) == expected_keys
            assert np.allclose(
                np.array(list(shot_res[1].values())) / s,
                expected_probs,
                atol=0.1,
            )

            assert shot_res[2].shape == (s, 2)

    def test_custom_wire_labels(self, subspace):
        """Test that custom wire labels works as expected"""
        num_shots = 10000
        x, y = np.array(0.732), np.array(0.488)
        qs = qml.tape.QuantumScript(
            [
                qml.TRX(x, wires="b", subspace=subspace),
                qml.TAdd(wires=["b", "a"]),
                qml.TRY(y, wires="a", subspace=subspace),
            ],
            [
                qml.expval(qml.GellMann("b", 3)),
                qml.counts(wires=["a", "b"]),
                qml.sample(wires=["b", "a"]),
            ],
            shots=num_shots,
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], np.float64)
        assert isinstance(result[1], dict)
        assert isinstance(result[2], np.ndarray)

        assert np.allclose(result[0], self.expval_of_TRY_circ(x, subspace), atol=0.1)

        expected_keys, expected_probs = self.probs_of_2_qutrit_circ(x, y, subspace)
        assert list(result[1].keys()) == expected_keys
        assert np.allclose(
            np.array(list(result[1].values())) / num_shots,
            expected_probs,
            atol=0.1,
        )

        assert result[2].shape == (num_shots, 2)
