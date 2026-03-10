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
"""Unit tests for simulate in devices/qubit_mixed."""
import numpy as np
import pytest

import pennylane as qml
from pennylane import math
from pennylane.devices.qubit_mixed import get_final_state, measure_final_state, simulate

ml_interfaces = ["numpy", "autograd", "jax", "torch"]


# pylint: disable=too-few-public-methods
class TestResultInterface:
    """Test that the result interface is correct."""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "op", [qml.RX(np.pi, [0]), qml.BasisState(np.array([1, 1]), wires=range(2))]
    )
    @pytest.mark.parametrize("interface", ml_interfaces)
    def test_result_has_correct_interface(self, op, interface):
        """Test that even if no interface parameters are given, result is correct."""
        qs = qml.tape.QuantumScript([op], [qml.expval(qml.Z(0))])
        res = simulate(qs, interface=interface)

        assert qml.math.get_interface(res) == interface


# pylint: disable=too-few-public-methods
class TestStatePrepBase:
    """Tests integration with various state prep methods."""

    def test_basis_state(self):
        """Test that the BasisState operator prepares the desired state."""
        qs = qml.tape.QuantumScript(
            ops=[qml.BasisState(np.array([1, 1]), wires=[0, 1])],  # prod state |1, 1>
            measurements=[qml.probs(wires=[0, 1])],  # measure only the wires we prepare
        )
        probs = simulate(qs)

        # For state |1, 1>, only the |11> probability should be 1, others 0
        expected = np.zeros(4)
        expected[3] = 1.0  # |11> is the last basis state
        assert np.allclose(probs, expected)

    def test_basis_state_padding(self):
        """Test that the BasisState operator prepares the desired state, with actual wires larger than the initial."""
        qs = qml.tape.QuantumScript(
            ops=[qml.BasisState(np.array([1, 1]), wires=[0, 1])],  # prod state |1, 1>
            measurements=[qml.probs(wires=[0, 1, 2])],
        )
        probs = simulate(qs)
        expected = np.zeros(8)
        expected[6] = 1.0  # Should be |110> = |6>
        assert qml.math.allclose(probs, expected)

    def test_state_mp(self):
        """Test that the current two supported statemps are equivalent.
        This test ensure the fix for measurementprocess.raw_wires is working."""
        state = np.array([0, 1])
        device = qml.device("default.mixed", wires=[0, 1])

        @qml.qnode(device)
        def circuit0():
            qml.StatePrep(state, wires=[1])
            return qml.density_matrix(wires=[0, 1])

        @qml.qnode(device)
        def circuit():
            qml.StatePrep(state, wires=[1])
            return qml.state()

        dm0 = circuit0()
        dm = circuit()
        assert np.allclose(dm0, dm)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ml_interfaces)
    def test_state_prep_single_batch_size(self, interface):
        """Test a special case, when the state is of shape (1, n)"""
        state = np.zeros((1, 4))
        state[0, 0] = 1.0

        state = qml.math.asarray(state, like=interface)

        qs = qml.tape.QuantumScript(
            ops=[qml.StatePrep(state, wires=[0, 1]), qml.X(0)],
            measurements=[qml.expval(qml.Z(0)), qml.expval(qml.Z(1)), qml.state()],
        )
        # Dev Note: there used to be a shape-related bug that only appears in usage when you measure all wires.
        res, _, dm = simulate(qs)
        assert dm.shape == (1, 4, 4)
        assert np.allclose(res, -1.0)


@pytest.mark.parametrize("wires", [0, 1, 2])
class TestBasicCircuit:
    """Tests a basic circuit with one RX gate and a few simple expectation values."""

    @staticmethod
    def get_quantum_script(phi, wires):
        """Get the quantum script where RX is applied then observables are measured"""
        ops = [qml.RX(phi, wires=wires)]
        obs = [
            qml.expval(qml.PauliX(wires)),
            qml.expval(qml.PauliY(wires)),
            qml.expval(qml.PauliZ(wires)),
        ]
        return qml.tape.QuantumScript(ops, obs)

    def test_basic_circuit_numpy(self, wires):
        """Test execution with a basic circuit, only one wire."""
        phi = np.array(0.397)

        qs = self.get_quantum_script(phi, wires)
        result = simulate(qs)

        # After applying RX(phi) to |0⟩, the state becomes:
        # |ψ⟩ = cos(phi/2)|0⟩ - i sin(phi/2)|1⟩
        # The expectation values are calculated as:
        # ⟨X⟩ = ⟨ψ|X|ψ⟩ = 0
        # ⟨Y⟩ = ⟨ψ|Y|ψ⟩ = -sin(phi)
        # ⟨Z⟩ = ⟨ψ|Z|ψ⟩ = cos(phi)
        expected_measurements = (
            0,
            -np.sin(phi),
            np.cos(phi),
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert np.allclose(result, expected_measurements)

    @pytest.mark.autograd
    def test_autograd_results_and_backprop(self, wires):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        def f(x):
            qs = self.get_quantum_script(x, wires)
            return qml.numpy.array(simulate(qs))

        result = f(phi)
        expected = (0, -np.sin(phi), np.cos(phi))
        assert qml.math.allclose(result, expected)

        g = qml.jacobian(f)(phi)
        expected = (0, -np.cos(phi), -np.sin(phi))
        assert qml.math.allclose(g, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_results_and_backprop(self, use_jit, wires):
        """Tests execution and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        def f(x):
            qs = self.get_quantum_script(x, wires)
            return simulate(qs)

        if use_jit:
            f = jax.jit(f)

        result = f(phi)
        expected = (0, -np.sin(phi), np.cos(phi))
        assert qml.math.allclose(result, expected)

        g = jax.jacobian(f)(phi)
        expected = (0, -np.cos(phi), -np.sin(phi))
        assert qml.math.allclose(g, expected)

    @pytest.mark.torch
    def test_torch_results_and_backprop(self, wires):
        """Tests execution and gradients with torch."""
        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        def f(x):
            qs = self.get_quantum_script(x, wires)
            return simulate(qs)

        result = f(phi)
        expected = (0, -np.sin(phi.detach().numpy()), np.cos(phi.detach().numpy()))

        result_detached = math.asarray(result, like="torch").detach().numpy()
        assert math.allclose(result_detached, expected)

        # Convert complex jacobian to real and take only real part for comparison
        jacobian = math.asarray(torch.autograd.functional.jacobian(f, phi + 0j), like="torch")
        jacobian = jacobian.real if hasattr(jacobian, "real") else jacobian
        expected = (0, -np.cos(phi.detach().numpy()), -np.sin(phi.detach().numpy()))
        assert math.allclose(jacobian.detach().numpy(), expected)

    @pytest.mark.tf
    def test_tf_results_and_backprop(self, wires):
        """Tests execution and gradients with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = self.get_quantum_script(phi, wires)
            result = simulate(qs)

        expected = (0, -np.sin(float(phi)), np.cos(float(phi)))
        assert qml.math.allclose(result, expected)

        expected = (0, -np.cos(float(phi)), -np.sin(float(phi)))
        assert math.all(
            [
                math.allclose(grad_tape.jacobian(one_obs_result, [phi])[0], one_obs_expected)
                for one_obs_result, one_obs_expected in zip(result, expected)
            ]
        )

    def test_state_cache(self, wires):
        """Test that the state_cache parameter properly stores the final state when accounting for wire mapping."""
        phi = np.array(0.397)

        # Create a cache dictionary to store states
        state_cache = {}

        # Create and map the circuit to standard wires first
        qs = self.get_quantum_script(phi, wires)
        mapped_qs = qs.map_to_standard_wires()
        mapped_hash = mapped_qs.hash

        # Run the circuit with cache
        result1 = simulate(qs, state_cache=state_cache)

        # Verify the mapped circuit's hash is in the cache
        assert mapped_hash in state_cache, "Mapped circuit hash should be in cache"

        # Verify the cached state has correct shape
        cached_state = state_cache[mapped_hash]
        assert cached_state.shape == (2, 2), "Cached state should be 2x2 density matrix"

        # Run same circuit again and verify results are consistent
        result2 = simulate(qs, state_cache=state_cache)
        assert np.allclose(result1, result2)

        # Verify results match theoretical expectations
        expected = (0, -np.sin(phi), np.cos(phi))
        assert np.allclose(result1, expected)


class TestBroadcasting:
    """Test that simulate works with broadcasted parameters."""

    @staticmethod
    def get_expected_state(x):
        """Gets the expected final state of the circuit described in `get_ops_and_measurements`."""
        states = []
        for x_val in x:
            cos = np.cos(x_val / 2)
            sin = np.sin(x_val / 2)
            state = np.array([[cos**2, 0.5j * np.sin(x_val)], [-0.5j * np.sin(x_val), sin**2]])
            states.append(state)
        return np.stack(states)

    @staticmethod
    def get_quantum_script(x, wire=0, shots=None, extra_wire=False):
        """Gets quantum script of a circuit that includes parameter broadcasted operations and measurements."""

        ops = [qml.RX(x, wires=wire)]
        measurements = [qml.expval(qml.PauliY(wire)), qml.expval(qml.PauliZ(wire))]
        if extra_wire:
            # Add measurement on the last wire for the extra wire case
            measurements.insert(0, qml.expval(qml.PauliY(wire + 2)))

        return qml.tape.QuantumScript(ops, measurements, shots=shots)

    def test_broadcasted_op_state(self):
        """Test that simulate works for state measurements
        when an operation has broadcasted parameters"""
        x = np.array([0.8, 1.0, 1.2, 1.4])

        qs = self.get_quantum_script(x)
        res = simulate(qs)

        expected = [-np.sin(x), np.cos(x)]
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res, expected)

        state, is_state_batched = get_final_state(qs)
        res = measure_final_state(qs, state, is_state_batched)

        assert np.allclose(state, self.get_expected_state(x))
        assert is_state_batched
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res, expected)

    def test_broadcasting_with_extra_measurement_wires(self, mocker):
        """Test that broadcasting works when the operations don't act on all wires."""
        spy = mocker.spy(qml, "map_wires")
        x = np.array([0.8, 1.0, 1.2, 1.4])
        qs = self.get_quantum_script(x, extra_wire=True)
        res = simulate(qs)

        # Supoosed to be: three values, one for each measurement, see
        # `get_quantum_script`. Each value is a vector of length 4, same as the
        # length of x.
        assert isinstance(res, tuple)
        assert len(res) == 3
        assert np.allclose(res[0], np.zeros_like(x))
        assert np.allclose(res[1:], [-np.sin(x), np.cos(x)])
        # The mapping should be consistent with the wire ordering in get_quantum_script
        assert spy.call_args_list[0].args == (qs, {0: 0, 2: 1})


@pytest.mark.all_interfaces
class TestSampleMeasurements:
    """Tests circuits with sample-based measurements"""

    @staticmethod
    def expval_of_RY_circ(x):
        """Find the expval of PauliZ on simple RY circuit"""
        return np.cos(x)

    @staticmethod
    def sample_sum_of_RY_circ(x):
        """Find the expval of computational basis bitstring value for both wires on simple RY circuit"""
        return [np.sin(x / 2) ** 2, 0]

    @staticmethod
    def expval_of_2_qubit_circ(x):
        """Gets the expval of PauliZ on wire=0 on the 2-qubit circuit used"""
        return np.cos(x)

    @staticmethod
    def probs_of_2_qubit_circ(x, y):
        """Possible measurement values and probabilities for the 2-qubit circuit used"""
        probs = (
            np.array(
                [
                    np.cos(x / 2) * np.cos(y / 2),
                    np.cos(x / 2) * np.sin(y / 2),
                    np.sin(x / 2) * np.sin(y / 2),
                    np.sin(x / 2) * np.cos(y / 2),
                ]
            )
            ** 2
        )
        return ["00", "01", "10", "11"], probs

    @pytest.mark.parametrize("interface", ml_interfaces)
    @pytest.mark.parametrize("x", [0.732, 0.488])
    def test_single_expval(self, x, interface, seed):
        """Test a simple circuit with a single expval measurement"""
        qs = qml.tape.QuantumScript(
            [qml.RY(x, wires=0)],
            [qml.expval(qml.PauliZ(0))],
            shots=10000,
        )
        result = simulate(qs, rng=seed, interface=interface)
        if not interface == "jax":
            assert isinstance(result, np.float64)
        else:
            assert result.dtype == np.float64
        assert result.shape == ()

    @pytest.mark.parametrize("x", [0.732, 0.488])
    def test_single_sample(self, x, seed):
        """Test a simple circuit with a single sample measurement"""
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=10000)
        result = simulate(qs, rng=seed)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10000, 2)

    @pytest.mark.parametrize("x", [0.732, 0.488])
    @pytest.mark.parametrize("y", [0.732, 0.488])
    def test_multi_measurements(self, x, y, seed):
        """Test a simple circuit containing multiple measurements"""
        num_shots = 10000
        qs = qml.tape.QuantumScript(
            [
                qml.RX(x, wires=0),
                qml.RY(y, wires=1),
                qml.CNOT(wires=[0, 1]),
            ],
            [
                qml.expval(qml.PauliZ(0)),
                qml.counts(wires=range(2)),
                qml.sample(wires=range(2)),
            ],
            shots=num_shots,
        )
        result = simulate(qs, rng=seed)

        assert isinstance(result, tuple)
        assert len(result) == 3

        expected_keys, _ = self.probs_of_2_qubit_circ(x, y)
        assert list(result[1].keys()) == expected_keys

        assert result[2].shape == (10000, 2)

    shots_data = [
        [10000, 10000],
        [(10000, 2)],
        [10000, 20000],
        [(10000, 2), 20000],
        [(10000, 3), 20000, (30000, 2)],
    ]

    @pytest.mark.parametrize("x", [0.732, 0.488])
    @pytest.mark.parametrize("shots", shots_data)
    def test_expval_shot_vector(self, shots, x, seed):
        """Test a simple circuit with a single expval measurement for shot vectors"""
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.expval(qml.PauliZ(0))], shots=shots)
        result = simulate(qs, rng=seed)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, np.float64) for res in result)
        assert all(res.shape == () for res in result)

    @pytest.mark.parametrize("x", [0.732, 0.488])
    @pytest.mark.parametrize("shots", shots_data)
    def test_sample_shot_vector(self, shots, x, seed):
        """Test a simple circuit with a single sample measurement for shot vectors"""
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=shots)
        result = simulate(qs, rng=seed)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, np.ndarray) for res in result)
        assert all(res.shape == (s, 2) for res, s in zip(result, shots))

    @pytest.mark.parametrize("x", [0.732, 0.488])
    @pytest.mark.parametrize("y", [0.732, 0.488])
    @pytest.mark.parametrize("shots", shots_data)
    def test_multi_measurement_shot_vector(self, shots, x, y, seed):
        """Test a simple circuit containing multiple measurements for shot vectors"""
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript(
            [
                qml.RX(x, wires=0),
                qml.RY(y, wires=1),
                qml.CNOT(wires=[0, 1]),
            ],
            [
                qml.expval(qml.PauliZ(0)),
                qml.counts(wires=range(2)),
                qml.sample(wires=range(2)),
            ],
            shots=shots,
        )
        result = simulate(qs, seed)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        for shot_res, s in zip(result, shots):
            assert isinstance(shot_res, tuple)
            assert len(shot_res) == 3

            assert isinstance(shot_res[0], np.float64)
            assert isinstance(shot_res[1], dict)
            assert isinstance(shot_res[2], np.ndarray)

            expected_keys, _ = self.probs_of_2_qubit_circ(x, y)
            assert list(shot_res[1].keys()) == expected_keys

            assert shot_res[2].shape == (s, 2)

    @pytest.mark.parametrize("x", [0.732, 0.488])
    @pytest.mark.parametrize("y", [0.732, 0.488])
    @pytest.mark.parametrize("shots", shots_data)
    def test_custom_wire_labels(self, shots, x, y, seed):
        """Test that custom wire labels works as expected"""
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript(
            [
                qml.RX(x, wires="b"),
                qml.RY(y, wires="a"),
                qml.CNOT(wires=["b", "a"]),
            ],
            [
                qml.expval(qml.PauliZ("b")),
                qml.counts(wires=["a", "b"]),
                qml.sample(wires=["b", "a"]),
            ],
            shots=shots,
        )
        result = simulate(qs, rng=seed)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        for shot_res, s in zip(result, shots):
            assert isinstance(shot_res, tuple)
            assert len(shot_res) == 3

            assert isinstance(shot_res[0], np.float64)
            assert isinstance(shot_res[1], dict)
            assert isinstance(shot_res[2], np.ndarray)

            expected_keys, _ = self.probs_of_2_qubit_circ(x, y)
            assert list(shot_res[1].keys()) == expected_keys

            assert shot_res[2].shape == (s, 2)
