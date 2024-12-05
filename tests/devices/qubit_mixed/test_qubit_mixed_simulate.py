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

ml_interfaces = ["numpy", "autograd", "jax", "torch", "tensorflow"]


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
        """Test execution with a basic circuit."""
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

        # Test state evolution and measurement separately
        state, is_state_batched = get_final_state(qs)
        result = measure_final_state(qs, state, is_state_batched)

        expected_state = np.array(
            [
                [np.cos(phi / 2) ** 2, 0.5j * np.sin(phi)],
                [-0.5j * np.sin(phi), np.sin(phi / 2) ** 2],
            ]
        )

        assert np.allclose(state, expected_state)
        assert not is_state_batched
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


def test_finite_shot_not_implemented():
    op = qml.RX(np.pi, [0])
    qs = qml.tape.QuantumScript([op], [qml.expval(qml.Z(0))], shots=1000)
    with pytest.raises(NotImplementedError):
        simulate(qs)
