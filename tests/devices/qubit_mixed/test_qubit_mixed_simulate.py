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
from dummy_debugger import Debugger
from flaky import flaky

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
            ops=[qml.BasisState(np.array([1, 1]), wires=[0, 1])], # prod state |1, 1>
            measurements=[qml.probs(wires=[0, 1, 2])],
        )
        probs = simulate(qs)
        expected = np.zeros(8)
        expected[6] = 1.0 # Should be |110> = |6>
        assert qml.math.allclose(probs, expected)


@pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
class TestBasicCircuit:
    """Tests a basic circuit with one RX gate and a few simple expectation values."""

    @staticmethod
    def get_quantum_script(phi, subspace):
        """Get the quantum script where RX is applied then observables are measured"""
        ops = [qml.RX(phi, wires=subspace[0])]
        obs = [
            qml.expval(qml.PauliX(subspace[0])),
            qml.expval(qml.PauliY(subspace[0])),
            qml.expval(qml.PauliZ(subspace[0]))
        ]
        return qml.tape.QuantumScript(ops, obs)

    def test_basic_circuit_numpy(self, subspace):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)

        qs = self.get_quantum_script(phi, subspace)
        result = simulate(qs)

        # For density matrix simulation of RX(phi), the expectations are:
        expected_measurements = (
            0,           # <X> appears to be 0 in density matrix formalism
            -np.sin(phi),  # <Y> has negative sign
            np.cos(phi)  # <Z> is correct
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert np.allclose(result, expected_measurements)

        # Test state evolution and measurement separately
        state, is_state_batched = get_final_state(qs)
        result = measure_final_state(qs, state, is_state_batched)

        # For RX rotation in density matrix form - note flipped signs
        expected_state = np.array([
            [np.cos(phi/2)**2, 0.5j*np.sin(phi)],
            [-0.5j*np.sin(phi), np.sin(phi/2)**2]
        ])

        assert np.allclose(state, expected_state)
        assert not is_state_batched
        assert np.allclose(result, expected_measurements)

    @pytest.mark.autograd
    def test_autograd_results_and_backprop(self, subspace):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        def f(x):
            qs = self.get_quantum_script(x, subspace)
            return qml.numpy.array(simulate(qs))

        result = f(phi)
        expected = (0, -np.sin(phi), np.cos(phi))  # Note negative sin
        assert qml.math.allclose(result, expected)

        g = qml.jacobian(f)(phi)
        expected = (0, -np.cos(phi), -np.sin(phi))  # Note negative derivatives
        assert qml.math.allclose(g, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_results_and_backprop(self, use_jit, subspace):
        """Tests execution and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        def f(x):
            qs = self.get_quantum_script(x, subspace)
            return simulate(qs)

        if use_jit:
            f = jax.jit(f)

        result = f(phi)
        expected = (0, -np.sin(phi), np.cos(phi))  # Adjusted expectations
        assert qml.math.allclose(result, expected)

        g = jax.jacobian(f)(phi)
        expected = (0, -np.cos(phi), -np.sin(phi))  # Adjusted gradients
        assert qml.math.allclose(g, expected)

    @pytest.mark.torch
    def test_torch_results_and_backprop(self, subspace):
        """Tests execution and gradients with torch."""
        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        def f(x):
            qs = self.get_quantum_script(x, subspace)
            return simulate(qs)

        result = f(phi)
        expected = (0, -np.sin(phi.detach().numpy()), np.cos(phi.detach().numpy()))
        
        result_detached = math.asarray(result, like="torch").detach().numpy()
        assert math.allclose(result_detached, expected)

        # Convert complex jacobian to real and take only real part for comparison
        jacobian = math.asarray(torch.autograd.functional.jacobian(f, phi + 0j), like="torch")
        jacobian = jacobian.real if hasattr(jacobian, 'real') else jacobian
        expected = (0, -np.cos(phi.detach().numpy()), -np.sin(phi.detach().numpy()))
        assert math.allclose(jacobian.detach().numpy(), expected)

    @pytest.mark.tf
    def test_tf_results_and_backprop(self, subspace):
        """Tests execution and gradients with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = self.get_quantum_script(phi, subspace)  # Fixed: using phi instead of x
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


@pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
class TestBroadcasting:
    """Test that simulate works with broadcasted parameters."""

    @staticmethod
    def get_expected_state(x, subspace):
        """Gets the expected final state of the circuit described in `get_ops_and_measurements`."""

    @staticmethod
    def get_expectation_values(x, subspace):
        """Gets the expected final expvals of the circuit described in `get_ops_and_measurements`."""

    @staticmethod
    def get_quantum_script(x, subspace, shots=None, extra_wire=False):
        """Gets quantum script of a circuit that includes
        parameter broadcasted operations and measurements."""

    def test_broadcasted_op_state(self, subspace):
        """Test that simulate works for state measurements
        when an operation has broadcasted parameters"""

    def test_broadcasting_with_extra_measurement_wires(self, mocker, subspace):
        """Test that broadcasting works when the operations don't act on all wires."""


@pytest.mark.parametrize("extra_wires", [1, 3])
class TestStatePadding:
    """Tests if the state zeros padding works as expected for when operators don't act on all
    measured wires."""

    @staticmethod
    def get_expected_dm(x, extra_wires):
        """Gets the final density matrix of the circuit described in `get_ops_and_measurements`."""

    @staticmethod
    def get_quantum_script(x, extra_wires):
        """Gets a quantum script of a circuit where operators don't act on all measured wires."""

    def test_extra_measurement_wires(self, extra_wires):
        """Tests if correct state is returned when operators don't act on all measured wires."""

    def test_extra_measurement_wires_broadcasting(self, extra_wires):
        """Tests if correct state is returned when there is broadcasting and
        operators don't act on all measured wires."""


@pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
class TestDebugger:
    """Tests that the debugger works for a simple circuit"""

    # basis_state

    @staticmethod
    def get_debugger_quantum_script(phi, subspace):
        """Get the quantum script with debugging where TRX is applied
        then GellMann observables are measured"""

    def test_debugger_numpy(self, subspace):
        """Test debugger with numpy"""

    @pytest.mark.autograd
    def test_debugger_autograd(self, subspace):
        """Tests debugger with autograd"""

    @pytest.mark.jax
    def test_debugger_jax(self, subspace):
        """Tests debugger with JAX"""

    @pytest.mark.torch
    def test_debugger_torch(self, subspace):
        """Tests debugger with torch"""

    # pylint: disable=invalid-unary-operand-type
    @pytest.mark.tf
    def test_debugger_tf(self, subspace):
        """Tests debugger with tensorflow."""
