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


# pylint: disable=too-few-public-methods
class TestResultInterface:
    """Test that the result interface is correct."""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "op", [qml.RX(np.pi, [0]), qml.BasisState(np.array([1, 1]), wires=range(2))]
    )
    @pytest.mark.parametrize("interface", ("jax", "tensorflow", "torch", "autograd", "numpy"))
    def test_result_has_correct_interface(self, op, interface):
        """Test that even if no interface parameters are given, result is correct."""
        qs = qml.tape.QuantumScript([op], [qml.expval(qml.Z(0))])
        res = simulate(qs, interface=interface)

        assert qml.math.get_interface(res) == interface


# pylint: disable=too-few-public-methods
class TestStatePrepBase:
    """Tests integration with various state prep methods."""


@pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
class TestBasicCircuit:
    """Tests a basic circuit with one RX gate and a few simple expectation values."""


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
