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
"""Unit tests for create_initial_state in devices/qutrit_mixed."""

import pytest

import pennylane as qp
from pennylane import QutritBasisState, QutritDensityMatrix
from pennylane import numpy as np
from pennylane.devices.qutrit_mixed import create_initial_state
from pennylane.operation import StatePrepBase

ml_interfaces = ["numpy", "autograd", "jax", "torch", "tensorflow"]


class TestInitializeState:
    """Test the functions in qutrit_mixed/initialize_state.py"""

    # pylint:disable=unused-argument,too-few-public-methods
    class DefaultPrep(StatePrepBase):
        """A dummy class that assumes it was given a state vector."""

        def state_vector(self, wire_order=None):
            return self.parameters[0]

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ml_interfaces)
    def test_create_initial_state_no_state_prep(self, interface):
        """Tests that create_initial_state works without a state-prep operation."""
        state = create_initial_state([0, 1], like=interface)
        expected = np.zeros((3, 3, 3, 3))
        expected[0, 0, 0, 0] = 1
        assert qp.math.allequal(state, expected)
        assert qp.math.get_interface(state) == interface

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ml_interfaces)
    def test_create_initial_state_with_state_prep(self, interface):
        """Tests that create_initial_state works with a state-prep operation."""
        prep_op = self.DefaultPrep(
            qp.math.array([1 / np.sqrt(9)] * 9, like=interface), wires=[0, 1]
        )
        state = create_initial_state([0, 1], prep_operation=prep_op)
        expected = np.reshape([1 / 9] * 81, [3, 3, 3, 3])

        assert qp.math.allequal(state, expected)
        if interface == "autograd":
            assert qp.math.get_interface(state) == "numpy"
        else:
            assert qp.math.get_interface(state) == interface

    def test_create_initial_state_with_BasisState(self):
        """Tests that create_initial_state works with a real state-prep operator."""
        prep_op = QutritBasisState([1, 2, 0], wires=[0, 1, 2])
        state = create_initial_state([0, 1, 2], prep_operation=prep_op)
        assert state[1, 2, 0, 1, 2, 0] == 1
        state[1, 2, 0, 1, 2, 0] = 0  # set to zero to make test below simple
        assert qp.math.allequal(state, np.zeros([3] * 6))

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ml_interfaces)
    def test_create_initial_state_with_QutritDensityMatrix(self, interface):
        """Tests that create_initial_state works with a state-prep operation."""
        wires = [0, 1]
        num_wires = len(wires)
        state_correct = np.zeros((3, 3) * num_wires, dtype=complex)
        state_correct[(0, 0) * num_wires] = 1
        state_correct = qp.math.asarray(state_correct, like=interface)
        prep_op = QutritDensityMatrix(qp.math.array(state_correct, like=interface), wires=wires)
        state = create_initial_state(wires, prep_operation=prep_op, like=interface)
        assert qp.math.allequal(state, state_correct)
        assert qp.math.get_interface(state) == interface

    @pytest.mark.parametrize("wires", [(0, 1), qp.wires.Wires([0, 1])])
    def test_create_initial_state_wires(self, wires):
        """Tests that create_initial_state works with qp.Wires object and list."""
        state = create_initial_state(wires)
        expected = np.zeros((3, 3, 3, 3))
        expected[0, 0, 0, 0] = 1
        assert qp.math.allequal(state, expected)

    # TODO: Add tests for qutrit state prep

    def test_create_initial_state_defaults_to_numpy(self):
        """Tests that the default interface is vanilla numpy."""
        state = qp.devices.qubit.create_initial_state((0, 1))
        assert qp.math.get_interface(state) == "numpy"

    @pytest.mark.torch
    def test_create_initial_state_casts_to_like_with_prep_op(self):
        """Tests that the like argument is not ignored when a prep-op is provided."""
        prep_op = self.DefaultPrep([1 / np.sqrt(9)] * 9, wires=[0, 1])
        state = create_initial_state([0, 1], prep_operation=prep_op, like="torch")
        assert qp.math.get_interface(state) == "torch"

def test_qutrit_density_matrix_qnode_integration():
    """Integration test for QutritDensityMatrix on entire set of wires using QNode.
    """
    n = 2
    dev = qp.device("default.qutrit.mixed", wires=2 * n)

    @qp.qnode(dev)
    def test_circuit(rho):
        # Initialize all 2n qutrits of rho
        qp.QutritDensityMatrix(rho, wires=range(0, 2 * n))

        # Apply THadamard gate to second set
        for a in range(n, 2 * n):
            qp.THadamard(a)

        return qp.probs(wires=range(n))

    # Define the 2-qutrit density matrix for GHZ state: (|00> + |11> + |22>)/sqrt(3)
    ghz = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=complex) / np.sqrt(3)
    ghz_dm = np.outer(ghz, np.conj(ghz))  # shape (9, 9)

    # This should not raise ValueError
    result = test_circuit(np.kron(ghz_dm, ghz_dm))

    # Expected: probabilities for GHZ state are [1/3, 0, 0,0, 1/3,0,0,0,1/3]
    expected = np.array([1/3, 0, 0,0, 1/3,0,0,0,1/3])
    assert np.allclose(result, expected, atol=1e-8)