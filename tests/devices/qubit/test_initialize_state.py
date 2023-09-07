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
"""Unit tests for create_initial_state in devices/qubit."""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices.qubit import create_initial_state
from pennylane.operation import StatePrepBase


class TestInitializeState:
    """Test the functions in initialize_state.py"""

    # pylint:disable=unused-argument,too-few-public-methods
    class DefaultPrep(StatePrepBase):
        """A dummy class that assumes it was given a state vector."""

        num_wires = qml.operation.AllWires

        def state_vector(self, wire_order=None):
            return self.parameters[0]

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    def test_create_initial_state_no_state_prep(self, interface):
        """Tests that create_initial_state works without a state-prep operation."""
        state = create_initial_state([0, 1], like=interface)
        assert qml.math.allequal(state, [[1, 0], [0, 0]])
        assert qml.math.get_interface(state) == interface

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    def test_create_initial_state_with_state_prep(self, interface):
        """Tests that create_initial_state works with a state-prep operation."""
        prep_op = self.DefaultPrep(qml.math.array([1 / 2] * 4, like=interface), wires=[0, 1])
        state = create_initial_state([0, 1], prep_operation=prep_op)
        assert qml.math.allequal(state, [1 / 2] * 4)
        assert qml.math.get_interface(state) == interface

    def test_create_initial_state_with_BasisState(self):
        """Tests that create_initial_state works with a real state-prep operator."""
        prep_op = qml.BasisState([0, 1, 0], wires=[0, 1, 2])
        state = create_initial_state([0, 1, 2], prep_operation=prep_op)
        assert state[0, 1, 0] == 1
        state[0, 1, 0] = 0  # set to zero to make test below simple
        assert qml.math.allequal(state, np.zeros((2, 2, 2)))

    def test_create_initial_state_with_StatePrep(self):
        """Tests that create_initial_state works with the StatePrep operator."""
        prep_op = qml.StatePrep(np.array([0, 1, 0, 0, 0, 0, 0, 1]) / np.sqrt(2), wires=[0, 1, 2])
        state = create_initial_state([0, 1, 2], prep_operation=prep_op)
        assert state[0, 0, 1] == 1 / np.sqrt(2)
        assert state[1, 1, 1] == 1 / np.sqrt(2)
        state[0, 0, 1] = 0
        state[1, 1, 1] = 0  # set to zero to make test below simple
        assert qml.math.allequal(state, np.zeros((2, 2, 2)))

    def test_create_initial_state_with_StatePrep_broadcasted(self):
        """Tests that create_initial_state works with a broadcasted StatePrep
        operator."""
        prep_op = qml.StatePrep(np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), wires=[0, 1])
        state = create_initial_state([0, 1], prep_operation=prep_op)
        assert state[0, 0, 1] == 1
        assert state[1, 1, 1] == 1
        assert state[2, 1, 0] == 1
        state[0, 0, 1] = 0
        state[1, 1, 1] = 0
        state[2, 1, 0] = 0  # set to zero to make test below simple
        assert qml.math.allequal(state, np.zeros((3, 2, 2)))

    def test_create_initial_state_casts_to_like_with_prep_op(self):
        """Tests that the like argument is ignored when a prep-op is provided."""
        prep_op = self.DefaultPrep([0, 0, 0, 1], wires=[0, 1])
        state = create_initial_state([0, 1], prep_operation=prep_op, like="torch")
        assert qml.math.get_interface(state) == "torch"

    def test_create_initial_state_defaults_to_numpy(self):
        """Tests that the default interface is vanilla numpy."""
        state = qml.devices.qubit.create_initial_state((0, 1))
        assert qml.math.get_interface(state) == "numpy"
