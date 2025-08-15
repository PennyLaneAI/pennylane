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
"""Unit tests for initialize_state in devices/qubit_mixed/initialize_state."""

import pytest

import pennylane as qml
from pennylane import BasisState, StatePrep, math
from pennylane import numpy as np
from pennylane.devices.qubit_mixed import create_initial_state
from pennylane.operation import StatePrepBase

ml_interfaces = ["numpy", "autograd", "jax", "torch"]


def allzero_vec(num_wires, interface="numpy"):
    """Returns the state vector of the all-zero state."""
    state = np.zeros(2**num_wires, dtype=complex)
    state[0] = 1
    state = math.asarray(state, like=interface)
    return state


def allzero_dm(num_wires, interface="numpy"):
    """Returns the density matrix of the all-zero state."""
    num_axes = 2 * num_wires
    dm = np.zeros((2,) * num_axes, dtype=complex)
    dm[(0,) * num_axes] = 1
    dm = math.asarray(dm, like=interface)
    return dm


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ml_interfaces)
class TestInitializeState:
    """Test the functions in initialize_state.py"""

    # pylint:disable=unused-argument,too-few-public-methods
    class DefaultPrep(StatePrepBase):
        """A dummy class that assumes it was given a state vector."""

        def __init__(self, *args, **kwargs):
            self.dtype = kwargs.pop("dtype", None)
            super().__init__(*args, **kwargs)

        def state_vector(self, wire_order=None):
            sv = self.parameters[0]
            if self.dtype is not None:
                sv = qml.math.cast(sv, self.dtype)
            return sv

    def test_create_initial_state_no_state_prep(self, interface):
        """Tests that create_initial_state works without a state-prep operation."""
        wires = [0, 1]
        num_wires = len(wires)
        state = create_initial_state(wires, like=interface)

        state_correct = allzero_dm(num_wires, interface)
        assert math.allequal(state, state_correct)
        assert math.get_interface(state) == interface
        assert "complex" in str(state.dtype)

    def test_create_initial_state_with_dummy_state_prep(self, interface):
        """Tests that create_initial_state works with a state-prep operation."""
        wires = [0, 1]
        num_wires = len(wires)

        vec_correct = allzero_vec(num_wires, interface)
        state_correct = allzero_dm(num_wires, interface)
        prep_op = self.DefaultPrep(qml.math.array(vec_correct, like=interface), wires=wires)
        state = create_initial_state(wires, prep_operation=prep_op, like=interface)
        assert math.allequal(state, state_correct)
        assert math.get_interface(state) == interface

    def test_create_initial_state_with_StatePrep(self, interface):
        """Tests that create_initial_state works with a state-prep operation."""
        wires = [0, 1]
        num_wires = len(wires)
        # The following 2 lines are for reusing the statevec code on the density matrices
        vec_correct = allzero_vec(num_wires, interface)
        state_correct = allzero_dm(num_wires, interface)
        state_correct_flatten = math.reshape(vec_correct, [-1])
        prep_op = StatePrep(qml.math.array(state_correct_flatten, like=interface), wires=wires)
        state = create_initial_state(wires, prep_operation=prep_op, like=interface)
        assert math.allequal(state, state_correct)
        assert math.get_interface(state) == interface

    def test_create_initial_state_with_StatePrep_subset(self, interface):
        """Tests that create_initial_state works with a subset state-prep operation."""
        wires = [0, 1]
        prep_op = BasisState([0, 1], wires=wires)
        state = create_initial_state(wires, prep_operation=prep_op, like=interface)

        prep_op_subset = BasisState([1], wires=[1])
        state_subset = create_initial_state(wires, prep_operation=prep_op_subset, like=interface)
        assert math.allequal(state, state_subset)
        assert math.get_interface(state) == interface
        assert math.get_interface(state_subset) == interface

    def test_create_initial_state_with_QubitDensityMatrix(self, interface):
        """Tests that create_initial_state works with a state-prep operation."""
        wires = [0, 1]
        num_wires = len(wires)
        # The following 2 lines are for reusing the statevec code on the density matrices
        state_correct = allzero_dm(num_wires, interface)
        prep_op = qml.QubitDensityMatrix(qml.math.array(state_correct, like=interface), wires=wires)
        state = create_initial_state(wires, prep_operation=prep_op, like=interface)
        assert math.allequal(state, state_correct)
        assert math.get_interface(state) == interface
