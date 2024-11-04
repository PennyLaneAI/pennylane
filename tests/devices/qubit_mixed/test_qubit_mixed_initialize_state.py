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
from pennylane import math
from pennylane import numpy as np
from pennylane.devices.qubit_mixed import create_initial_state
from pennylane.devices.qubit_mixed.initialize_state import _apply_state_vector, _create_basis_state
from pennylane.operation import StatePrepBase

ml_interfaces = ["numpy", "autograd", "jax", "torch", "tensorflow"]


def allzero_dm(num_wires, interface="numpy"):
    """Returns the density matrix of the all-zero state."""
    num_axes = 2 * num_wires
    dm = np.zeros((2,) * num_axes, dtype=complex)
    dm[(0,) * num_axes] = 1
    dm = math.asarray(dm, like=interface)
    return dm


class TestInitializeState:
    """Test the functions in initialize_state.py"""

    # pylint:disable=unused-argument,too-few-public-methods
    class DefaultPrep(StatePrepBase):
        """A dummy class that assumes it was given a state vector."""

        num_wires = qml.operation.AllWires

        def __init__(self, *args, **kwargs):
            self.dtype = kwargs.pop("dtype", None)
            super().__init__(*args, **kwargs)

        def state_vector(self, wire_order=None):
            sv = self.parameters[0]
            if self.dtype is not None:
                sv = qml.math.cast(sv, self.dtype)
            return sv

    # @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ml_interfaces)
    def test_create_initial_state_no_state_prep(self, interface):
        """Tests that create_initial_state works without a state-prep operation."""
        wires = [0, 1]
        num_wires = len(wires)
        state = create_initial_state(wires, like=interface)

        state_correct = allzero_dm(num_wires, interface)
        assert math.allequal(state, state_correct)
        assert math.get_interface(state) == interface
        assert "complex" in str(state.dtype)

    # @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ml_interfaces)
    def test_create_initial_state_with_state_prep(self, interface):
        """Tests that create_initial_state works with a state-prep operation."""
        wires = [0, 1]
        num_wires = len(wires)
        state_correct = allzero_dm(num_wires, interface)
        prep_op = self.DefaultPrep(qml.math.array(state_correct, like=interface), wires=wires)
        state = create_initial_state(wires, prep_operation=prep_op, like=interface)
        assert math.allequal(state, state_correct)
        assert math.get_interface(state) == interface


class TestHelperFuncs:
    """Test the helper functions in initialize_state.py"""

    @pytest.mark.parametrize("num_wires", [1, 2, 3])
    def test_create_basis_state(self, num_wires):
        """Test that the function _create_basis_state returns the correct basis state."""
        wires = list(range(num_wires))
        basis_state = _create_basis_state(wires, num_wires, 0)
        basis_state_correct = np.zeros([2] * (2 * num_wires), dtype=complex)
        basis_state_correct[(0,) * (2 * num_wires)] = 1
        basis_state_correct = math.asarray(basis_state_correct, like="numpy")
        assert math.allequal(basis_state, basis_state_correct)

    @pytest.mark.parametrize("num_wires", [1, 2, 3])
    def test_apply_state_vector(self, num_wires):
        """Test that the function _apply_state_vector correctly applies a state vector."""
        wires = list(range(num_wires))
        state = np.array([1, 0, 0, 0], dtype=complex)
        state = math.asarray(state, like="numpy")
        basis_state = _create_basis_state(wires, num_wires, 0)
        state_correct = np.zeros([2] * (2 * num_wires), dtype=complex)
        state_correct[(0,) * num_wires] = 1
        state_correct = math.asarray(state_correct, like="numpy")
        state_applied = _apply_state_vector(basis_state, state)
        assert math.allequal(state_applied, state_correct)

    @pytest.mark.parametrize("num_wires", [1, 2, 3])
    def test_apply_state_vector_raises(self, num_wires):
        """Test that the function _apply_state_vector raises an error if the state vector is the wrong size."""
        wires = list(range(num_wires))
        state = np.array([1, 0, 0], dtype=complex)
        state = math.asarray(state, like="numpy")
        basis_state = _create_basis_state(wires, num_wires, 0)
        with pytest.raises(ValueError, match="State vector must be of length 2**num_wires"):
            _apply_state_vector(basis_state, state)
