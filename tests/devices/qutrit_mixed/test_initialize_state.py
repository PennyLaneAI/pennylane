"""Unit tests for create_initial_state in devices/qubit."""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices.qutrit_mixed import create_initial_state
from pennylane import (
    QutritBasisState,
)

class TestInitializeState:
    """Test the functions in qutrit_mixed/initialize_state.py"""

    # TODO add pylint disable

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    def test_create_initial_state_no_state_prep(self, interface):
        """Tests that create_initial_state works without a state-prep operation."""
        state = create_initial_state([0, 1], like=interface)
        assert qml.math.allequal(state, None ) #TODO[[1, 0], [0, 0]])
        assert qml.math.get_interface(state) == interface

    def test_create_initial_state_with_BasisState(self):
        """Tests that create_initial_state works with a real state-prep operator."""
        prep_op = QutritBasisState([0, 1, 0], wires=[0, 1, 2])
        state = create_initial_state([0, 1, 2], prep_operation=prep_op)
        assert state[0, 1, 0] == 1
        state[0, 1, 0] = 0  # set to zero to make test below simple
        assert qml.math.allequal(state, np.zeros((2, 2, 2)))

    def test_create_initial_state_with_BasisState_broadcasted(self):
        """Tests that create_initial_state works with a broadcasted StatePrep
        operator."""
        prep_op = QutritBasisState(np.array([[0, 1], [1, 0], [2, 1]]), wires=[0, 1])
        state = create_initial_state([0, 1], prep_operation=prep_op)
        expected = None #TODO np.zeros((3, 2, 2))
        expected[0, 0, 1] = expected[1, 1, 1] = expected[2, 1, 0] = 1
        assert np.array_equal(state, expected)

    def test_create_initial_state_defaults_to_numpy(self):
        """Tests that the default interface is vanilla numpy."""
        state = qml.devices.qubit.create_initial_state((0, 1))
        assert qml.math.get_interface(state) == "numpy"

    def test_create_initial_state_with_New_State_prep(self):
        """"""
        pass

    @pytest.mark.parametrize("", ["", "", "", ""])
    def test_create_initial_state_with_Qubit_State_prep(self):
        """"""
        pass
