"""Unit tests for create_initial_state in devices/qubit."""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices.qutrit_mixed import create_initial_state
from pennylane import (
    QutritBasisState,
)
from pennylane.operation import StatePrepBase

class TestInitializeState:
    """Test the functions in qutrit_mixed/initialize_state.py"""

    # TODO add pylint disable
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
        expected = np.zeros((3, 3, 3, 3))
        expected[0, 0, 0, 0] = 1
        assert qml.math.allequal(state, expected)
        assert qml.math.get_interface(state) == interface

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    def test_create_initial_state_with_state_prep(self, interface):
        """Tests that create_initial_state works with a state-prep operation."""
        prep_op = self.DefaultPrep(qml.math.array([1 / np.sqrt(9)] * 9, like=interface), wires=[0, 1])
        state = create_initial_state([0, 1], prep_operation=prep_op)
        #assert qml.math.allequal(state, [1 / 2] * 4) TODO fix
        assert qml.math.get_interface(state) == interface

    def test_create_initial_state_with_BasisState(self):
        """Tests that create_initial_state works with a real state-prep operator."""
        prep_op = QutritBasisState([1, 2, 0], wires=[0, 1, 2])
        state = create_initial_state([0, 1, 2], prep_operation=prep_op)
        assert state[1, 2, 0, 1, 2, 0] == 1
        state[1, 2, 0, 1, 2, 0] = 0  # set to zero to make test below simple
        assert qml.math.allequal(state, np.zeros(([3] * 6)))

    # def test_create_initial_state_with_StatePrep(self, prep_op_cls):
    #     """Tests that create_initial_state works with the StatePrep operator."""
    #     prep_op = prep_op_cls(np.array([0, 1, 0, 0, 0, 0, 0, 1]) / np.sqrt(2), wires=[0, 1, 2])
    #     state = create_initial_state([0, 1, 2], prep_operation=prep_op)
    #     expected = np.zeros((2, 2, 2))
    #     expected[0, 0, 1] = expected[1, 1, 1] = 1 / np.sqrt(2)
    #     assert np.array_equal(state, expected)

    # def test_create_initial_state_with_StatePrep_broadcasted(self):
    #     """Tests that create_initial_state works with a broadcasted StatePrep
    #     operator."""
    #     prep_op = QutritBasisState(np.array([[1, 0], [2, 1]]), wires=[0, 1])
    #     state = create_initial_state([0, 1], prep_operation=prep_op)
    #     expected = np.zeros((3, 3, 3))
    #     expected[0, 1, 0, 1, 0] = expected[1, 2, 1, 2, 1] = 1
    #     assert np.array_equal(state, expected)

    def test_create_initial_state_defaults_to_numpy(self):
        """Tests that the default interface is vanilla numpy."""
        state = qml.devices.qubit.create_initial_state((0, 1))
        assert qml.math.get_interface(state) == "numpy"


    @pytest.mark.torch
    def test_create_initial_state_casts_to_like_with_prep_op(self):
        """Tests that the like argument is not ignored when a prep-op is provided."""
        prep_op = self.DefaultPrep([1 / np.sqrt(9)] * 9, wires=[0, 1])
        state = create_initial_state([0, 1], prep_operation=prep_op, like="torch")
        assert qml.math.get_interface(state) == "torch"