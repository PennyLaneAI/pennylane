import pytest
import numpy as np
import pennylane as qml
from pennylane.templates.tensornetworks.mps import *

"""
Unit tests for the MPS template.
"""

class TestInputs:
    """Test inputs and pre-processing (ensure the correct exceptions are thrown for the inputs)"""
    
    def my_minimal_quantum_function(weights, wires):
        """The simplest function"""
        qml.RZ(weights[0], wires=wires[0])
        qml.CNOT(wires=wires[:])

    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "loc", "msg_match"),
        [
            (None, None, [1,2,3], 4, "loc must be smaller than or equal to the number of wires; got loc = 4 and number of wires = 3"),
            (None, None, [1,2], 2, "number of wires must be greater than or equal to 3; got 2"),
            (None, None, [1,2], 1, "loc must be larger than or equal to 2; got 1"),
        ],
    )
    def test_exception_wrong_dim(self, block, n_params_block, wires, loc, msg_match):
        """Verifies that an exception is raised if the number of wires or loc is incorrect."""
        with pytest.raises(AssertionError, match=msg_match):
            MPS(wires, loc, block, n_params_block, wires)

    @pytest.mark.parametrize(
        ("wires", "loc", "expected_indices"),
        [
            ([1,2,3], 2, [[1,2],[2,3]]),
            (['a','b','c'], 2, [['a','b'],['b','c']]),
            ([1,3,2,4], 2, [[1,3],[3,2],[2,4]]),
        ],
    )
    def test_indices_output(self, wires, loc, expected_indices):

        indices = compute_indices_MPS(wires, loc)
        for i in range(len(expected_indices)):
            assert all(indices[i] == expected_indices[i])

            

class TestAttributes:
    """Tests additional methods and attributes"""
    
    @pytest.mark.parametrize(
        "n_wires, loc, n_params_block, expected_shape",
        [
            (3, 2, 1, (2, 10)),
            (4, 2, 1, (3, 1)),
            (16, 4, 1,(7, 1)),
        ],
    )
    def test_shape(self, n_layers, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.MPS.shape(n_wires, loc, n_params_block)
        assert shape == expected_shape

            
            
class TestDifferentiability:
    """Test that the template is differentiable"""
    #TODO
