import pytest
import random
import numpy as np
import pennylane as qml
from pennylane.templates.tensornetworks.mps import *

"""
Unit tests for the MPS template.
"""

class TestIndicesMPS: 
    """Test function that computes MPS indices"""
    
    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the number of wires is incorrect."""
        
        n_wires = random.randrange(1,20,2)
        loc = 2
        with pytest.raises(AssertionError, match="number of wires must be an even integer; got {}".format(n_wires)):
            compute_indices_MPS(range(n_wires), loc)

        n_wires = random.randrange(2,20,2)
        loc = random.randrange(1,n_wires,2)
        with pytest.raises(AssertionError, match="loc must be an even integer; got {}".format(loc)):
            compute_indices_MPS(range(n_wires), loc)


    @pytest.mark.parametrize(
        ("wires", "loc", "expected_indices"),
        [
            ([1,2,3,4], 2, [[1,2],[2,3],[3,4]]),
            (['a','b','c','d'], 2, [['a','b'],['b','c'],['c','d']]),
        ],
    )
    def test_indices_output(self, wires, loc, expected_indices):

        indices = compute_indices_MPS(wires, loc)
        for i in range(len(expected_indices)):
            assert all(indices[i] == expected_indices[i])

class TestInputs:
    """Test inputs and pre-processing (ensure the correct exceptions are thrown for the inputs)"""

    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "loc", "msg_match"),
        [
            (None, None, [1,2,3,4], 5, "loc must be smaller than or equal to the number of wires; got loc = 5 and number of wires = 4"),
            (None, None, [1,2], 2, "number of wires must be greater than or equal to 3; got 2"),
            (None, None, [1,2,3,4], 1, "number of wires in each block must be larger than or equal to 2; got loc=1"),
        ],
    )
    def test_exception_wrong_dim(self, block, n_params_block, wires, loc, msg_match):
        """Verifies that an exception is raised if the number of wires or loc is incorrect."""
        with pytest.raises(ValueError, match=msg_match):
            MPS(wires, loc, block, n_params_block)

    def circuit1(weights, wires):
            qml.RZ(weights[0], wires = wires[0])
            qml.RZ(weights[1], wires = wires[1])
    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "loc", "weights", "msg_match"),
        [
            (circuit1, 2, [1,2,3,4], 2,  np.random.rand(3,3), "Weights tensor must have first dimension of length 2; got 3"),
            (circuit1, 2, [1,2,3,4], 2,  np.random.rand(2,4), "Weights tensor must have last dimension of length 3; got 4"),
        ],
    )
    def test_exception_wrong_weight_shape(self, block, n_params_block, wires, loc, weights, msg_match):
        """Verifies that an exception is raised if the weights shape is incorrect."""
        with pytest.raises(ValueError, match=msg_match):
            MPS(wires, loc, block, n_params_block, weights)

class TestAttributes:
    """Tests additional methods and attributes"""
    
    @pytest.mark.parametrize(
        ("n_wires", "loc", "n_params_block", "expected_shape"),
        [
            (4, 2, 1, (3, 1)),
            (16, 4, 1,(7, 1)),
        ],
    )
    def test_shape(self, n_wires, loc, n_params_block, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.MPS.shape(n_wires, loc, n_params_block)
        assert shape == expected_shape

    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "loc", "expected_nblock"),
        [
            (None, 1, [1,2,3,4], 2, 3),
        ],
    )
    def test_nblocks(self, wires, loc, block, n_params_block, expected_nblock):
        """Test that the number of blocks attribute returns the correct number"""

        mps = MPS(wires, loc, block, n_params_block)
        assert mps.n_blocks == expected_nblock

