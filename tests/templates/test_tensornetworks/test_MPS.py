
import random
import pytest
import sys
sys.path.append('../src/templates')
from MPS import *

"""
Unit tests for the MPS template.
"""

class TestIndicesMPS: 
    """Test function that computes MPS indices"""
    
    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the number of wires is incorrect."""

    
        with pytest.raises(AssertionError, match="n_wires should be an even integer"):
            n_wires = random.randrange(1,20,2)
            loc = 2
            compute_indices_MPS(n_wires, loc)

        with pytest.raises(AssertionError, match="loc should be an even integer"):
            n_wires = random.randrange(2,20,2)
            loc = random.randrange(1,n_wires,2)
            compute_indices_MPS(n_wires, loc)

        with pytest.raises(AssertionError, match="loc should be smaller or equal than num_wires"):
            n_wires = random.randrange(2,20,2)
            loc = random.randrange(n_wires+2,2*n_wires+2,2)
            compute_indices_MPS(n_wires, loc)


    # TO DO: FINISH THIS TEST 
    QUEUES = [
    (4,2, np.array([[0,1], [1,2], [2,3]]))
    ]

    @pytest.mark.parametrize("n_wires, loc, expected_indices", QUEUES)
    def test_indices_output(self, n_wires, loc, expected_indices):

        indices = compute_indices_MPS(n_wires, loc)
        for i in len(indices):
            assert indices[i] == expected_indices[i]