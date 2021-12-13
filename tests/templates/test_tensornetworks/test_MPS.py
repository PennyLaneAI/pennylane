# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Tests for the MPS template.
"""
import math
import pytest
import random
import numpy as np
import pennylane as qml
from pennylane.templates.tensornetworks.mps import *


class TestIndicesMPS: 
    """Test function that computes MPS indices"""

    # @pytest.mark.parametrize(
    #     ("n_wires", "loc", "msg_match"),
    #     [
    #         (2, 5, "loc must be an even integer; got 5"),
    #         (4, 6, "loc must be smaller than or equal to the number of wires; got loc = 6 and number of wires = 4"),
    #         (5, 0, "number of wires in each block must be larger than or equal to 2; got loc=0"),
    #     ],
    # )
    def test_exception_loc_uneven(self):
        """Verifies that an exception is raised if loc is not even."""

        n_wires = random.randrange(2,20)
        loc = random.randrange(1,n_wires,2)
        with pytest.raises(ValueError, match=f'loc must be an even integer; got {loc}'):
            compute_indices_MPS(range(n_wires), loc)

    def test_exception_loc_large(self):
        """Verifies that an exception is raised when loc is too large."""

        n_wires = random.randrange(2,20)
        loc = random.randrange(np.ceil(n_wires/2)*2,n_wires+20,2) #an even number between n_wires and n_wires + 20
        with pytest.raises(ValueError, match=f'loc must be smaller than or equal to the number of wires; got loc = {loc} and number of wires = {n_wires}'):
            compute_indices_MPS(range(n_wires), loc)

    def test_exception_loc_small(self):
        """Verifies that an exception is raised when loc is less than 2."""

        n_wires = 2
        loc = 0
        with pytest.raises(ValueError, match=f'number of wires in each block must be larger than or equal to 2; got loc = {loc}'):
            compute_indices_MPS(range(n_wires), loc)

    @pytest.mark.parametrize(
        ("n_wires", "loc"),
        [
            (5,4),
            (9,4),
            (7,6),
        ],
    )
    def test_warning_many_wires(self, n_wires, loc):
        """Verifies that a warning is raised if n_wires doesn't correspond to loc."""

        with pytest.warns(Warning, match=f"The number of wires should be a multiple of {int(loc/2)}; got {n_wires}"):
            compute_indices_MPS(range(n_wires), loc)

    @pytest.mark.parametrize(
        ("wires", "loc", "expected_indices"),
        [
            ([1,2,3,4], 2, [[1,2],[2,3],[3,4]]),
            (['a','b','c','d'], 2, [['a','b'],['b','c'],['c','d']]),
        ],
    )
    def test_indices_output(self, wires, loc, expected_indices):
        """Verifies the indices are correct for both integer and string wire labels."""
        indices = compute_indices_MPS(wires, loc)
        for i in range(len(expected_indices)):
            assert all(indices[i] == expected_indices[i])

class TestTemplateInputs:
    """Test template inputs and pre-processing (ensure the correct exceptions are thrown for the inputs)"""

    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "loc", "msg_match"),
        [
            (None, None, [1,2,3,4], 7, "loc must be an even integer; got 7"),
            (None, None, [1,2,3,4], 6, "loc must be smaller than or equal to the number of wires; got loc = 6 and number of wires = 4"),
            (None, None, [1,2,3,4], 0, "number of wires in each block must be larger than or equal to 2; got loc = 0"),
        ],
    )
    def test_exception_wrong_input(self, block, n_params_block, wires, loc, msg_match):
        """Verifies that an exception is raised if the number of wires or loc is incorrect."""
        with pytest.raises(ValueError, match=msg_match):
            MPS(wires, loc, block, n_params_block)

    def test_warning_many_wires(self):
        """Verifies that a warning is raised if n_wires doesn't correspond to loc."""

        loc = 4
        wires = [1,2,3,4,5]
        n_wires = len(wires)
        n_params_block = 1
        with pytest.warns(Warning, match=f"The number of wires should be a multiple of {int(loc/2)}; got {n_wires}"):
            MPS(wires, loc, block = None, n_params_block=n_params_block)

    def circuit1(weights, wires):
        qml.RZ(weights[0], wires = wires[0])
        qml.RZ(weights[1], wires = wires[1])
    def circuit2(block_weights,block_wires):
        qml.CNOT(wires=[block_wires[0],block_wires[1]])
        SELweights = np.array([[[block_weights[0],block_weights[1],block_weights[2]],
        [block_weights[3],block_weights[4],block_weights[5]]]])
        qml.StronglyEntanglingLayers(SELweights,wires=block_wires)

    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "loc", "block_weights", "msg_match"),
        [
            (circuit1, 2, [1,2,3,4], 2,  np.random.rand(3,3), "Weights tensor must have last dimension of length 2; got 3"),
            (circuit1, 2, [1,2,3,4], 2,  np.random.rand(4,2), "Weights tensor must have first dimension of length 3; got 4"),
        ],
    )
    def test_exception_wrong_weight_shape(self, block, n_params_block, wires, loc, block_weights, msg_match):
        """Verifies that an exception is raised if the weights shape is incorrect."""
        with pytest.raises(ValueError, match=msg_match):
            MPS(wires, loc, block, n_params_block, block_weights)

    # @pytest.mark.parametrize(
    #     ("weights"),
    #     [
    #         ([1,2],),
    #         (circuit1, 2, [1,2,3,4], 2,  np.random.rand(2,4), "Weights tensor must have last dimension of length 3; got 4"),
    #     ],
    # )
    # def test_weights_type(self, weights):
    #     """Verifies that both lists and numpy objects are accepted as weights"""

    # def test_wire_loc_match_warning():
    #     """Tests whether a warning is raised when the number of wires doesn't correspond to loc"""

    # def test_list_array_weights():
        # """Tests whether the template accepts both lists and numpy arrays as weights"""

    

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

