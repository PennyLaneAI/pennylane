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
"""Functions and variables to be utilized by qutrit mixed state simulator."""
from pennylane import math

QUDIT_DIM = 3  # specifies qudit dimension


def reshape_state_as_matrix(state, num_wires):
    """Given a non-flat, potentially batched state, flatten it to square matrix or matrices if batched.

    Args:
        state (TensorLike): A state that needs to be reshaped to a square matrix or matrices if batched
        num_wires (int): The number of wires the state represents

    Returns:
        Tensorlike: A reshaped, square state, with an extra batch dimension if necessary
    """
    dim = QUDIT_DIM**num_wires
    batch_size = math.get_batch_size(state, ((QUDIT_DIM,) * (num_wires * 2)), dim**2)
    shape = (batch_size, dim, dim) if batch_size is not None else (dim, dim)
    return math.reshape(state, shape)


def get_num_wires(state, is_state_batched: bool = False):
    """Finds the number of wires associated with a state

    Args:
        state (TensorLike): A device compatible state that may or may not be batched
        is_state_batched (int): Boolean representing whether the state is batched or not

    Returns:
        int: Number of wires associated with state
    """
    len_row_plus_col = len(math.shape(state)) - is_state_batched
    return int(len_row_plus_col / 2)
