# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Assertion test for multi_dispatch function/decorator
"""
import autoray
import numpy as onp
import pytest
from pennylane import numpy as np
from pennylane import math as fn


tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")

test_multi_dispatch_stack_data = [
    [[1.0, 0.0], [2.0, 3.0]],
    ([1.0, 0.0], [2.0, 3.0]),
    onp.array([[1.0, 0.0], [2.0, 3.0]]),
    np.array([[1.0, 0.0], [2.0, 3.0]]),
    # torch.tensor([[1.,0.],[2.,3.]]),
    tf.Variable([[1.0, 0.0], [2.0, 3.0]]),
    tf.constant([[1.0, 0.0], [2.0, 3.0]]),
]


@pytest.mark.parametrize("x", test_multi_dispatch_stack_data)
def test_multi_dispatch_stack(x):
    """Test that the decorated autoray function stack can handle all inputs"""
    stack = fn.multi_dispatch(argnum=0, tensor_list=0)(autoray.numpy.stack)
    res = stack(x)
    print(res)
    assert fn.allequal(res, [[1.0, 0.0], [2.0, 3.0]])
