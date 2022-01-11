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
import pennylane as qml
import autoray
import numpy as onp
import pytest

def test_multi_dispatch():
    x = [[0,1],[0,2]]
    stack = qml.math.multi_dispatch.multi_dispatch(argnum=0, tensor_list=0)(autoray.numpy.stack)
    result = stack(x)
    expected = onp.stack(x)
    assert (result == expected).all()
