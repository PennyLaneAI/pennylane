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
"""
Unit tests for ensuring that objects that are pickled/unpickled are identical to the original.
"""

import pickle

import pennylane as qml


def test_unpickling_tensor():
    """Tests whether qml.numpy.tensor objects are pickleable."""

    x = qml.numpy.random.random(15)
    x_str = pickle.dumps(x)
    x_reloaded = pickle.loads(x_str)

    assert qml.numpy.allclose(x, x_reloaded)
    assert x.__dict__ == x_reloaded.__dict__
    assert hasattr(x_reloaded, "requires_grad")
