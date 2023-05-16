# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unittests for the _grad module.
"""
import pytest

import pennylane as qml


def test_return_types_error_caught():
    """Test that an informative error message is raised if an error occurs
    inside qml.jacobian."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, 0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    x = qml.numpy.array(0.1)
    with pytest.raises(ValueError, match=r"PennyLane has a new return shape specification"):
        qml.jacobian(circuit)(x)


def test_same_error_if_no_return_types():
    """Tests that the error is not changed if return types are disabled."""
    qml.disable_return()

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        raise TypeError

    x = qml.numpy.array(0.1)
    with pytest.raises(TypeError):
        qml.jacobian(circuit)(x)
    qml.enable_return()
