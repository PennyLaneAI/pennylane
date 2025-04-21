# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for reference qubit.
"""

import pytest

import pennylane as qml


@pytest.mark.parametrize(
    "interface",
    (
        pytest.param("autograd", marks=pytest.mark.autograd),
        pytest.param("jax", marks=pytest.mark.jax),
        pytest.param("torch", marks=pytest.mark.torch),
        pytest.param("tensorflow", marks=pytest.mark.tf),
    ),
)
def test_error_on_non_numpy_data(interface):
    """Test that an error is thrown in the interface data is not numpy."""

    x = qml.math.asarray(0.5, like=interface)
    tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.Z(0))])
    dev = qml.device("reference.qubit", wires=1)

    with pytest.raises(ValueError, match="Reference qubit can only work with numpy data."):
        dev.execute(tape)
