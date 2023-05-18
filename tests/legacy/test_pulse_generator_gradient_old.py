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
"""Tests for the gradients.pulse_generator module.
Note that the module is implemented for the new return type system
so that this test suite just tests for correctly raised errors."""
# pylint:disable=import-outside-toplevel
import pytest
import pennylane as qml
from pennylane.gradients.pulse_generator_gradient import pulse_generator


@pytest.mark.jax
def test_pulse_generator_raises():
    """Test that pulse_generator raises a NotImplementedError."""
    tape = qml.tape.QuantumScript()
    with pytest.raises(NotImplementedError, match="The pulse generator parameter-shift"):
        _ = pulse_generator(tape)
