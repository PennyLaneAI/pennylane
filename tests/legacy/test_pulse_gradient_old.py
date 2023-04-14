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
"""Tests for the gradients.pulse_gradient module.
Note that the module is implemented for the new return type system
so that this test suite just tests for correctly raised errors."""
# pylint:disable=import-outside-toplevel
import pytest
import pennylane as qml
from pennylane.gradients.pulse_gradient import stoch_pulse_grad


@pytest.mark.jax
def test_stoch_pulse_grad_raises():
    """Test that stoch_pulse_grad raises a NotImplementedError."""
    tape = qml.tape.QuantumScript()
    with pytest.raises(NotImplementedError, match="The stochastic pulse parameter-shift"):
        qml.gradients.stoch_pulse_grad(tape)


def test_stoch_pulse_grad_raises_without_jax_installed():
    """Test that an error is raised if a stoch_pulse_grad is called without jax installed"""
    try:
        import jax  # pylint: disable=unused-import

        pytest.skip()
    except ImportError:
        tape = qml.tape.QuantumScript([], [])
        with pytest.raises(ImportError, match="Module jax is required"):
            stoch_pulse_grad(tape)
