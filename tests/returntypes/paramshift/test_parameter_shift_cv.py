# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the gradients.parameter_shift_cv module."""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import param_shift_cv


@pytest.mark.parametrize(
    "gradient_recipes",
    [None, ([[1 / np.sqrt(2), 1, np.pi / 4], [-1 / np.sqrt(2), 1, -np.pi / 4]],)],
)
def test_error(gradient_recipes):
    """Test the gradient raises an error with the new return type."""
    dev = qml.device("default.gaussian", wires=2, hbar=2)

    alpha = 0.5643
    theta = 0.23354

    with qml.queuing.AnnotatedQueue() as q:
        qml.Displacement(alpha, 0.0, wires=[0])
        qml.Rotation(theta, wires=[0])
        qml.expval(qml.X(0))

    tape = qml.tape.QuantumScript.from_queue(q)
    tape.trainable_params = {2}

    with pytest.raises(
        ValueError,
        match="The parameter shift gradient for CV devices only work with the old return types.",
    ):
        param_shift_cv(tape, dev, gradient_recipes=gradient_recipes)