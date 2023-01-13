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
Unit tests for the :mod:`pennylane.plugin.PulseQubit` device.
"""

import pytest
import numpy as np
import pennylane as qml


def test_initialization():
    device = qml.DefaultPulse(
        wires=[0, 1], shots=1000, dt=1e-3, dim=3, drift=2 * qml.PauliX(0) + qml.PauliY(1)
    )
    assert device.shots == 1000
    assert device.dt == 1e-3
    assert device.dim == 3
    assert qml.equal(device.drift, 2 * qml.PauliX(0) + qml.PauliY(1))

    assert device.r_dtype == np.float64
    assert device.c_dtype == np.complex128
