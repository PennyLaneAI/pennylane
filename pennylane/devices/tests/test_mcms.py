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
"""
Tests that devices can handle mid circuit measurements.
"""
import numpy as np

import pennylane as qml


def test_simple_mcm_present(device, tol):
    """Test that the device can execute a circuit with a mid circuit measurement."""

    dev = device(wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.measure(0)
        return qml.expval(qml.Z(0))

    res = circuit()
    assert qml.math.allclose(res, 1, atol=tol(dev.shots))


def test_mcm_conditional(device, tol):
    """Test that the device execute a circuit with an MCM and a conditional."""

    dev = device(wires=2)

    @qml.qnode(dev)
    def circuit(x):
        m0 = qml.measure(0)
        qml.cond(~m0, qml.RX)(x, 0)
        return qml.expval(qml.Z(0))

    res = circuit(0.5)
    assert qml.math.allclose(res, np.cos(0.5), atol=tol(dev.shots))
