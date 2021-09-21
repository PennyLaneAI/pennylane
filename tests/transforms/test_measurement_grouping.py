# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
import pennylane as qml


def test_measurement_grouping():
    """Test that measurement grouping works as expected."""

    with qml.tape.QuantumTape() as tape:
        qml.RX(0.1, wires=0)
        qml.RX(0.2, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])

    obs = [qml.PauliZ(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliX(2)]
    coeffs = [2.0, -0.54, 0.1]

    tapes, fn = qml.transforms.measurement_grouping(tape, obs, coeffs)
    assert len(tapes) == 2

    dev = qml.device("default.qubit", wires=3)
    res = fn(dev.batch_execute(tapes))
    assert np.isclose(res, 2.0007186)
