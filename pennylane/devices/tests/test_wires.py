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
"""Tests that a device has the right attributes, arguments and methods."""
# pylint: disable=no-self-use,too-many-arguments,too-few-public-methods
import pytest
import pennylane as qml
from pennylane import numpy as np


# ===== Factories for circuits using arbitrary wire labels and numbers


def make_simple_circuit_expval(device, wires):
    """Factory for a qnode returning expvals."""

    n_wires = len(wires)

    @qml.qnode(device)
    def circuit():
        qml.RX(0.5, wires=wires[0 % n_wires])
        qml.RY(2.0, wires=wires[1 % n_wires])
        if n_wires > 1:
            qml.CNOT(wires=[wires[0], wires[1]])
        return [qml.expval(qml.Z(w)) for w in wires]

    return circuit


# =====


# pylint: disable=too-few-public-methods
class TestWiresIntegration:
    """Test that the device integrates with PennyLane's wire management."""

    @pytest.mark.parametrize(
        "wires1, wires2",
        [
            (["a", "c", "d"], [2, 3, 0]),
            ([-1, -2, -3], ["q1", "ancilla", 2]),
            (["a", "c"], [3, 0]),
            ([-1, -2], ["ancilla", 2]),
            (["a"], ["nothing"]),
        ],
    )
    @pytest.mark.parametrize("circuit_factory", [make_simple_circuit_expval])
    def test_wires_expval(
        self, device, circuit_factory, wires1, wires2, tol
    ):  # pylint: disable=too-many-arguments
        """Test that the expectation of a circuit is independent from the wire labels used."""
        dev1 = device(wires1)
        dev2 = device(wires2)

        circuit1 = circuit_factory(dev1, wires1)
        circuit2 = circuit_factory(dev2, wires2)

        assert np.allclose(circuit1(), circuit2(), atol=tol(dev1.shots))
