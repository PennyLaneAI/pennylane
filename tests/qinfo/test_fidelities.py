# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for differentiable quantum entropies.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")


class TestFidelityQnode:
    """Tests for Fidelity function between two QNodes ."""

    devices = ["default.qubit", "default.mixed"]

    @pytest.mark.parametrize("device", devices)
    def test_not_same_number_wires(self, device):
        """Test that wires must have the same length"""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0():
            return qml.state()

        @qml.qnode(dev)
        def circuit1():
            return qml.state()

        with pytest.raises(
            qml.QuantumFunctionError, match="The two states must have the same number of wires"
        ):
            qml.qinfo.fidelity(circuit0, circuit1, wires0=[0, 1], wires1=[0])()
