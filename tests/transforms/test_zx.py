# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the `pennylane.transforms.zx` folder.
"""

import pytest
import pyzx
import pennylane as qml

pytestmark = pytest.mark.zx

with qml.tape.QuantumTape() as tape:
    qml.RX(0.432, wires=0)
    qml.RY(0.543, wires="a")
    qml.CNOT(wires=[0, "a"])
    qml.CRZ(0.5, wires=["a", 0])
    qml.RZ(0.240, wires=0)
    qml.RZ(0.133, wires="a")
    qml.expval(qml.PauliZ(wires=[0]))


class TestConverters:
    """Test converters tape_to_graph_zx and graph_zx_to_tape."""

    def test_tape_to_graph_zx_simple_tape(self):
        """Test the tape to graph zx tape."""
        g = qml.transforms.zx.tape_to_graph_zx(tape)
        assert isinstance(g, pyzx.graph.base.BaseGraph)
        t = qml.transforms.zx.graph_zx_to_tape(g)
        assert isinstance(t, qml.tape.QuantumTape)
