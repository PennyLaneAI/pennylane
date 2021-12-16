# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the `pennylane.draw_text` function.
"""

import pytest
import pennylane as qml

from pennylane.circuit_drawer import draw_text
from pennylane.circuit_drawer.draw_text import _add_op
from pennylane.tape import QuantumTape

default_wire_map = {0:0, 1:1, 2:2, 3:3}

add_op_data = [(qml.PauliX(0), ["─X","─","─","─"]),
(qml.CNOT(wires=(0,2)), ['╭C', '│', '╰X', '─']),
(qml.Toffoli(wires=(0,1,3)), ['╭C', '├C', '│', '╰X'])
(qml.IsingXX(1.23, wires=(0,2)), ['╭IsingXX', '│', '╰IsingXX', '─'])
]

add_op_data2 = [
    (qml.PauliY(1), ['─X', '─Y', '─', '─']),
    (qml.CNOT(wires=(1,2)), ['─X', '╭C', '╰X', '─']),
    (qml.CRX(1.23, wires=(2,3)), ['─X', '─', '╭C', '╰RX'])
]

class TestAddOp:

    @pytest.parametrize("op, out", add_op_data)
    def test_add_op(self, op, out):
        """Test adding the first operation to array of strings"""
        assert out == _add_op(op, ["─"]*4, default_wire_map, None)

    @pytest.parametrize("op, out", add_op_data2)
    def test_add_second_op(self, op, out):
        """Test adding a second operation to the array of strings"""
        start = _add_op(qml.PauliX(0),  ["─"]*4, default_wire_map, None)
        assert out == _add_op(op, start, default_wire_map, None)