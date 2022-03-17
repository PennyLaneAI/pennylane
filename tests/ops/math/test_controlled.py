# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=too-many-arguments

import pytest

import pennylane as qml
from pennylane.wires import Wires
from pennylane.ops.math import Controlled

def test_default_property_assignment():

    base_op = qml.PauliX(0)
    op = qml.Controlled(base_op, control_wires=1, id="hi")

    assert op.num_wires == qml.operation.AnyWires

    assert op.base == base_op

    assert op.control_wires == Wires(1)
    assert op.control_values == [1]

    assert op.work_wires == Wires([])
    assert op.id == "hi"

    assert op.data == []

    assert op.hyperparameters == {'control_wires': Wires(1),
        'control_values': [1],
        'base': base_op,
        'work_wires': Wires([])}
    
    assert op.name == "C(PauliX)"
    assert str(op) == 'C(PauliX)(wires=[1, 0])'
    assert repr(op) == 'C(PauliX)(wires=[1, 0])'


def test_queueing():

    with qml.tape.QuantumTape() as tape:
        base = qml.PauliX(0)
        op = Controlled(base, 1)

    assert tape.operations == [op]
    assert len(tape._queue) == 2
    assert tape._queue[base] == {'owner': op}
    assert tape._queue[op] == {'owns': base}

@pytest.mark.parametrize("control_wires", (0, (0,1), (0,1,2)))
def test_matrix(control_wires):

    base = qml.RX(1.234, wires=3)
    op = Controlled(base, control_wires)
    mat = op.get_matrix()

    assert qml.math.allclose(mat[-2:, -2:], base.get_matrix)
    assert qml.math.allclose(mat[0:-2, 0:-2], qml.numpy.eye(len(control_wires)))