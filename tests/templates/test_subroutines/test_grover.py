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
"""
Tests for the Grover Diffusion Operator template
"""
import pytest
import numpy as np
import pennylane as qml


def test_work_wires():
    """Assert work wires get passed to MultiControlledX"""
    wires = ("a", "b")
    work_wire = ("aux",)

    op = qml.templates.GroverOperator(wires=wires, work_wires=work_wire)

    assert op.work_wires == work_wire

    ops = op.expand().operations

    assert ops[2]._work_wires == work_wire


@pytest.mark.parametrize("bad_wires", [0, (0,), tuple()])
def test_single_wire_error(bad_wires):
    """Assert error raised when called with only a single wire"""

    with pytest.raises(ValueError, match="GroverOperator must have at least"):
        op = qml.templates.GroverOperator(wires=bad_wires)


def test_do_queue():
    """Assert do_queue=False is not queued"""

    with qml.tape.QuantumTape() as tape:
        qml.templates.GroverOperator(wires=(0, 1), do_queue=False)

    assert len(tape.operations) == 0


def test_id():
    """Assert id keyword works"""

    op = qml.templates.GroverOperator(wires=(0, 1), id="hello")

    assert op.id == "hello"


decomp_3wires = [
    qml.Hadamard,
    qml.Hadamard,
    qml.PauliZ,
    qml.MultiControlledX,
    qml.PauliZ,
    qml.Hadamard,
    qml.Hadamard,
]


def decomposition_wires(wires):
    wire_order = [
        wires[0],
        wires[1],
        wires[2],
        wires,
        wires[2],
        wires[0],
        wires[1],
    ]
    return wire_order


@pytest.mark.parametrize("wires", ((0, 1, 2), ("a", "c", "b")))
def test_expand(wires):
    """Asserts decomposition uses expected operations and wires"""
    op = qml.templates.GroverOperator(wires=wires)

    decomp = op.expand().operations

    expected_wires = decomposition_wires(wires)

    for actual_op, expected_class, expected_wires in zip(decomp, decomp_3wires, expected_wires):
        assert isinstance(actual_op, expected_class)
        assert actual_op.wires == qml.wires.Wires(expected_wires)


def test_findstate():
    """Asserts can find state marked by oracle."""
    wires = range(6)

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circ():
        for wire in wires:
            qml.Hadamard(wire)

        for _ in range(5):
            qml.Hadamard(wires[0])
            qml.MultiControlledX(wires=wires[0], control_wires=wires[1:])
            qml.Hadamard(wires[0])
            qml.templates.GroverOperator(wires=wires)

        return qml.probs(wires=wires)

    probs = circ()

    assert np.argmax(probs) == len(probs) - 1
